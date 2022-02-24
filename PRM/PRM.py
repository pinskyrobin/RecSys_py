#! -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
from keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import TimeDistributed, Dense, Embedding, Add, Dropout, MultiHeadAttention, Flatten, \
    Activation, LayerNormalization, Conv1D
from tensorflow.python.keras.models import Model


class PositionalEncoding(Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1:
            max_dims += 1  # max_dims must be even
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]


class PositionWiseFeedForward(Layer):
    def __init__(self, model_dim, inner_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.w_1 = Conv1D(inner_dim, 1, activation='relu')
        self.w_2 = Conv1D(model_dim, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)

    def call(self, inputs, *args, **kwargs):
        output = self.w_1(inputs)
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, inputs])
        return self.layer_norm(output)


class EncoderLayer(Layer):
    def __init__(self, model_dim, inner_dim, n_head, key_dim, value_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.mul_head_attn = MultiHeadAttention(n_head, key_dim=key_dim, value_dim=value_dim, dropout=dropout)
        self.pos_ffn_layer = PositionWiseFeedForward(model_dim, inner_dim)

    def call(self, inputs, *args, **kwargs):
        output, self_attn = self.mul_head_attn(inputs, inputs, return_attention_scores=True)
        # Add & Norm
        output += inputs
        output = LayerNormalization()(output)
        # Feed-Forward
        output = self.pos_ffn_layer(output)
        # Add & Norm
        output += inputs
        output = LayerNormalization()(output)
        return output, self_attn


class Encoder(Layer):
    def __init__(self, model_dim, inner_dim, n_head, key_dim, value_dim, blocks, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = Dropout(dropout)
        self.blocks = [EncoderLayer(model_dim, inner_dim, n_head, key_dim, value_dim, dropout) for _ in range(blocks)]

    def call(self, inputs, return_attention_scores=False, *args, **kwargs):
        output = self.dropout(inputs)
        atts = None
        if return_attention_scores:
            atts = []
        for block in self.blocks:
            output, att = block(output)
            if return_attention_scores:
                atts.append(att)
        return (output, atts) if return_attention_scores else output


class PRM:
    def __init__(self, seq_len, feature_dim, model_dim=64, inner_dim=128, n_head=1, key_dim=64, value_dim=64, blocks=2,
                 dropout=0.1):
        self.model = None
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.model_dim = model_dim
        self.inner_dim = inner_dim
        self.n_head = n_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.blocks = blocks
        self.dropout = dropout

        self.v_input = Input(shape=(seq_len, feature_dim), name='v_input')
        self.pos_input = Input(shape=(seq_len,), dtype='int32', name='pos_input')

    def build(self):
        d0 = TimeDistributed(Dense(self.model_dim))(self.v_input)
        encoder_embeddings = Embedding(self.seq_len, self.model_dim)(self.pos_input)
        pos = PositionalEncoding(self.seq_len, self.model_dim)
        p0 = pos(encoder_embeddings)
        encoder_input = Add()([d0, p0])

        encoder_input = TimeDistributed(Dense(self.model_dim, activation='tanh'))(encoder_input)
        encoder = Encoder(self.model_dim, self.inner_dim, self.n_head,
                          self.key_dim, self.value_dim, self.blocks, self.dropout)

        encoder_output = encoder(encoder_input)

        time_score_dense1 = TimeDistributed(Dense(self.seq_len, activation='tanh'))(encoder_output)
        time_score_dense2 = TimeDistributed(Dense(1))(time_score_dense1)
        flat = Flatten()(time_score_dense2)
        score_output = Activation(activation='softmax')(flat)
        self.model = Model([self.pos_input, self.v_input], score_output)
        return self.model
