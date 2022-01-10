import numpy as np
import pickle
# from utils import UserSimilarity
from operator import itemgetter


def recommend(user_id, user_movie, rating_map, k, rank_num):
    rank = {}
    # W = UserSimilarity()
    with open("weights_users.pkl", "rb") as tf:
        W = pickle.load(tf)
    # 当前用户感兴趣的电影集合
    interacted_movies = user_movie[user_id]

    # 按权重排序,取与当前用户最相近(Wuv)的k个用户(v)

    for v, Wuv in sorted(W[user_id].items(), key=lambda x: x[1], reverse=True)[0:k]:
        # 相近用户所有电影评分信息
        for movie, rating in rating_map[v].items():
            # 如果当前电影两个人都看过,则跳过
            if movie in interacted_movies:
                continue

            if movie not in rank:
                rank[movie] = 0
            rank[movie] += Wuv * rating_map[v][movie]

    rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:rank_num]
    return rank


# def evaluate(curr_user_movie, rank):
#     hit = 0
#     for movie in rank.keys():
#         if movie in curr_user_movie.values():
#             hit += 1
#
#     precision = 0
#     recall = 0
