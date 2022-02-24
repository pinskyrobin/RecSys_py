python  PRM_test.py ^
        --train true ^
        --train_set dataset/rec_train_set ^
        --validation_set dataset/rec_validation_set ^
        --model_type 2 ^
        --batch_size 128 ^
        --train_epochs 200 ^
        --train_steps_per_epoch 2000 ^
        --validation_steps 1000 ^
        --early_stop_patience 100 ^
        --lr_per_step 4000 ^
        --d_feature 24 ^
        --saved_model_name trained_model/PRM_model.h5

python PRM_test.py ^
        --test_set dataset/rec_test_set ^
        --batch_size 2 ^
        --model_type 2 ^
        --saved_model_name trained_model/PRM_model.h5 ^
        --d_feature 24

python PRM_evaluate.py