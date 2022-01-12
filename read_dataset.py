import numpy as np
import pandas as pd


def process(read_path, save_path):
    data = pd.read_csv(read_path, header=None)
    np.save(save_path, data)


'''
    @:param mode: 1表示正排,其他表示倒排
    @:return: 正排或倒排的字典
'''


def get_um_map(mode=1):
    raw_data = np.load("dataset/ratings.npy", allow_pickle=True)
    mapper = {}
    if mode == 1:
        for item in raw_data:
            # 建立正排表
            if item[0] not in mapper:
                mapper[int(item[0])] = []
            mapper[item[0]].append(int(item[1]))
    else:
        for item in raw_data:
            # 建立倒排表
            if item[1] not in mapper:
                mapper[int(item[1])] = []
            mapper[item[1]].append(int(item[0]))
    return mapper


'''
    获得如下形式的字典
    {
        用户: {
            电影名: 评分
            ...
        }
        ...
    }
'''


def get_rating_map():
    raw_data = np.load("dataset/ratings.npy", allow_pickle=True)
    mapper = {}
    for item in raw_data:
        if item[0] not in mapper:
            mapper[int(item[0])] = {}
        mapper[item[0]][int(item[1])] = item[2]
    return mapper


# process("dataset/movies.csv", "dataset/movies.npy")
# process("dataset/Y.csv", "dataset/Y.npy")
# process("dataset/tags.csv", "dataset/tags.npy")
# get_um_map(1)
