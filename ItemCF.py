import pickle


def recommend(user_id, user_movie, rating_map, k, rank_num):
    rank = {}
    # W = UserSimilarity()
    with open("../rs_movie_py copy/weights_movies.pkl", "rb") as tf:
        W = pickle.load(tf)

    # 当前用户感兴趣的电影集合
    interacted_movies = user_movie[user_id]

    # 这一循环获取用户看过的电影及其评分
    for i, rating in rating_map[user_id].items():
        # 这一循环获取与这只电影最相近(Wij)的k部电影(j)
        for j, Wij in sorted(W[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
            if j in interacted_movies:
                continue
            if j not in rank:
                rank[j] = 0
            # 计算新电影的推荐程度
            rank[j] += rating * Wij

    rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:rank_num]
    return rank
