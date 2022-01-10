import math
import pickle

from read_dataset import get_um_map


# 计算用户之间的相似度
def UserSimilarity():
    movie_user = get_um_map(0)
    C = {}
    N = {}

    for movie, users in movie_user.items():
        for u in users:
            if u not in N:
                N[u] = 0
            if u not in C:
                C[u] = {}
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                if v not in C[u]:
                    C[u][v] = 0
                C[u][v] += 1
    W = {}

    for u, related_users in C.items():
        if u not in W:
            W[u] = {}
        for v, Cuv in related_users.items():
            W[u][v] = Cuv / math.sqrt(N[u] * N[v])

    with open("weights_users.pkl", "wb") as tf:
        pickle.dump(W, tf)
    # return W


# 计算电影之间的相似度
def ItemSimilarity():
    N = {}
    C = {}
    user_movie = get_um_map(1)

    for u, movies in user_movie.items():
        for m in movies:
            if m not in N:
                N[m] = 0
            if m not in C:
                C[m] = {}
            N[m] += 1
            for n in movies:
                if m == n:
                    continue
                if n not in C[m]:
                    C[m][n] = 0
                C[m][n] += 1

    W = {}

    for i, related_movies in C.items():
        if i not in W:
            W[i] = {}
        for j, Cij in related_movies.items():
            if j not in W[i]:
                W[i][j] = 0
            W[i][j] = Cij / math.sqrt(N[i] * N[j])

    with open("weights_movies.pkl", "wb") as tf:
        pickle.dump(W, tf)


if __name__ == '__main__':
    UserSimilarity()
    ItemSimilarity()
