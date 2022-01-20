import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from MyALS.Matrix import Matrix
from random import random


class MyALS:
    def __init__(self, rank=10, max_iter=10, reg_param=0, early_stop_enable=False):
        # Processor params
        self.user_ids = None
        self.item_ids = None
        self.user_ids_dict = None
        self.item_ids_dict = None
        self.user_matrix = None
        self.item_matrix = None
        self.user_items = None
        self.shape = None
        self.rmse = None
        self.es_val = 0.0
        self.es_count = 0
        self.early_stop = False

        # Hyper params
        self.rank = rank
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.early_stop_enable = early_stop_enable

    def _process_data(self, X):

        # Process user ids.
        self.user_ids = tuple((set(map(lambda x: x[0], X))))
        self.user_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.user_ids)))

        # Process item ids.
        self.item_ids = tuple((set(map(lambda x: x[1], X))))
        self.item_ids_dict = dict(map(lambda x: x[::-1],
                                      enumerate(self.item_ids)))

        # The shape of item rating data matrix.
        self.shape = (len(self.user_ids), len(self.item_ids))

        # Sparse matrix and its inverse of item rating data.
        ratings = defaultdict(lambda: defaultdict(int))
        ratings_T = defaultdict(lambda: defaultdict(int))
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating

        return ratings, ratings_T

    def _mul(self, X, Y, tag):
        def f(X_row, id):
            ids = iter(Y[id].keys())
            scores = iter(Y[id].values())
            if tag == 'user':
                col_nos = map(lambda x: self.user_ids_dict[x], ids)
            else:
                col_nos = map(lambda x: self.item_ids_dict[x], ids)
            _X_row = map(lambda x: X_row[x], col_nos)
            return sum(a * b for a, b in zip(_X_row, scores))

        if tag == 'user':
            ret = [[f(X_row, id) for id in self.item_ids]
                   for X_row in X.data]
        else:
            ret = [[f(X_row, id) for id in self.user_ids]
                   for X_row in X.data]

        return Matrix(ret)

    def _get_rmse(self, ratings):

        m, n = self.shape
        mse = 0.0
        n_elements = sum(map(len, ratings.values()))
        for i in range(m):
            for j in range(n):
                user_id = self.user_ids[i]
                item_id = self.item_ids[j]
                rating = ratings[user_id][item_id]
                if rating > 0:
                    user_row = self.user_matrix.col(i).transpose
                    item_col = self.item_matrix.col(j)
                    rating_hat = user_row.mat_mul(item_col).data[0][0]
                    square_error = (rating - rating_hat) ** 2
                    mse += square_error / n_elements
        return mse ** 0.5

    def fit(self, X):

        # Process item rating data.
        ratings, ratings_T = self._process_data(X)
        # Store what X has been viewed by users.
        self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
        # Parameter validation.
        m, n = self.shape

        # Initialize users and X matrix.
        self.user_matrix = Matrix([[random() for _ in range(m)] for _ in range(self.rank)])

        # Minimize the RMSE by EM algorithms.
        for i in range(self.max_iter):
            if i % 2:
                # U = (I * I_transpose) ^ (-1) * I * R_transpose

                items = self.item_matrix
                reg_I = Matrix([([0] * self.item_matrix.shape[0]) for i in range(self.item_matrix.shape[0])]).scala_mul(
                    self.reg_param)
                self.user_matrix = self._mul(
                    # items.mat_mul(items.transpose).inverse.mat_mul(items),
                    items.mat_mul(items.transpose).add(reg_I).inverse.mat_mul(items),
                    ratings,
                    "item"
                )
            else:
                # I = (U * U_transpose) ^ (-1) * U * R
                users = self.user_matrix
                reg_U = Matrix([([0] * self.user_matrix.shape[0]) for i in range(self.user_matrix.shape[0])]).scala_mul(
                    self.reg_param)
                self.item_matrix = self._mul(
                    users.mat_mul(users.transpose).add(reg_U).inverse.mat_mul(users),
                    ratings_T,
                    "user"
                )
            rmse = self._get_rmse(ratings)

            print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))

            # Early stop
            if self.early_stop_enable:
                if abs(rmse - self.es_val) < 0.001:
                    self.es_count += 1
                    if self.es_count >= 3:
                        print("**Early stopped!**")
                        break
                self.es_val = rmse

        # Final RMSE.
        self.rmse = rmse

    def predict(self, user_ids, n_items=10):

        res = []
        for user_id in user_ids:
            # Get column in user_matrix.
            users_col = self.user_matrix.col(self.user_ids_dict[user_id])
            users_col = users_col.transpose
            # Multiply user column with item_matrix.
            items_col = enumerate(users_col.mat_mul(self.item_matrix).data[0])
            # Get the item_id by column index.
            items_scores = map(lambda x: (self.item_ids[x[0]], x[1]), items_col)
            # Filter the item which user has already viewed.
            viewed_items = self.user_items[user_id]
            items_scores = filter(lambda x: x[0] not in viewed_items, items_scores)
            # Get the top n_items by item score.
            res.append(sorted(items_scores, key=lambda x: x[1], reverse=True)[:n_items])

        return res


if __name__ == '__main__':
    data = np.load("/Users/pinskyrobin/OneDrive/Skills/RecommenderSystem/RS_learning/dataset/ratings.npy")
    data = np.delete(data, -1, axis=1)

    X, Y = train_test_split(data, train_size=0.6, random_state=0)
    Y = np.delete(Y, -1, axis=1)

    model = ALS()
    model.fit(X, k=10, max_iter=10, reg_param=3)
