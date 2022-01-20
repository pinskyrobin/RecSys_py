import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from operator import itemgetter
from utils import recall, precision, coverage

from MyALS.MyALS import MyALS

data = np.load("../ml-latest-small/ratings.npy", allow_pickle=True)
data = np.delete(data, -1, axis=1)

X, Y = train_test_split(data, train_size=0.6, random_state=0)
Y = np.delete(Y, -1, axis=1)

movies = pd.DataFrame(np.load("../ml-latest-small/movies.npy", allow_pickle=True))


def get_items_list(raw_data):
    raw_data = sorted(raw_data, key=(lambda x: x[0]))
    item_list = []
    tmp_list = []
    for item in raw_data:
        if int(item[0]) <= len(item_list) + 1:
            tmp_list.append(int(item[1]))
            continue
        item_list.append(tmp_list)
        tmp_list = []
    item_list.append(tmp_list)
    return item_list


def get_prediction(predictions):
    train_recs = []
    for items in predictions:
        train_recs.append(list(map(int, map(itemgetter(0), items))))
    test = get_items_list(Y)
    return train_recs, test


def evaluate(train_recs, test):
    print("recall:{}".format(recall(train_recs, test)))
    print("precision:{}".format(precision(train_recs, test)))
    print("coverage:{}".format(coverage(train_recs, test)))


def print_predictions(user_ids, predictions):
    for user_id, prediction in zip(user_ids, predictions):
        _prediction = []
        for item_id, score in prediction:
            item_name = movies[movies[0] == item_id][1].item()
            _prediction.append(item_name)
            # _prediction.append([item_name, score])
        print("user_id:%d recommedation: %s" % (user_id, _prediction))


def test01():
    print("with reg_params")
    model = MyALS(rank=10, max_iter=10, reg_param=3, early_stop_enable=True)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test02():
    print("with different reg_params")
    model = MyALS(rank=10, max_iter=10, reg_param=1, early_stop_enable=True)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test03():
    print("with different reg_params")
    model = MyALS(rank=10, max_iter=10, reg_param=0.1, early_stop_enable=True)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test04():
    print("with different rank")
    model = MyALS(rank=20, max_iter=10, reg_param=1, early_stop_enable=True)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test05():
    print("with different rank")
    model = MyALS(rank=5, max_iter=10, reg_param=1, early_stop_enable=True)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test06():
    print("without reg_params")
    model = MyALS(rank=10, max_iter=10)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)


def test07():
    print("without early stop")
    model = MyALS(rank=20, max_iter=10, reg_param=1, early_stop_enable=False)
    model.fit(X)
    train_recs, test = get_prediction(model.predict(range(1, 611), n_items=80))
    evaluate(train_recs, test)
