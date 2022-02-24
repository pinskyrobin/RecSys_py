"""
    This script is used to transform .csv files(with headers) into .npy files
"""
import numpy as np
import pandas as pd


def csv2npy(csv_path, npy_path):
    data = pd.read_csv(csv_path).to_numpy()
    np.save(npy_path, data)


def transform():
    csv2npy("ml-latest/links.csv", "ml-latest/links.npy")
    csv2npy("ml-latest/movies.csv", "ml-latest/movies.npy")
    csv2npy("ml-latest/ratings.csv", "ml-latest/ratings.npy")
    csv2npy("ml-latest/tags.csv", "ml-latest/tags.npy")

    csv2npy("ml-latest-small/links.csv", "ml-latest-small/links.npy")
    csv2npy("ml-latest-small/movies.csv", "ml-latest-small/movies.npy")
    csv2npy("ml-latest-small/ratings.csv", "ml-latest-small/ratings.npy")
    csv2npy("ml-latest-small/tags.csv", "ml-latest-small/tags.npy")


def test00():
    print("transforming...")
    transform()


def test01():
    print("testing ml-latest-small...")
    test_data = np.load("ml-latest-small/movies.npy", allow_pickle=True)
    print(test_data)
    print(test_data.shape)


def test02():
    print("testing ml-latest...")
    test_data = np.load("ml-latest/ratings.npy", allow_pickle=True)
    print(test_data)
    print(test_data.shape)
