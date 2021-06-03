import os

import numpy as np
import re
import csv

# 絶対パスにする
BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train.tsv')
TEST_PATH = os.path.join(BASE_PATH, 'dataset/test.tsv')
DEV_PATH = os.path.join(BASE_PATH, 'dataset/dev.tsv')

def load_train_data(numpy=True):
    """sst-2の学習用データを出力する

    sst-2の学習用データの組を出力する
    ラベルの説明 0 bad, 1 good

    Args:
        bool: numpyとして出力するか

    Returns:
        ([string], [int]): (学習用データ, ラベル)
    """

    train_data = []
    train_label_data = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter='\t')
        texts.__next__()
        for row in texts:
            train_data.append(row[0])
            train_label_data.append(int(row[1]))

    if numpy:
        return (np.array(train_data), np.array(train_label_data))

    return (train_data, train_label_data)

def load_test_data(numpy=True):
    """sst-2のテスト用(devファイル)データを出力する

    sst-2の学習用データの組を出力する
    ラベルの説明 0 bad, 1 good

    Args:
        bool: numpyとして出力するか

    Returns:
        ([string], [int]): (テスト用データ, ラベル)
    """

    test_data = []
    test_label_data = []
    with open(DEV_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter='\t')
        texts.__next__()
        for row in texts:
            test_data.append(row[0])
            test_label_data.append(int(row[1]))

    if numpy:
        return (np.array(test_data), np.array(test_label_data))

    return (test_data, test_label_data)

def load_data(numpy=True):
    """sst-2の学習用とテスト用(devファイル)データを出力する

    sst-2の学習用データの組を出力する
    ラベルの説明 0 bad, 1 good

    Args:
        bool: numpyとして出力するか
        
    Returns:
        ([string], [int]), ([string], [int]): (学習用データ, ラベル),  (テスト用データ, ラベル)
    """
    
    (train_data, train_label_data) = load_train_data(numpy)

    (test_data, test_label_data) = load_test_data(numpy)


    if numpy:
        return (np.array(train_data), np.array(train_label_data)), (np.array(test_data), np.array(test_label_data))

    return (train_data, train_label_data), (test_data, test_label_data)

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = load_data()
    #x_train, y_train = load_train_data()

    for i in range(10):
        print("train: {} => label: {}".format(x_train[i], y_train[i]))
        print("test: {} => label: {}".format(x_test[i], y_test[i]))
