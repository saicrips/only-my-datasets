import os

import numpy as np
import re
import csv

BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train.csv')
TEST_PATH = os.path.join(BASE_PATH, 'dataset/test.csv')

def load_train_data():
    """ag_newsの学習用データを出力する

    ag_newsの学習用データの組を出力する
    ラベルの説明 1 World, 2 Sports, 3 Business, 4 Sci/Tech

    Returns:
        ([string], [int]): (学習用データ, ラベル)
    """

    train_data = []
    train_label_data = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            train_data.append(text)
            train_label_data.append(int(row[0]))

    return (train_data, train_label_data)

def load_test_data():
    """ag_newsのテスト用データを出力する

    ag_newsのテスト用データの組を出力する
    ラベルの説明 1 World, 2 Sports, 3 Business, 4 Sci/Tech

    Returns:
        ([string], [int]): (テスト用データ, ラベル)
    """

    test_data = []
    test_label_data = []
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            test_data.append(text)
            test_label_data.append(int(row[0]))

    return (test_data, test_label_data)

def load_data():
    """ag_newsの学習用テスト用データを出力する

    ag_newsの学習用テスト用データの組を出力する
    ラベルの説明 1 World, 2 Sports, 3 Business, 4 Sci/Tech

    Returns:
        ([string], [int]), ([string], [int]): (学習用データ, ラベル),  (テスト用データ, ラベル)
    """

    return load_train_data(), load_test_data()

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = load_data()

    for i in range(10):
        print("text: {} => label: {}".format(x_train[i], y_train[i]))
