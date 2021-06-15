import os

import numpy as np
import re
import csv

# 絶対パスにする
BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train.tsv')
TEST_PATH = os.path.join(BASE_PATH, 'dataset/test.tsv')
DEV_PATH = os.path.join(BASE_PATH, 'dataset/dev.tsv')


def load_data(num_words=100000, is_id=True):
    """sst-2の学習用とテスト用(devファイル)データを出力する

    sst-2の学習用データの組を出力する
    ラベルの説明 0 bad, 1 good

    Args:
        bool: numpyとして出力するか
        
    Returns:
        ([string], [int]), ([string], [int]): (学習用データ, ラベル),  (テスト用データ, ラベル), word2id, id2word
    """
    
    train_text_data = []
    train_label_data = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter='\t')
        texts.__next__()
        for row in texts:
            train_text_data.append(row[0].lower())
            train_label_data.append(int(row[1]))

    test_text_data = []
    test_label_data = []
    with open(DEV_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter='\t')
        texts.__next__()
        for row in texts:
            test_text_data.append(row[0].lower())
            test_label_data.append(int(row[1]))

    if not is_id:
        return (np.array(train_text_data), np.array(train_label_data)), (np.array(test_text_data), np.array(test_label_data))

    tokens = []
    for text in train_text_data:
        tokens.extend(text.split())

    vocab, word2id, index = {}, {}, 1
    vocab['<pad>'] = 0
    word2id['<pad>'] = 0
    for token in tokens:
        if token not in vocab:
            if index < num_words:
                vocab[token] = index
                word2id[token] = index
                index += 1
            else:
                word2id[token] = 0


    train_data = [[word2id[word] for word in sentence.split()] for sentence in train_text_data]

    id2word = {index: word for index, word in enumerate(word2id)}

    tokens = []
    for text in test_text_data:
        tokens.extend(text.split())

    for token in tokens:
        if token not in vocab:
            word2id[token] = 0

    test_data = [[word2id[word] for word in sentence.split()] for sentence in test_text_data]
    
    
    return (np.array(train_data), np.array(train_label_data)), (np.array(test_data), np.array(test_label_data)), vocab, word2id, id2word
    

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test), vocab, word2id, id2word= load_data(num_words=10000)

    print(len(vocab))

    for i in range(10):
        print("train: {} => label: {}".format(x_train[i], y_train[i]))
        print("test: {} => label: {}".format(x_test[i], y_test[i]))
