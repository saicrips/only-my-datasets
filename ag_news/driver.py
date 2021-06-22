import os

import numpy as np
import re
import csv

BASE_PATH = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_PATH, 'dataset/train.csv')
TEST_PATH = os.path.join(BASE_PATH, 'dataset/test.csv')


def load_data(num_words=147232, is_id=False):
    """ag_newsの学習用テスト用データを出力する

    ag_newsの学習用テスト用データの組を出力する
    ラベルの説明 1 World, 2 Sports, 3 Business, 4 Sci/Tech

    Args:
        bool: numpyとして出力するか
        
    Returns:
        ([string], [int]), ([string], [int]): (学習用データ, ラベル),  (テスト用データ, ラベル)
    """
    train_data = []
    train_label_data = []
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            train_data.append(re.sub(r'\\', ' ', text.lower()))
            train_label_data.append(int(row[0]))

    test_data = []
    test_label_data = []
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        texts = csv.reader(f, delimiter=',', quotechar='"')
        for row in texts:
            text = ""
            for s in row[1:]:
                    text = text + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
            test_data.append(re.sub(r'\\', ' ', text.lower()))
            test_label_data.append(int(row[0]))

    if is_id:
        return (np.array(train_data), np.array(train_label_data)), (np.array(test_data), np.array(test_label_data))

    tokens = []
    for text in train_data:
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


    train_data = [[word2id[word] for word in sentence.split()] for sentence in train_data]

    id2word = {index: word for index, word in enumerate(word2id)}

    tokens = []
    for text in test_data:
        tokens.extend(text.split())

    for token in tokens:
        if token not in vocab:
            word2id[token] = 0

    test_data = [[word2id[word] for word in sentence.split()] for sentence in test_data]
    
    
    return (np.array(train_data), np.array(train_label_data)), (np.array(test_data), np.array(test_label_data)), vocab, word2id, id2word


def id2sentence(encoded_text_list, id2word):
    word_list = []
    for text in encoded_text_list:
        word_list.append(id2word[text])
    return ' '.join(word_list)


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test), vocab, word2id, id2word = load_data(num_words=147232)

    for i in range(10):
        print("text: {} => label: {}".format(id2sentence(x_train[i], id2word), y_train[i]))

    print(len(vocab))
