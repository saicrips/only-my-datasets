"""livedoor dataset loader

livedoor ニュースコーパスのデータセットを読み込む

available: https://www.rondhuit.com/download.html  livedoor ニュースコーパス

    - トピックニュース
        http://news.livedoor.com/category/vender/news/
    - Sports Watch
        http://news.livedoor.com/category/vender/208/
    - ITライフハック
        http://news.livedoor.com/category/vender/223/
    - 家電チャンネル
        http://news.livedoor.com/category/vender/kadench/
    - MOVIE ENTER
        http://news.livedoor.com/category/vender/movie_enter/
    - 独女通信
        http://news.livedoor.com/category/vender/90/
    - エスマックス
        http://news.livedoor.com/category/vender/smax/
    - livedoor HOMME
        http://news.livedoor.com/category/vender/homme/
    - Peachy
        http://news.livedoor.com/category/vender/ldgirls/


トピックごとにデータを収集することを想定する

1 トピックニュース  /topic-news,
2 Sports Watch   /sports-watch,
3 ITライフハック   /it-life-hack,
4 家電チャンネル   /kaden-channel,
5 MOVIE ENTER    /movie-enter,
6 独女通信        /dokujo-tsushin,
7 エスマックス    /smax,
8 livedoor HOMME /livedoor-homme,
9 Peachy        /peachy
"""

import os
import sys
sys.path.append("../../only-my-datasets/")
import glob

import numpy as np
import re
import csv

from preprocessing import preprocess

BASE_PATH = os.path.dirname(__file__)
TOPIC_NUM = 9
TOPICS = ['dataset/topic-news/',
          'dataset/sports-watch/',
          'dataset/it-life-hack/', 
          'dataset/kaden-channel/',
          'dataset/movie-enter/',
          'dataset/dokujo-tsushin/',
          'dataset/smax/',
          'dataset/livedoor-homme/',
          'dataset/peachy/']
DATASET_PATHS = []
for topic in TOPICS:
    DATASET_PATHS.append(os.path.join(BASE_PATH, topic))

def load_plain_data(print_info=True):
    """
    Returns:
        {
            1 : [[title, sentence], [title, sentence],...],
            2 : [[title, sentence], [title, sentence],...],
            :
            :
            9 : [[title, sentence], [title, sentence],...]
        }
    """
    data = {}
    for i, path in enumerate(DATASET_PATHS):
        file_paths = glob.glob(path + '*')
        topic_data = []
        for file_path in file_paths:
            if file_path[-11:] == 'LICENSE.txt':
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                file_list = f.readlines()
                title = file_list[2] # 3行目にタイトルがある
                text = ''
                for s in file_list[3:]: # 4行目以降に本文がある
                    text = text + s
                file_data = [title, text]
            topic_data.append(file_data)
        data[i+1] = topic_data

    if print_info:
        for i in range(TOPIC_NUM):
            print("topic {} => {} articles".format(i+1, len(data[i+1])))
    return  data


def load_data(num_words=57319, data_size=500):
    """livedoor dataset load
    トピックごとファイルごとの配列でデータを出力する
    
    1 トピックニュース
    2 Sports Watch
    3 ITライフハック
    4 家電チャンネル
    5 MOVIE ENTER
    6 独女通信
    7 エスマックス
    8 livedoor HOMME
    9 Peachy

    Args:
        data_size(int): トピックごとの返すデータの個数.
            topic 1 => 770 articles
            topic 2 => 900 articles
            topic 3 => 870 articles
            topic 4 => 864 articles
            topic 5 => 870 articles
            topic 6 => 870 articles
            topic 7 => 870 articles
            topic 8 => 511 articles
            topic 9 => 842 articles

            511まで同じ割合になり, 最大は900個まで　(エラーにはならない)

    Returns:
        それぞれのトピックは500個ずつ取り出し、文章とラベルの配列にする
        (train_data, label_data)
        * シャッフルはされていません
    """

    data = load_plain_data(print_info=False)

    train_data = []
    for i in range(TOPIC_NUM):
        for j, file_data in enumerate(data[i+1]):
            if j >= data_size:
                break
            train_data.append([file_data[1], i+1])

    train_data = np.array(train_data)
    train_text_data, train_label = train_data.T
    train_label = train_label.astype(np.int32)

    train_data = [preprocess(sentence) for sentence in train_text_data]  

    vocab, word2id, index = {}, {}, 1
    vocab['<pad>'] = 0
    word2id['<pad>'] = 0
    tokens = []
    for sentence in train_data:
    	tokens.extend(sentence)
    
    for token in tokens:
        if token not in vocab:
            if index < num_words:
                vocab[token] = index
                word2id[token] = index
                index += 1
            else:
                word2id[token] = 0
	
    train_data = [[word2id[word] for word in sentence] for sentence in train_data]

    id2word = {index: word for index, word in enumerate(word2id)}

    return (train_data, train_label), vocab, word2id, id2word

def id2sentence(encoded_text_list, id2word):
    word_list = []
    for text in encoded_text_list:
        word_list.append(id2word[text])
    return ' '.join(word_list)


if __name__ == '__main__':
    (train_data, label), vocab, word2id, id2word = load_data(num_words=51555)

    for i in range(10):
        print(id2sentence(train_data[i], id2word))

    print(len(vocab))