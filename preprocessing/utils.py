import os
import re
import urllib.request
import MeCab

def wakati_list(text, pos_list=["名詞", "動詞", "形容詞"]):
    tagger = MeCab.Tagger('')
    tagger.parse('')
    node = tagger.parseToNode(text)
    output = []
    while node:
        pos = node.feature.split(",")[0]
        if pos in pos_list:
            word = node.surface
            output.append(word)
        node = node.next
    return output

def clean_text(text):
    replaced_text = text.lower()
    replaced_text = re.sub(r'[【】]', ' ', replaced_text)       # 【】の除去
    replaced_text = re.sub(r'[（）()]', ' ', replaced_text)     # （）の除去
    replaced_text = re.sub(r'[［］\[\]]', ' ', replaced_text)   # ［］の除去
    replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
    replaced_text = re.sub(r'https?:\/\/.*?[\r\n ]', '', replaced_text)  # URLの除去
    replaced_text = re.sub(r'　', ' ', replaced_text)  # 全角空白の除去
    return replaced_text

def download_stopwords(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)

def normalize_number(text):
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

def stopwords_list(file_path):
    stop_words = []
    for w in open(file_path, 'r'):
        w = w.replace('\n', '')
        if len(w) > 0:
            stop_words.append(w)
    return stop_words