import sys
import os
BASE_PATH = os.path.dirname(__file__)
from preprocessing import utils

def preprocess(text):
    cleaned_text = utils.clean_text(text)
    normalized_text = utils.normalize_number(cleaned_text)
    wakati_text = utils.wakati_list(normalized_text)
    
    path = 'stopwords_list.txt'
    path = os.path.join(BASE_PATH, path)
    
    stopwords_list = utils.stopwords_list(path)
    output = []
    for word in wakati_text:
        if word in stopwords_list:
            continue
        output.append(word)

    return output