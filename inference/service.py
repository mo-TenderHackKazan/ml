from annoy import AnnoyIndex
import pickle
from numpy import dot
from numpy.linalg import norm
import pandas as pd

import re 

vectorizer = None
ann = AnnoyIndex(631, 'euclidean')
ann.load('search.ann')

with open('tfidf.pickle', 'rb') as file:
    vectorizer = pickle.load(file)

df = pd.read_csv('./prod.csv')



def has_cyrillic(text):
    return bool(re.search("[а-яА-Я]", text))


def cos_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def clean_data(text):
    rus = has_cyrillic(text)
    sp = text.split()
    new_text = ''
    class_category = ''
    for word in sp:
        if '.' in word:
            if len(class_category) > 0 and has_cyrillic(word): continue
            class_category = word
        if rus and not bool(re.search("[а-яА-Я:']", word)): continue
        if '.' in word or ':' in word: continue
        if len(new_text):
            word = ' ' + word
        new_text += word
    return new_text.strip(), class_category


def jsonify_item(item):
    return {
        'log': item['log'],
        'cluster': item['label_cluster'],
        'subcluster': item['label_subcluster'],
        'label': item['label_text']
    }


def process(text):
    vec = vectorizer.transform([clean_data(text.lower())[0]]).toarray()[0]
    nearest = ann.get_nns_by_vector(vec, 5)
    return {
        'label': df.iloc[nearest[0]]['label_text'],
        'cluster': df.iloc[nearest[0]]['label_cluster'],
        'subcluster': df.iloc[nearest[0]]['label_subcluster'],
        'nearest': list(
            map(
                lambda x: jsonify_item(df.iloc[x]),
                nearest
            )
        )
    }