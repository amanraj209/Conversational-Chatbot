import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.preprocessing import Imputer , Normalizer , scale, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas import get_dummies
import xgboost as xgb
import scipy
import math
import json
import sys
import csv
import os
import tqdm
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, Embedding
from keras.optimizers import SGD
from tqdm import tqdm_notebook
from nltk.corpus import stopwords
import string
from collections import Counter
from string import punctuation
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from gensim.models import Word2Vec
import re
from string import digits
import operator
import gc


def build_vocab(texts, vocab):
    for sentence in texts:
        for word in sentence.split(' '):
            if word == '' or word == ' ':
                continue
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == 'wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index


def load_doc(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    result = []
    for line in lines:
        result.append(line)
    return result


def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]
    return unknown_words


def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")


def known_contractions(embed, contraction_mapping):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known


def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text


def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '',
                'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    return text


def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x

"""
Main code starts here
"""
questions = load_doc("questions.txt")
answers = load_doc("answers.txt")
list_of_tuples = list(zip(questions, answers))
dataset = pd.DataFrame(list_of_tuples, columns = ['Questions', 'Answers'])

glove = 'glove.6B.300d.txt'
wiki_news = 'wiki-news-300d-1M.vec'

print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
print("Extracting FastText embedding")
embed_fasttext = load_embed(wiki_news)


vocab = {}
vocab = build_vocab(dataset['Questions'], vocab)
vocab = build_vocab(dataset['Answers'], vocab)
dataset['lowered_question'] = dataset['Questions'].apply(lambda x: x.lower())
dataset['lowered_answer'] = dataset['Answers'].apply(lambda x: x.lower())
add_lower(embed_glove, vocab)
add_lower(embed_fasttext, vocab)

contraction_mapping = {"ain't": "is not", "aren't": "are not", "http://www.icts-learninganalytics.com": "the official site",
                       "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                       "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                       "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                       "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                       "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have",
                       "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have",
                       "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                       "mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                       "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                       "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
                       "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
                       "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have",
                       "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                       "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have",
                       "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",
                       "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                       "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                       "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
                       "y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have", "you're": "you are", "you've": "you have"}

dataset['treated_question'] = dataset['lowered_question'].apply(lambda x: clean_contractions(x, contraction_mapping))
dataset['treated_answer'] = dataset['lowered_answer'].apply(lambda x: clean_contractions(x, contraction_mapping))

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-",
                 "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta',
                 '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

dataset['treated_question'] = dataset['treated_question'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
dataset['treated_answer'] = dataset['treated_answer'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))

vocab = {}
vocab = build_vocab(dataset['treated_question'], vocab)
vocab = build_vocab(dataset['treated_answer'], vocab)
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)

treated_questions = dataset['treated_question'].tolist()
treated_answers = dataset['treated_answer'].tolist()

with open('treated_questions.txt', 'w') as f:
    for item in treated_questions:
        f.write("%s\n" % item)

with open('treated_answers.txt', 'w') as f:
    for item in treated_answers:
        f.write("%s\n" % str("_start_ " + item + "_end_"))