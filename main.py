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
from keras.layers import Input, Dense, LSTM, Dropout, GRU, Bidirectional, Flatten, Embedding, TimeDistributed
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
from keras.utils import plot_model, to_categorical

"""
Preprocessing functions
"""
def load_doc(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    result = []
    for line in lines:
        result.append(line)
    return result

def maxlen(ar):
    ll = 0
    for line in ar:
        ll = max(ll , len(line.split()))
    return ll

def build_vocab(texts, vocab):
    #sentences = texts.apply(lambda x: x.split()).values
    for sentence in texts:
        for word in sentence.split(' '):
            if word == '' or word == ' ':
                continue
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def load_glove(word_index, max_features):
    EMBEDDING_FILE = 'glove.6B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_fasttext(word_index, max_features):
    EMBEDDING_FILE = 'wiki-news-300d-1M.vec'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o) > 100 and o.split(" ")[0] in word_index)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix


"""
Main code starts here
"""
questions = load_doc("treated_questions.txt")
answers = load_doc("treated_answers.txt")
list_of_tuples = list(zip(questions, answers))
dataset = pd.DataFrame(list_of_tuples, columns = ['Questions', 'Answers'])

questions = dataset['Questions'].values
answers = dataset['Answers'].values

vocab = {}
vocab = build_vocab(questions, vocab)
vocab = build_vocab(answers, vocab)

unique_words = []
for x in vocab:
    unique_words.append(x)

sentences = []
for s in questions:
    sentences.append(s)
for s in answers:
    sentences.append(s)

print("Total %s sentences", len(sentences))

tokenizer = Tokenizer(num_words=len(unique_words), filters='')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=60)
labels = to_categorical(np.asarray(data))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.3 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join('./', 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("Shape of embedding matrix %s:", embedding_matrix.shape)

maxlenWords = 60
embed_dim = 300
vocab_size = len(unique_words) + 1

encoderInput = []
decoderInput = []
targetOuput = []

for sentence in questions:
    temp = []
    for word in sentence.split(' '):
        if word == '' or word == ' ':
            continue
        temp.append(word_index[word])
    to_add = maxlenWords - len(temp)
    for i in range(to_add):
        temp.append(0)
    encoderInput.append(temp)

for sentence in answers:
    temp = []
    for word in sentence.split(' '):
        if word == '' or word == ' ':
            continue
        temp.append(word_index[word])
    to_add = maxlenWords - len(temp)
    for i in range(to_add):
        temp.append(0)
    decoderInput.append(temp)

for sentence in answers:
    temp = []
    count = -1
    for word in sentence.split(' '):
        count = count + 1
        if word == '' or word == ' ':
            continue
        if count == 0:
            continue
        temp.append([word_index[word]])
    to_add = maxlenWords - len(temp)
    for i in range(to_add):
        temp.append([0])
    targetOuput.append(temp)

encoderInput = np.array(encoderInput)
decoderInput = np.array(decoderInput)
targetOuput = np.array(targetOuput)
targetOuput = to_categorical(targetOuput)

print("Shape of encoder input: ", encoderInput.shape)
print("Shape of decoder input: ", decoderInput.shape)
print("Shape of target output: ", targetOuput.shape)


####Model definition######
model_input = Input(shape=(maxlenWords, ))
embed_layer = Embedding(weights=[embedding_matrix], trainable=False, output_dim=embed_dim, input_dim=vocab_size, input_length=maxlenWords)(model_input)
encoder_output, state_h, state_c = LSTM(30, return_state=True)(embed_layer)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(maxlenWords, ))
decoder_embed = Embedding(weights=[embedding_matrix], trainable=False, output_dim=embed_dim, input_dim=vocab_size, input_length=maxlenWords)(decoder_inputs)
decoder_outputs, _, _, = LSTM(30, return_state=True, return_sequences=True)(decoder_embed, initial_state= encoder_states)
decoder_outputs = Dense(vocab_size, activation='softmax', name="Dense_layer")(decoder_outputs)
model = Model([model_input, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

callback = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=0,verbose=0, mode='auto')]

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

model.fit([encoderInput, decoderInput], targetOuput, batch_size=2, epochs=200, validation_split=0.3,
          callbacks= callback)

model.save('chatbot.h5')



"""
Sampling model or inference model
"""
encoder_model = Model(model_input, encoder_states)
decoder_state_input_h = Input(shape=(30,))
decoder_state_input_c = Input(shape=(30,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embed = Embedding(weights=[embedding_matrix], trainable=False, output_dim=embed_dim, input_dim=vocab_size, input_length=maxlenWords)(decoder_inputs)
decoder_outputs, state_h, state_c = LSTM(30, return_state=True)(decoder_embed, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = Dense(vocab_size, activation='softmax', name="Dense_layer")(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

reverse_input_char_index = dict(
    (i, char) for char, i in word_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in word_index.items())




input_seq = encoderInput[0:1]
inputt = np.array(input_seq)
print(inputt.shape)
states_value = encoder_model.predict(inputt)

target_seq = np.zeros((1, maxlenWords))
target_seq[0, 0] = word_index['_start_']
count = 1
stop_condition = False
decoded_sentence = ''
while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[-1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_char

        if (sampled_char == '_end_' or
                len(decoded_sentence) > 57):
            stop_condition = True

        target_seq = np.zeros((1, maxlenWords))
        target_seq[0, count] = sampled_token_index
        count = count + 1

        states_value = [h, c]

print(decoded_sentence)

