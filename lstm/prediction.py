# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 18:20:31 2020

@author: marut
"""

from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

file = open("en_US.txt", "r", encoding = "utf8")
lines = []
for i in file:
    lines.append(i)
    
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])

data = ""

for i in lines:
    data = ' '. join(lines)

#data cleaning    
cleaned = re.sub(r'\W+', ' ', data).lower()

#tokenization
tokens = word_tokenize(cleaned)

train_len = 4
text_seq = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_seq.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_seq)
word_index=tokenizer.word_index
sequences = tokenizer.texts_to_sequences(text_seq) 

n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

X = n_sequences[:,:-1]
y = n_sequences[:,-1]
y = to_categorical(y, num_classes=vocabulary_size)
seq_len = X.shape[1]
print(X.shape)
print(y[0])

vocabulary_size = len(tokenizer.word_counts)+1
model = load_model("mymodel3.h5")

from keras.preprocessing.sequence import pad_sequences
while(1):
    input_text = input().strip().lower()
    if(input_text=="exit"):
        break
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    print(encoded_text, pad_encoded)
    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
      pred_word = tokenizer.index_word[i]
      print("Next word suggestion:",pred_word)