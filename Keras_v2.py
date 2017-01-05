# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 01:19:05 2016

"""

import keras.preprocessing.text
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

print('Loading data...')
import pandas

thedata = pandas.read_csv("./dataCalls/allCalls.csv", sep=', ', delimiter=',', header='infer', names=None)

x = thedata['text']
numagent = {"batch1":1 ,"batch2":2,"batch3":3 ,"batch4":4,"batch5":5 ,"batch6":6,"batch7":7 ,"batch8":8,"batch9":9 ,"batch10":10}
thedata['agentNum'] = thedata['agent'].replace(numagent)
thedata['agentNum'] = thedata['agentNum'].convert_objects(convert_numeric=True)
y = thedata['agentNum']

x = x.iloc[:].values
y = y.iloc[:].values

###################################
tk = keras.preprocessing.text.Tokenizer(nb_words=40000, filters=keras.preprocessing.text.base_filter(), lower=True, split=" ")
tk.fit_on_texts(x)

x = tk.texts_to_sequences(x)


###################################
max_len = 1000
print("max_len: " + str(max_len))
print('Pad sequences (samples x time)')

x = sequence.pad_sequences(x, maxlen=max_len)

#########################
max_features = 50000
model = Sequential()
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 1024, input_length=max_len, dropout=0.9))
model.add(LSTM(4096, dropout_W=0.9, dropout_U=0.9))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x, y=y, batch_size=32, nb_epoch=50000, verbose=1, validation_split=0.2, shuffle=True)
