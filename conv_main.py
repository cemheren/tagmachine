import os
import string
import pickle
from six.moves import urllib

import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import collections
from tflearn.data_utils import *
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

print("loading data...")
x1 = pickle.load(open('all_input_dense_10000.pickle', 'rb'))
y1 = pickle.load(open('all_labels.pickle', 'rb'))
#
# xP = x[:1000]
# yP = y[:1000]
# pickle.dump(xP, open('1k_input_dense_10000.pickle', 'wb'))
# pickle.dump(yP, open('1k_labels.pickle', 'wb'))
x = []
y = []

count = 0
k = 0

e = max(x1, key=len)
print(e)
print(len(e))

for i in range(len(y1)):
    c = y1[i].count(1)

    for r in range(len(y1[i])):
        if y1[i][r] == 1:
            n = np.zeros(len(y1[i]))
            n[r] = 1

            x.append(x1[i])
            y.append(n)

    if c > 1:
        k += 1

    if c == 0:
        count += 1
    # else:
    #     x.append(x1[i])
    #     y.append(y1[i])

print(k)
print(count)
print(len(x))
print(len(y))

print("splitting training and validation sets...")
# split training set into validation set
n_samples = len(x)
# sidx = np.random.permutation(n_samples)
# n_train = int(np.round(n_samples * (1. - 0.1)))
# # training and validation sets
# train_x = [x[s] for s in sidx[:n_train]]
# train_y = [y[s] for s in sidx[:n_train]]
# valid_x = [x[s] for s in sidx[n_train:]]
# valid_y = [y[s] for s in sidx[n_train:]]

train_x = x[:(n_samples - 8000)] # last 8k is all travel
train_y = y[:(n_samples - 8000)]
valid_x = x[(n_samples - 8000):]
valid_y = y[(n_samples - 8000):]

trainX = pad_sequences(train_x, maxlen=120, value=0.)
validX = pad_sequences(valid_x, maxlen=120, value=0.)

trainY = pad_sequences(train_y, maxlen=120, value=0.)
validY = pad_sequences(valid_y, maxlen=120, value=0.)

hidden_dim = 128

print("generating model...")
network = input_data(shape=[None, 120], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=hidden_dim)
branch1 = conv_1d(network, hidden_dim, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, hidden_dim, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, hidden_dim, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)

network = fully_connected(network, 120, activation='softmax')

network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')


m = tflearn.DNN(network, clip_gradients=5.0)

print("starting training.")

for i in range(30):
    m.fit(trainX, trainY, validation_set=(validX, validY), shuffle=True, show_metric=True, batch_size=32, n_epoch=10, run_id=str(i))
    print("-- TESTING...")
    q = m.predict(np.reshape(trainX[0], (1, 120)))[0]
    q = np.argmax(q, axis=0)
    print("prediction = ", q)

    print("actual = ", np.argmax(trainY[0], axis=0))
    #print(m.predict(np.reshape(trainX[99], (1, 120))).astype(np.int64))


print("saving model: conv.model")
m.save('conv.model')