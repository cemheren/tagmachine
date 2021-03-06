import os
import string
import pickle
from six.moves import urllib

import tflearn
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

n_travel = 20000

train_x = x[:(n_samples - n_travel)] # last 8k is all travel
train_y = y[:(n_samples - n_travel)]
valid_x = x[(n_samples - n_travel):]
valid_y = y[(n_samples - n_travel):]

trainX = pad_sequences(train_x, maxlen=120, value=0.)
validX = pad_sequences(valid_x, maxlen=120, value=0.)

trainY = pad_sequences(train_y, maxlen=120, value=0.)
validY = pad_sequences(valid_y, maxlen=120, value=0.)

hidden_dim = 128

print("generating model...")
g = tflearn.input_data([None, 120])
g = tflearn.embedding(g, input_dim=10000, output_dim=hidden_dim)

g = tflearn.fully_connected(g, hidden_dim, activation='tanh')
g = tflearn.dropout(g, 0.3)

# g = tflearn.lstm(g, 128, dynamic=True)
# g = tflearn.dropout(g, 0.3)

g = tflearn.fully_connected(g, 120, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.005)

m = tflearn.DNN(g, clip_gradients=5.0)

print("starting training.")

for i in range(30):
    m.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True, batch_size=32, n_epoch=2, run_id=str(i))
    print("-- TESTING...")
    q = m.predict(np.reshape(trainX[0], (1, 120)))[0]
    q = np.argmax(q, axis=0)
    print("prediction = ", q)

    print("actual = ", np.argmax(trainY[0], axis=0))


print("saving model: linear.model")
m.save('linear.model')