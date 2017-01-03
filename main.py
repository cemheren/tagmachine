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
x = pickle.load(open('all_input_dense_10000.pickle', 'rb'))
y = pickle.load(open('all_labels.pickle', 'rb'))
#
# xP = x[:1000]
# yP = y[:1000]
# pickle.dump(xP, open('1k_input_dense_10000.pickle', 'wb'))
# pickle.dump(yP, open('1k_labels.pickle', 'wb'))

print("splitting training and validation sets...")
# split training set into validation set
n_samples = len(x)
sidx = np.random.permutation(n_samples)
n_train = int(np.round(n_samples * (1. - 0.1)))
# training and validation sets
train_x = [x[s] for s in sidx[:n_train]]
train_y = [y[s] for s in sidx[:n_train]]
valid_x = [x[s] for s in sidx[n_train:]]
valid_y = [y[s] for s in sidx[n_train:]]

trainX = pad_sequences(train_x, maxlen=120, value=0.)
validX = pad_sequences(valid_x, maxlen=120, value=0.)

trainY = pad_sequences(train_y, maxlen=120, value=0.)
validY = pad_sequences(valid_y, maxlen=120, value=0.)

print("generating model...")
g = tflearn.input_data([None, 120])
g = tflearn.embedding(g, input_dim=10000, output_dim=128)

g = bidirectional_rnn(g, BasicLSTMCell(256), BasicLSTMCell(256))
g = tflearn.dropout(g, 0.3)

# g = tflearn.lstm(g, 128, dynamic=True)
# g = tflearn.dropout(g, 0.3)

g = tflearn.fully_connected(g, 120, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

m = tflearn.DNN(g, clip_gradients=5.0)

print("starting training.")

for i in range(50):
    m.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=32, n_epoch=2, run_id=str(i))
    print("-- TESTING...")
    q = m.predict(np.reshape(trainX[0], (1, 120)))[0]
    q = map(int, q)
    print(q)
    #print(m.predict(np.reshape(trainX[99], (1, 120))).astype(np.int64))


print("saving model: lstm.model")
m.save('lstm.model')

#
#
# for i in range(50):
#     seed = random_sequence_from_textfile(path, maxlen)
#     m.fit(X, Y, validation_set=0.1, batch_size=128,
#           n_epoch=1, run_id='nazim')
#     print("-- TESTING...")
#     print("-- Test with temperature of 1.0 --")
#     print(m.generate(600, temperature=1.0, seq_seed=seed))
#     print("-- Test with temperature of 0.5 --")
#     print(m.generate(600, temperature=0.5, seq_seed=seed))
