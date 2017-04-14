import os
import string
import pickle
from six.moves import urllib

import tflearn
import numpy as np
import collections
from tflearn.data_utils import *
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

max_len = 4

def get_sequences(arr, seq_maxlen = 10, redun_step = 1):
    sequences = []
    for i in range(0, len(arr) - seq_maxlen, redun_step):
        sequences.append(arr[i: i + seq_maxlen])
    return sequences

print("loading test data...")
x = pickle.load(open('test.minor.pickle', 'rb'))

# dict = pickle.load(open('dict.temp', 'rb'))
#
# final = []
#
# for i in range(len(x[0])): #x[0][1] and x[1]
#     h = x[0][i][1]
#     q = x[1][i]
#
#     l = []
#     d = []
#
#     for w in h.split():
#         clean = w.lower().translate(None, string.punctuation)
#         l.append(clean)
#         if clean in dict:
#             d.append(dict[clean])
#         else:
#             d.append(0)
#
#     for w in q.split():
#         clean = w.lower().translate(None, string.punctuation)
#         l.append(clean)
#         if clean in dict:
#             d.append(dict[clean])
#         else:
#             d.append(0)
#
#     p = []
#     p.append(l)
#     p.append(d)
#
#     final.append(p)
#
# pickle.dump(final, open('test.final.pickle', 'wb'))

x1 = np.array(x)[:, 1]

xP = []
for line in x1:
    xP.extend(line)
    xP.append(10000)  # stop symbol


xxx = []
seq = get_sequences(xP, max_len)
for k in range(len(seq)):
    dform = []
    for g in seq[k]:
        p = np.zeros(1, dtype=np.int)
        p[0] = g
        dform.append(p)
    xxx.append(dform)


print("generating model...")
g = tflearn.input_data([None, max_len, 1])
# g = tflearn.embedding(g, input_dim=10000, output_dim=256)

g = tflearn.lstm(g, 256, return_seq=True)
g = tflearn.dropout(g, 0.3)

g = tflearn.lstm(g, 256)
g = tflearn.dropout(g, 0.3)

g = tflearn.fully_connected(g, 2, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

m = tflearn.DNN(g, clip_gradients=5.0)

print("loading previous model...")
m.load("./fullyTrainedModels/lstm_v2.model")

k = 0
for i in range(30):

    print("-- TESTING...")

    arr = []
    for _ in range(10000):
        q = m.predict(np.reshape(xxx[k], (1, max_len, 1)))[0]
        q = np.argmax(q, axis=0)
        arr.append(q)

        k += 1

        if xxx[k][0] == 10000:
            break;

    print(x[i][0])
    print(x[i][1])

    print(arr)

    print(len(arr))
    print(len(x[i][0]))

    for t in range(len(x[i][0])):
        if arr[t] == 1:
            print(x[i][0][t])
