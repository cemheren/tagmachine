import os
import string
import pickle
from six.moves import urllib

import tflearn
import numpy as np
import collections
from tflearn.data_utils import *
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

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

xP = np.array(x)[:, 1]
bigX = pad_sequences(xP, maxlen=120, value=0.)

print("generating model...")
g = tflearn.input_data([None, 120])
g = tflearn.embedding(g, input_dim=10000, output_dim=256)

g = tflearn.lstm(g, 256)
g = tflearn.dropout(g, 0.3)

g = tflearn.fully_connected(g, 120, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

m = tflearn.DNN(g, clip_gradients=5.0)

print("loading previous model...")
m.load("./fullyTrainedModels/lstm.model")


for i in range(30):
    print("-- TESTING...")
    q = m.predict(np.reshape(bigX[i], (1, 120)))[0]
    q = np.argmax(q, axis=0)
    print("prediction = ", q)

    print(x[i][0])
    print(x[i][1])

    if q < len(x[i][0]):
        print("word = ", x[i][0][q])
    else:
        print("no prediction")
