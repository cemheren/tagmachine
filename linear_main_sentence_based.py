import os
import string
import pickle
from six.moves import urllib

import tflearn
import numpy as np
import random
import collections
from tflearn.data_utils import *
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell

print("loading data...")
all_data = pickle.load(open('all_only_valid.pickle', 'rb'))

print("splitting training and validation sets...")
n_samples = len(all_data)
n_travel = 60000

n_padding = 30

train_all_data = all_data[:(n_samples - n_travel)] # last n_travel is all travel
valid_all_data = all_data[(n_samples - n_travel):]

train_x = [k[1] for k in train_all_data]
train_y = [k[2] for k in train_all_data]

valid_x = [k[1] for k in valid_all_data]
valid_y = [k[2] for k in valid_all_data]

trainX = pad_sequences(train_x, maxlen=n_padding, value=0.)
validX = pad_sequences(valid_x, maxlen=n_padding, value=0.)

trainY = pad_sequences(train_y, maxlen=n_padding, value=0.)
validY = pad_sequences(valid_y, maxlen=n_padding, value=0.)

hidden_dim = 256

print("generating model...")
g = tflearn.input_data([None, n_padding])
g = tflearn.embedding(g, input_dim=10002, output_dim=hidden_dim)

g = tflearn.fully_connected(g, hidden_dim, activation='sigmoid')
g = tflearn.dropout(g, 0.3)

# g = tflearn.lstm(g, 128, dynamic=True)
# g = tflearn.dropout(g, 0.3)

g = tflearn.fully_connected(g, n_padding, activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.005)

m = tflearn.DNN(g, clip_gradients=5.0)

print("starting training.")

for i in range(30):
    m.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True, batch_size=1024, n_epoch=2, run_id=str(i))
    print("-- TESTING...")

    for k in range(10):
        rand = random.randint(0, len(validX))
        print(valid_all_data[rand][0])

        q = m.predict(np.reshape(validX[rand], (1, n_padding)))[0]
        q = np.argmax(q, axis=0)

        print("prediction = ", q)
        if len(valid_all_data[rand][0]) > q:
            print(valid_all_data[rand][0][q])

        print("actual = ")
        for index in range(len(validY[rand])):
            if validY[rand][index] == 1:
                print(valid_all_data[rand][0][index])

        print("------------------------------------------")


print("saving model: linear_valid_biology_v2.model")
m.save('linear_valid_biology_v2.model')