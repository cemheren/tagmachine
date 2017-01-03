from __future__ import absolute_import, division, print_function

import os
import string
import pickle
from six.moves import urllib

import tflearn
import collections
from tflearn.data_utils import *

input = []
input.append(pickle.load(open('biology.pickle', 'rb')))
input.append(pickle.load(open('cooking.pickle', 'rb')))
input.append(pickle.load(open('crypto.pickle', 'rb')))
input.append(pickle.load(open('diy.pickle', 'rb')))
input.append(pickle.load(open('robotics.pickle', 'rb')))
input.append(pickle.load(open('travel.pickle', 'rb')))

s = []
labels = input[1]
all_labels = []

for kind in input:
    for i in range(len(kind[0])):
        line = kind[0][i]
        label = kind[1][i]

        ws = []
        a = line[1].split()
        for w in a:
            t = w.lower().translate(None, string.punctuation)
            ws.append(t)

        a = line[2].split()
        for w in a:
            t = w.lower().translate(None, string.punctuation)
            ws.append(t)

        s.append(ws)

        a = label.split()
        l = []
        ll = []

        for li in a:
            local = li.lower().translate(None, string.punctuation)
            l.append(local)

        for w in ws:
            # t = w.lower().translate(None, string.punctuation)
            if w in l:
                ll.append(1)
            else:
                ll.append(0)

        all_labels.append(ll)



count = [['UNK', -1]]
count.extend(collections.Counter(s).most_common(10000 - 1))

dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)

dictionary = pickle.load(open('dict.temp', 'rb'))
input = pickle.load(open('input.temp', 'rb'))

data = list()
unk_count = 0

for line in input:
    dLine = []
    for word in line[1].split():
        word = word.lower().translate(None, string.punctuation)

        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1

        dLine.append(index)

    for word in line[2].split():
        word = word.lower().translate(None, string.punctuation)

        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1

        dLine.append(index)
    data.append(dLine)

#count[0][1] = unk_count

reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
print(len(dictionary))
#g = tflearn.input_data([None, 25,30 ])
