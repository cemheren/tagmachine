from __future__ import absolute_import, division, print_function

import os
import string
import pickle
from six.moves import urllib

import numpy
import re
import tflearn
import collections
from tflearn.data_utils import *

input = []
input.append(pickle.load(open('biology.pickle', 'rb')))
# input.append(pickle.load(open('cooking.pickle', 'rb')))
# input.append(pickle.load(open('crypto.pickle', 'rb')))
# input.append(pickle.load(open('diy.pickle', 'rb')))
# input.append(pickle.load(open('robotics.pickle', 'rb')))
# input.append(pickle.load(open('travel.pickle', 'rb')))

dictionary = pickle.load(open('dict.temp', 'rb'))

all_labels = []
all_data = []

for kind in input:
    for i in range(len(kind[0])):
        line = kind[0][i]
        label = kind[1][i]

        labelWords = filter(None, re.split("[, \-!?:]+", label))
        sentences = filter(None, re.split("[\n.!?]+", line[1]))
        bodySentences = filter(None, re.split("[\n.!?]+", line[2]))
        sentences.extend(bodySentences)

        for sentence in sentences:
            words = sentence.split()
            if len(words) == 0:
                continue

            words.append("<end>")

            wordArray = []
            dictArray = []
            labelArray = numpy.zeros(len(words))

            for wordIndex in range(len(words)):
                transformed = words[wordIndex].lower().translate(None, string.punctuation)
                wordArray.append(transformed)

                if transformed in dictionary:
                    dictEntry = dictionary[transformed]
                else:
                    if words[wordIndex] == "<end>":
                        dictEntry = 10002
                    else:
                        dictEntry = 0  # dictionary['UNK']

                dictArray.append(dictEntry)

                if transformed in labelWords:
                    labelArray[wordIndex] = 1

            if (labelArray == 0).sum() == len(words):
                labelArray[wordIndex] = 1

            all_data.append([wordArray, dictArray, labelArray])


pickle.dump(all_data ,open('all_biology_v2.pickle', 'wb'))



