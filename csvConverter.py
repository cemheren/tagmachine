from __future__ import absolute_import, division, print_function

import re
import pickle

import os
import pickle
from six.moves import urllib
import tflearn
from tflearn.data_utils import *
import sys

match = re.compile('<.*?>');

csvname = sys.argv[1]

def clean_html(raw_html):
    for _ in raw_html:
        fixed_doc = match.sub(r' ', raw_html)
    return fixed_doc


csvPath = "./" + csvname + ".csv"

csvfile = tflearn.data_utils.load_csv(csvPath, has_header=True)

l = len(csvfile[1])

for i in range(l):
    csvfile[0][i][2] = clean_html(csvfile[0][i][2])

    if i % 100 == 0:
        print(i)

with open(csvname + '.pickle', 'wb') as handle:
    pickle.dump(csvfile, handle, protocol=pickle.HIGHEST_PROTOCOL)

#print(csvfile)
