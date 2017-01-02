from __future__ import absolute_import, division, print_function

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

csvPath = "./biology.csv"

csvfile = tflearn.data_utils.load_csv(csvPath, has_header=True)

print(csvfile)