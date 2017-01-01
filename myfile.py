# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import re
import math
from nltk.corpus import stopwords

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
stop_words = set(stopwords.words('english'))

def main():
    data_path = "../input/"
    print(check_output(["ls", "../input"]).decode("utf8"))
    
    in_file = open(data_path + "biology.csv")
    
    docs = []
    reader = csv.DictReader(in_file)
    count = 0
    for row in reader:
        if(count>0):
            break
        doc = int(row['id'])
        docs.append(doc)
        
        text = clean_html(row["title"]) + ' ' + clean_html(row["content"])
        print(text)
        words = get_words(text)
        print(words)
        print(clean_stop_words(words))

    count = count + 1


def clean_html(raw_html):
    match = re.compile('<.*?>')
    for unwanted in raw_html:
        fixed_doc = match.sub(r' ',raw_html)
    
    return fixed_doc


def get_words(text):
    word_split = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    return [word.strip().lower() for word in word_split.split(text)]

def clean_stop_words(words):
    myList=[]
    for word in words:
        if word not in stop_words and word.isalpha():
            myList.append(word)
    return myList

if __name__ == "__main__":
    print("Starting program.")
    main()
# Any results you write to the current directory are saved as output.