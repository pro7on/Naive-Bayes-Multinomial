import os
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
stemmer = SnowballStemmer('english')
from nltk.corpus import stopwords
import math
#   import sys

path = os.getcwd()
for_train1 = path + "\\"+r"train\ham"
for_train2 = path + "\\"+r"train\spam"
for_test1 = path + "\\"+r"test\ham"
for_test2 = path + "\\"+r"test\spam"

data = pd.DataFrame(data = {'ham': [for_train1], 'spam': [for_train2]})


file_name1 = os.listdir(for_train1)
arr_words=[]
for file in file_name1:
    file_open = open(for_train1+'\\'+file, 'r', encoding="utf8", errors = 'ignore')
    low_case = file_open.read().lower()
    arr_words = arr_words +  ((re.findall(('\w+'), low_case)))
    file_open.close()


file_name2 = os.listdir(for_train2)
arr_words2=[]
for file in file_name2:
    file_open2 = open(for_train2+'\\'+file, 'r', encoding="utf8",errors = 'ignore')
    low_case2 = file_open2.read().lower()
    arr_words2 = arr_words2 + ((re.findall(('\w+'), low_case2)))
    file_open.close()

stopWords = set(stopwords.words('english'))
def get_arr():
    arr_all = list(set(arr_words + arr_words2))   #set to list
    #words = word_tokenize(data)
    wordsFiltered = []
    if(stop == 1):
        for w in arr_all:
            if w not in stopWords:
                wordsFiltered.append(w)
        arr_all = wordsFiltered
    return arr_all

stemmed = []
for q in arr_words:
    n = stemmer.stem(q)
    stemmed.append(n)    #stemmed


def ExtractTokensFromDoc(d):
    tokens=[]
    token_gen = open(d, 'r+', encoding="utf8" , errors = 'ignore')
    fileNme = token_gen.read().lower()                  #convert to lower
    tokens = (re.findall(('\w+'), fileNme))
    token_gen.close()
    wordsFiltered2 = []
    for w in tokens:
        if w not in stopWords:
            wordsFiltered2.append(w)
    return(tokens)

def ConcatenateTextOfAllDocsInClass(D,c):
    files_get = os.listdir(D.loc[0,c]);
    tokens1 = []
    for f in files_get:
        file_path = D.loc[0,c]+'\\'+f
        tokens1 = tokens1 + ExtractTokensFromDoc(file_path)
    wordsFiltered1 = []
    for w in tokens1:
        if w not in stopWords:
            wordsFiltered1.append(w)
    return(tokens1)


def CountDocsInClass(for_arr):
    counts_arr = os.listdir(for_arr);
    return len(counts_arr)

def CountDocs():
    N = CountDocsInClass(for_train1)+CountDocsInClass(for_train2)
    return N


def trainMultinomialNB():
    cols = ["ham","spam"]
    D = pd.DataFrame(data = {'ham': [for_train1], 'spam': [for_train2]})
    V = get_arr()
    N = CountDocs()
    prior = pd.DataFrame(np.zeros([1,2]), columns = cols)
    condprob = pd.DataFrame(np.zeros((len(V),2)), index = V, columns = cols)
    d = pd.DataFrame(np.zeros([1,2]), columns = cols)
    for c in cols:           
        d.loc[0,c] =0
        Nc = CountDocsInClass(D.loc[0,c])
        prior.loc[0,c] =Nc/N
        textc = ConcatenateTextOfAllDocsInClass(D,c)
        for t in V:
            d.loc[0,c] = d.loc[0,c] + textc.count(t) +1   #taking summation of denomenator by addding 1
        for tt in V:
            Tct = textc.count(tt) +1
            condprob.loc[tt,c]= Tct/d.loc[0,c]
    return V, prior, condprob



def ApplyMultinomialNB(C, V, prior, condprob,name,path):
    file_path = os.listdir(path);
    No = len(file_path)
    max_argc = 0
    numberOfAccurateClassifications = 0;
    for f in file_path:
        file_get = path+'\\'+f
        W = ExtractTokensFromDoc(file_get)
        score = pd.DataFrame(np.zeros([1,2]), columns = C)
        for c in C:
            score.loc[0,c]= np.log10(prior.loc[0,c])
            for t in W:
                if(V.count(t) != 0):
                    score.loc[0,c] += np.log10(condprob.loc[t,c])
        R = score.idxmax(axis = 1).values.tolist()[0]
        if (R == name):
            max_argc = max_argc + 1
    return No, max_argc

cols = ["ham","spam"]
stop = 0
V, prior, condprob = trainMultinomialNB()
C1,max_arg1 = ApplyMultinomialNB(cols, V, prior, condprob, "ham", for_test1)
C2,max_arg2 = ApplyMultinomialNB(cols, V, prior, condprob, "spam", for_test2)
print("Accuracy for test before filtering stop words: "),
print((max_arg1+max_arg2)/(C1+C2))
stop = 1
V, prior, condprob = trainMultinomialNB()
C12,max_arg21 = ApplyMultinomialNB(cols, V, prior, condprob, "ham", for_test1)
C11,max_arg22 = ApplyMultinomialNB(cols, V, prior, condprob, "spam", for_test2)
print("Accuracy for test after filtering stop words: "),
print((max_arg21+max_arg22)/(C12+C11))
