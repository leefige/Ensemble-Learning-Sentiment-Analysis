#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

import math
import random

from util import *

from sklearn.feature_extraction.text import TfidfVectorizer

negPrefix = "NOT_"

negaWords = set(["不", "不是", "没", "没有", "木有"])
stopWord = set(["的", "你", "我", "地", "得", "了", "很"])
punc = set(['，', '。', '！', '~', '？', '、', '…', '~', '《', '》', '“', '”', '‘', '’', '?', '!', ',', '.', '...', '@', '#', '*', '$', '￥', '%', '/', '-'])

# XSet: [[], [], ...]
# return: set()
def calcTopWords(XSet, N):
    topSet = {}
    for li in XSet:
        for word in li:
            if word not in topSet.keys():
                topSet[word] = 0
            topSet[word] += 1
            
    # sort
    sortedWords = list(sorted(topSet, key=lambda x: topSet[x], reverse=True))
    resSet = set()
    cnt = 0
    for word in sortedWords:
        resSet.add(word)
        cnt += 1
        if cnt >= N:
            break
    return resSet

def genTopWordSet(X, Y, N):
    good = []
    norm = []
    bad = []
    cnt = len(Y)
    for i in range(0, cnt):
        if Y[i] == 1:
            good.append(X[i])
        elif Y[i] == 0:
            norm.append(X[i])
        elif Y[i] == -1:
            bad.append(X[i])
        else:
            print("Bad label %s @ %d" % (str(Y[i]), i))
    topGood = calcTopWords(good, N)
    topBad = calcTopWords(bad, N)
    topNorm = calcTopWords(norm, N)
    return list(topGood.union(topBad).union(topNorm))

# self-implementation
def genIDFDict(X, N=None):
    total = len(X)
    allWord = {}
    # create item
    for li in X:
        # unify
        liSet = set(li)
        # count
        for word in liSet:
            if word not in allWord.keys():
                allWord[word] = 0
            allWord[word] += 1
    # calc idf
    for key in allWord.keys():
        allWord[key] = math.log(float(total) / (1 + allWord[key]))
    # get top
    if (N != None) and (N > 0):
        # sort
        sortedWords = list(sorted(allWord, key=lambda x: allWord[x], reverse=True))
        cnt = 0
        resDict = {}
        for word in sortedWords:
            resDict[word] = allWord[word]
            cnt += 1
            if cnt >= N:
                break
        return resDict
    # or, just return all
    else:
        return allWord

# function: 
#   parse negative words
# param:
#   X: ["xxx", "xxx", ...] x set of input
# return:
#   [[], [], ...]
def parseNeg(X):
    res = []
    for review in X:
        feat = []
        words = re.split(r"\s+", review)
        inNeg = False
        for word in words:
            # is stop word / null?
            if word in stopWord or word == '':
                continue
            # is number?
            if re.match(r"[\d\.]+", word):
                continue

            # is punctuation?
            if word in punc:
                # end of negative field
                inNeg = False
                if not (word == '，' or word == '。'):
                    feat.append(word)
            else:
                if word in negaWords:
                    inNeg = not inNeg
                    continue
                if inNeg:
                    word = negPrefix + word
                feat.append(word)

        res.append(feat)
    return res

def genXFeature(topSet, X):
    print("Generating features...")
    featNum = len(topSet)
    X_new = []
    # cur = 0
    for review in X:
        # cur += 1
        # if cur % 1000 == 0:
        #     print(cur)
        featDict = {}
        # init
        for tw in topSet:
            featDict[tw] = 0
        # gen feature
        for word in review:
            if word in featDict.keys():
                featDict[word] += 1
        # normalize
        wordCnt = len(review)
        for feat in featDict.keys():
            featDict[feat] = float(featDict[feat]) / wordCnt

        featLi = []
        for tw in topSet:
            featLi.append(featDict[tw])
        assert(len(featLi) == featNum)
        X_new.append(featLi)
    return X_new

# self-implementation
def genTFIDFFeature(idf, X):
    print("Generating tf-idf features...")
    featNum = len(idf)
    X_new = []
    for review in X:
        featDict = {}
        # init
        for tw in idf.keys():
            featDict[tw] = 0
        # calc freq
        for word in review:
            if word in featDict.keys():
                featDict[word] += 1
        # normalize tf
        wordCnt = len(review)
        for feat in featDict.keys():
            featDict[feat] = float(featDict[feat]) / wordCnt
        # generate list in order
        featLi = []
        for tw in idf.keys():
            featLi.append(featDict[tw])
        assert(len(featLi) == featNum)
        X_new.append(featLi)
    return X_new

# pre-process for using sklearn
def skTFIDFPreproc(X):
    res = []
    for li in X:
        res.append(" ".join(li))
    return res

# tf-idf with sklearn
def genSkTFIDF(X, maxFeatures=None):
    # print("stop: ", list(stopWord.union(punc)))
    tfidfVecorizer = TfidfVectorizer(analyzer='word', max_features=maxFeatures, stop_words=list(stopWord.union(punc)))
    tfidf = tfidfVecorizer.fit_transform(X)  
    # tfidf.todense()  
    vocab = tfidfVecorizer.vocabulary_ 
    words = tfidfVecorizer.get_feature_names()
    feature = tfidf.toarray()
    # print(feature[:10])
    return words, vocab, feature

# get data of train
def getTrainData():
    (X, Y) = readTrainData()
    X_no_neg = parseNeg(X)
    return (X_no_neg, Y)

# get data of valid
def getValidData():
    X = readValidData()
    return parseNeg(X)

def getTrainData_tfidf(N=None):
    (X, Y) = getTrainData()
    X = skTFIDFPreproc(X)
    words, vocab, X_ = genSkTFIDF(X, maxFeatures=N)
    return words, vocab, X_, Y

def getValidData_tfidf(words, vocab):
    X = getValidData()
    X = skTFIDFPreproc(X)
    tfidfVecorizer = TfidfVectorizer(analyzer='word', vocabulary=vocab, stop_words=list(stopWord.union(punc)))
    tfidf = tfidfVecorizer.fit_transform(X)  
    feature = tfidf.toarray()
    return feature

if __name__ == '__main__':
    words, vocab, X_, Y = getTrainData_tfidf(80)
    print(words[:20])
    print(X_[0])

    x_v = getValidData_tfidf(words, vocab)
    print(x_v[0])