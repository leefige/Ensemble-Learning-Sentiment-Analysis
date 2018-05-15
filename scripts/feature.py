#-*- coding: UTF-8 -*-
from util import *
import random

negPrefix = "NOT_"

negaWords = set(["不", "不是", "没", "没有", "木有"])
stopWord = set(["的", "你", "我", "地", "得", "了", "很"])
punc = set(['，', '。', '！', '~', '？', '、', '…', '~', '《', '》', '“', '”', '‘', '’', '?', '!', ',', '.', '...', '@', '#', '*', '$', '￥', '%'])

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

# get data of train
def getTrainData():
    (X, Y) = readTrainData()
    X_no_neg = parseNeg(X)
    return (X_no_neg, Y)

# get data of valid
def getValidData():
    X = readValidData()
    return parseNeg(X)

if __name__ == '__main__':
    (X, Y) = readTrainData()
    X_ = parseNeg(X)
    print(X_[:20])
