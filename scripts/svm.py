from feature import *
import sklearn.svm as svm
import math

# XSet: [[], [], ...]
# return: [[word, cnt], [word, cnt], ...]
def calcTopWords(XSet, N):
    topSet = {}
    for li in XSet:
        for word in li:
            if word not in topSet.keys():
                topSet[word] = 0
            topSet[word] += 1
            
    # sort
    sortedWords = list(sorted(topSet, key=lambda x: topSet[x], reverse=True))
    resLi = []
    for i in range(0, N):
        resLi.append([sortedWords[i], topSet[sortedWords[i]]])

    return resLi

# XSet: [[], [], ...]
# return: [[word, weight], [word, weight], ...]
def genWeight(XSet, N):
    assert(N > 0)
    topList = calcTopWords(XSet, N)
    maxFreq = topList[0][-1]
    for pair in topList:
        pair[-1] = float(pair[-1]) / maxFreq
    return topList

def genTopWordLists(X, Y, N):
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
    topGood = genWeight(good, N)
    topBad = genWeight(bad, N)
    topNorm = genWeight(norm, N)
    return (topGood, topNorm, topBad)
    

def svm():
    pass

if __name__ == '__main__':
    (X, Y) = getTrainFeature()
    (g, n, b) = genTopWordLists(X, Y, 50)
    print(g, '\n')
    print(n, '\n')
    print(b, '\n')
    