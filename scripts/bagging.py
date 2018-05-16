#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

import getopt
import random
import sys

import numpy as np
from sklearn.model_selection import train_test_split

import dtree
import svm
from feature import *

clfType = dtree
topWordSize = 50
dtNum = 5
testPercent = .1
sampleRate = 0.5

def sampling(X, Y, size):
    total = len(X)
    random.seed()
    splNum = random.sample(range(0, total), size)
    resX = []
    resY = []

    for i in range(0, size):
        resX.append(X[splNum[i]])
        resY.append(Y[splNum[i]])
    return resX, resY

def maxCntRes(results):
    finRes = []
    liCnt = len(results)
    assert(liCnt > 0)
    resSize = len(results[0])
    # for each sample
    for i in range(0, resSize):
        dic = {}
        # for each res list
        for j in range(0, liCnt):
            if results[j][i] not in dic.keys():
                dic[ results[j][i] ] = 0
            dic[ results[j][i] ] += 1

        # get cnt max
        maxCnt = -1
        maxCl = None
        for cl in dic.keys():
            if dic[cl] > maxCnt:
                maxCnt = dic[cl]
                maxCl = cl
        finRes.append(maxCl)

    return finRes
            

def bagging(clfType, topSet, X, Y):
    print("Bagging: %s" % clfType.__name__)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testPercent)
    trainLen = len(X_train)
    print("Bagging Train set size: ", trainLen)

    sampleSize = int(trainLen * sampleRate)
    forest = []
    for i in range(0, dtNum):
        print("\nGenerating %s %d..." % (clfType.__name__, i))
        X_smpl, y_smpl = sampling(X_train, y_train, random.randint(int(sampleSize / 2), sampleSize))
        tree = clfType.train(topSet, X_smpl, y_smpl, test_size=0.1)
        forest.append(tree)

    # test
    testLen = len(y_test)
    print("\nBagging Test set size: ", testLen)

    res = []
    for i in range(0, dtNum):
        res.append(clfType.validate(forest[i], topSet, X_test))
        # print(res[i][:10])

    baggingRes = maxCntRes(res)
    detail = "dtNum = %d, feature_num = %d, test_percent = %.2f, sample_size = %.2f" % (dtNum, len(topSet), testPercent, sampleRate)
    showTestResult(baggingRes, y_test, clType='Bagging_%s'%(clfType.__name__), title=detail)

    return forest
    
# validate
def validate(clfType, bag, topSet, X):
    print("\nBagging %s Validating..." % clfType.__name__)
    print("X: ", len(X))

    res = []
    for i in range(0, dtNum):
        res.append(clfType.validate(bag[i], topSet, X))
    baggingRes = maxCntRes(res)

    return baggingRes

if __name__ == '__main__':

    opts, args = getopt.getopt(sys.argv[1:], "hc:n:t:r:")

    for name, value in opts:
        if name == '-h':
            print("Usage: %s -c [dtree|svm] -t topWordSize -n clfNum -r sampleRate" % sys.argv[0])
            exit(-1)
        elif name == '-c':
            if value == 'd' or value == "dtree":
                clfType = dtree
            elif value == 's' or value == "svm":
                clfType = svm
            else:
                print("Bad clfType")
                exit(-2)
        elif name == '-t':
            topWordSize = int(value)
        elif name == '-n':
            dtNum = int(value)
        elif name == '-r':
            assert(float(value) <= 1)
            sampleRate = float(value)
        else:
            print("Ignore opt: %s" % name)

    (X, Y) = getTrainData()
    topSet = genTopWordSet(X, Y, topWordSize)

    X_new = genXFeature(topSet, X)
    forest = bagging(clfType, topSet, X_new, Y)

    X_valid = getValidData()
    X_valid_new = genXFeature(topSet, X_valid)
    y_valid = validate(clfType, forest, topSet, X_valid_new)
    print(y_valid[:10])
    genSubmission(y_valid)
