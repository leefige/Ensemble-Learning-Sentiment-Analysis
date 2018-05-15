#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

import math
import random

import numpy as np
from sklearn.model_selection import train_test_split

import dtree
import svm
from feature import *

maxIter = 10
topWordSize = 300
testPercent = .1

def reWeight(weight, res, target):
    total = len(res)
    err = 0.0
    for i in range(0, total):
        # for miss, add its to error rate
        if res[i] != target[i]:
            err += weight[i]

    # to weak, abort
    if err > 0.5:
        return None, 0
    
    # reWeight
    beta = err / (1.0 - err)
    neoWeight = []
    weightSum = 0.0
    for i in range(0, total):
        # correct
        if res[i] == target[i]:
            neoWeight.append(weight[i] * beta)
        # miss
        else:
            neoWeight.append(weight[i])
        weightSum += neoWeight[i]

    # normalize
    assert(weightSum <= 1)
    for i in range(0, total):
        neoWeight[i] = neoWeight[i] / weightSum
    
    return neoWeight, beta
    

def getBoostRes(results, betas):
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
            # voting with weight beta_t
            dic[ results[j][i] ] += math.log(1.0 / betas[j])

        # get cnt max
        maxNum = -1
        maxCl = None
        for cl in dic.keys():
            if dic[cl] > maxNum:
                maxNum = dic[cl]
                maxCl = cl
        finRes.append(maxCl)

    return finRes
            

def boostDtree(topSet, X, Y):
    print("Boosting: DTree")
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testPercent)
    trainLen = len(X_train)
    print("Boosting Train set size: ", trainLen)

    forest = []
    betas = []
    initWeight = 1.0 / trainLen
    sampleWeight = [initWeight for i in range(trainLen)]
    
    # boosting
    for i in range(0, maxIter):
        print("\nBoosting tree %d..." % i)
        # NOTE: test_size should be 0 to satisfy init weight
        tree = dtree.train(topSet, X_train, y_train, test_size=0, sample_weight=sampleWeight)
        y_train_res = dtree.validate(tree, topSet, X_train)
        # reweight
        sampleWeight, beta = reWeight(sampleWeight, y_train_res, y_train)
        # check abort
        if (sampleWeight == None) or (beta == 0):
            break
        else:
            forest.append(tree)
            betas.append(beta)
            
    # test
    testLen = len(y_test)
    print("\nBoosting Test set size: ", testLen)

    res = []
    clfCnt = len(forest)
    for i in range(0, clfCnt):
        res.append(dtree.validate(forest[i], topSet, X_test))
        # print(res[i][:10])

    boostRes = getBoostRes(res, betas)
    detail = "iter = %d, feature_num = %d, test_percent = %.2f" % (clfCnt, len(topSet), testPercent)
    showTestResult(boostRes, y_test, clType='Boosting_DTree', title=detail)

    return forest, betas

def boostSVM(topSet, X, Y):
    print("Boosting: SVM")
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testPercent)
    trainLen = len(X_train)
    print("Boosting Train set size: ", trainLen)

    forest = []
    betas = []
    initWeight = 1.0 / trainLen
    sampleWeight = [initWeight for i in range(trainLen)]
    
    # boosting
    for i in range(0, maxIter):
        print("\nBoosting SVM %d..." % i)
        # NOTE: test_size should be 0 to satisfy init weight
        tree = svm.train(topSet, X_train, y_train, test_size=0, sample_weight=sampleWeight)
        y_train_res = svm.validate(tree, topSet, X_train)
        # reweight
        sampleWeight, beta = reWeight(sampleWeight, y_train_res, y_train)
        # check abort
        if (sampleWeight == None) or (beta == 0):
            break
        else:
            forest.append(tree)
            betas.append(beta)
            
    # test
    testLen = len(y_test)
    print("\nBoosting Test set size: ", testLen)

    res = []
    clfCnt = len(forest)
    for i in range(0, clfCnt):
        res.append(svm.validate(forest[i], topSet, X_test))
        # print(res[i][:10])

    boostRes = getBoostRes(res, betas)
    detail = "iter = %d, feature_num = %d, test_percent = %.2f" % (clfCnt, len(topSet), testPercent)
    showTestResult(boostRes, y_test, clType='Boosting_SVM', title=detail)

    return forest, betas
    
# validate
def validateDTree(boostClf, betas, topSet, X):
    print("\nBoosting DTree Validating...")
    print("X: ", len(X))

    res = []
    clfCnt = len(boostClf)
    for i in range(0, clfCnt):
        res.append(dtree.validate(boostClf[i], topSet, X))
    boostRes = getBoostRes(res, betas)

    return boostRes

def validateSVM(boostClf, betas, topSet, X):
    print("\nBoosting SVM Validating...")
    print("X: ", len(X))

    res = []
    clfCnt = len(boostClf)
    for i in range(0, clfCnt):
        res.append(svm.validate(boostClf[i], topSet, X))
    boostRes = getBoostRes(res, betas)

    return boostRes

if __name__ == '__main__':
    (X, Y) = getTrainData()
    topSet = genTopWordSet(X, Y, topWordSize)

    forest, betas = boostDtree(topSet, X, Y)
    # forest = boostSVM(topSet, X, Y)

    X_valid = getValidData()
    y_valid = validateDTree(forest, betas, topSet, X_valid)
    # y_valid = validateSVM(forest, betas, topSet, X_valid)
    print(y_valid[:10])
    genSubmission(y_valid)
