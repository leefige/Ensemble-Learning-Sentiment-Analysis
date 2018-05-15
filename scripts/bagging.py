import dtree, svm
from feature import *

from sklearn.model_selection import train_test_split
import numpy as np
import random

topWordSize = 150
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
            

def baggingDtree(topSet, X, Y):
    print("Bagging: DTree")
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testPercent)
    trainLen = len(X_train)
    print("Bagging Train set size: ", trainLen)

    sampleSize = int(trainLen * sampleRate)
    forest = []
    for i in range(0, dtNum):
        print("\nGenerating tree %d..." % i)
        X_smpl, y_smpl = sampling(X_train, y_train, random.randint(int(sampleSize / 2), sampleSize))
        tree = dtree.train(topSet, X_smpl, y_smpl, test_size=0.1)
        forest.append(tree)

    # test
    testLen = len(y_test)
    # X_test_new = genXFeature(topSet, X_test)
    # X_test_arr = np.array(X_test_new)
    # Y_test_arr = np.array(y_test)
    print("\nBagging Test set size: ", testLen)

    res = []
    for i in range(0, dtNum):
        res.append(dtree.validate(forest[i], topSet, X_test))
        # print(res[i][:10])

    baggingRes = maxCntRes(res)
    detail = "dtNum = %d, feature_num = %d, test_percent = %.2f, sample_size = %.2f" % (dtNum, len(topSet), testPercent, sampleRate)
    showTestResult(baggingRes, y_test, clType='Bagging_DTree', title=detail)

    return forest

def baggingSVM(topSet, X, Y):
    print("Bagging: SVM")
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testPercent)
    trainLen = len(X_train)
    print("Bagging Train set size: ", trainLen)

    forest = []
    sampleSize = int(trainLen * sampleRate)
    for i in range(0, dtNum):
        print("\nGenerating svm %d..." % i)
        X_smpl, y_smpl = sampling(X_train, y_train, random.randint(int(sampleSize / 2), sampleSize))
        tree = svm.train(topSet, X_smpl, y_smpl, test_size=0.1)
        forest.append(tree)

    # test
    testLen = len(y_test)
    # X_test_new = genXFeature(topSet, X_test)
    # X_test_arr = np.array(X_test_new)
    # Y_test_arr = np.array(y_test)
    print("\nBagging Test set size: ", testLen)

    res = []
    for i in range(0, dtNum):
        res.append(svm.validate(forest[i], topSet, X_test))
        # print(res[i][:10])

    baggingRes = maxCntRes(res)
    detail = "dtNum = %d, feature_num = %d, test_percent = %.2f, sample_size = %.2f" % (dtNum, len(topSet), testPercent, sampleRate)
    showTestResult(baggingRes, y_test, clType='Bagging_SVM', title=detail)

    return forest
    
# validate
def validate(bag, topSet, X):
    print("\nBagging Validating...")
    # X_new = genXFeature(topSet, X)
    # X_arr = np.array(X_new)
    print("X: ", len(X))

    res = []
    for i in range(0, dtNum):
        res.append(dtree.validate(bag[i], topSet, X))
    baggingRes = maxCntRes(res)

    return baggingRes

if __name__ == '__main__':
    (X, Y) = getTrainData()
    topSet = genTopWordSet(X, Y, topWordSize)

    forest = baggingDtree(topSet, X, Y)
    # forest = baggingSVM(topSet, X, Y)

    X_valid = getValidData()
    y_valid = validate(forest, topSet, X_valid)
    print(y_valid[:10])
    genSubmission(y_valid)