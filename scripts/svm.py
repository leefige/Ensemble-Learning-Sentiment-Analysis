#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

from feature import *
from util import *

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

testPercent = .1

def train(topSet, X, Y, test_size=testPercent, sample_weight=None):
    X_new = genXFeature(topSet, X)
    X_arr = np.array(X_new)
    Y_arr = np.array(Y)

    # classify
    X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=test_size, random_state=0)
    print("Training...")
    print("train set:")
    print("X: ", X_train.shape)
    print("Y: ", y_train.shape)
    # print("X[0]: ", X_train[0])
    clf = LinearSVC(random_state=0)
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    # print(clf.coef_)
    # print(clf.intercept_)

    # test
    if test_size > 0:
        print("Testing...")
        print("test set:")
        print("X: ", X_test.shape)
        print("Y: ", y_test.shape)
        test_res = clf.predict(X_test)
        detail = "feature num: " + str(len(topSet)) + "\n"
        detail += "testPercent: " + str(test_size)
        showTestResult(test_res, y_test, clType='SVM', title=detail)

    return clf

def validate(clf, topSet, X):
    print("Validating...")
    X_new = genXFeature(topSet, X)
    X_arr = np.array(X_new)
    print("X: ", X_arr.shape)
    return clf.predict(X_arr)

if __name__ == '__main__':
    (X, Y) = getTrainData()
    topSet = genTopWordSet(X, Y, 1000)

    clf = train(topSet, X, Y)

    X_valid = getValidData()
    y_valid = validate(clf, topSet, X_valid)
    print(y_valid[:10])
    genSubmission(y_valid)
