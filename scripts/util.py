#!/usr/bin/env python3
#-*- coding: UTF-8 -*-

import re
from datetime import datetime
import math

dataDir = "../data/"
trainPath = dataDir + "exp2.train.csv"
validPath = dataDir + "exp2.validation_review.csv"

resDir = "../result/"

logDir = "../log/"

def readTrainData():
    print("Reading train set...")
    pat = re.compile(r"(\-?[0|1])\s+(\S.*)")
    # cnt = 0
    resX = []
    resY = []
    with open(trainPath, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.strip()
            reObj = pat.match(line)
            if reObj:
                resY.append(int(reObj.group(1)))
                review = reObj.group(2)
                resX.append(review)
            # cnt += 1
            # if cnt % 10000 == 0:
            #     print(cnt)
    return (resX, resY)

def readValidData():
    print("Reading validation set...")
    pat = re.compile(r"(\d+),(\S.*)")
    # cnt = 0
    resX = []
    with open(validPath, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.strip()
            reObj = pat.search(line)
            if reObj:
                review = reObj.group(2)
                resX.append(review)
            # cnt += 1
            # if cnt % 1000 == 0:
            #     print(cnt)
    return resX

def genSubmission(result):
    submitPath = resDir + "submit_%s.csv" % (datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(submitPath, 'w', encoding='utf8') as fout:
        fout.write("id,label\n")
        cnt = len(result)
        for i in range(0, cnt):
            fout.write("%d,%s\n" % (i+1, str(result[i])))
    return

def showTestResult(test_res, y_test, log=True, clType='DTree', title=''):
    testSize = len(y_test)
    correct = 0
    RMSE = 0.0
    for i in range(0, testSize):
        if test_res[i] == y_test[i]:
            correct += 1
        RMSE += math.pow(test_res[i] - y_test[i], 2)
    
    accuracy = float(correct) / testSize
    RMSE /= testSize
    RMSE = math.sqrt(RMSE)
    print("Test accuracy: %.4f" % accuracy)
    print("Test RMSE: %.4f" % RMSE)

    if log:
        logPath = logDir + clType + ".log"
        with open(logPath, 'a', encoding='utf8') as fout:
            fout.write(title + '\n')
            fout.write("Test accuracy: %.4f\n" % accuracy)
            fout.write("Test RMSE: %.4f\n" % RMSE)
            fout.write("------------------------------------\n\n")
    return

if __name__ == "__main__":
    tx, ty = readTrainData()
    vx = readValidData()
    print(tx[:10])
    print(ty[:10])