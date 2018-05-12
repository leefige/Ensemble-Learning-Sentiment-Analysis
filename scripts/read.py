import re

dataDir = "../data/"
trainPath = dataDir + "exp2.train.csv"
validPath = dataDir + "exp2.validation_review.csv"

def readTrainData():
    print("Reading train set...")
    pat = re.compile(r"(\-?[0|1])\s+(\S.*)")
    cnt = 0
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
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
    return (resX, resY)

def readValidData():
    print("Reading validation set...")
    pat = re.compile(r"(\d+),(\S.*)")
    cnt = 0
    resX = []
    with open(validPath, 'r', encoding='utf8') as fin:
        for line in fin:
            line = line.strip()
            reObj = pat.search(line)
            if reObj:
                review = reObj.group(2)
                resX.append(review)
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)
    return resX

if __name__ == "__main__":
    tx, ty = readTrainData()
    vx = readValidData()
    print(tx[:10])
    print(ty[:10])