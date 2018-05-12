from read import *

negPrefix = "NOT_"

negaWords = set(["不", "不是", "没", "没有", "木有"])
stopWord = set(["的", "你", "我", "地", "得", "了", "很"])
punc = set(['，', '。', '！', '~', '？', '、', '…', '~', '《', '》', '“', '”', '‘', '’', '?', '!', ',', '.', '...', '@', '#', '*', '$', '￥', '%'])

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

# get feature of train
def getTrainFeature():
    (X, Y) = readTrainData()
    X_no_neg = parseNeg(X)
    return (X_no_neg, Y)

# get feature of valid
def getValidFeature():
    X = readValidData()
    return parseNeg(X)

if __name__ == '__main__':
    (X, Y) = readTrainData()
    X_ = parseNeg(X)
    print(X_[:20])
