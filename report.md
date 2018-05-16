# 机器学习概论 实验二 实验报告

> 李逸飞 2015010062

## 实验内容

实现ensemble learning algorithms，并在给定的数据集上进行测试。数据集来源于淘宝评价，包括好评、中评、差评，即实现情感分析。

### 基本任务

实现以下ensemble learning algorithm和基本分类器的组合（共四种）：

- 两种ensemble learning algorithm：
    - Bagging 
    - AdaBoost.M1
- 两种基本分类器：
    - SVM 
    - Decision Tree

可以使用已有的分类器，但需要自行实现ensemble learning algorithm

### 可选任务

- 尝试其他分类器
- 分析不同feature的效果
- 调节ensemble learning algorithm的参数，分析其对性能的影响

## 平台说明

- OS：Ubuntu 17.10 (GNU/Linux 4.13.0-41-generic x86_64) - Azure虚拟机
- 语言：Python
- 运行平台：Python 3.6.3
- 分类器库：sklearn 0.19.1

## 设计实现

### 概述

整个设计实现全部独立完成。大致经过了以下阶段：

1. 读取原始数据，数据处理，特征提取（使用了词频作为特征，预处理中对否定做了处理）
2. 在已有库的基础上实现基础分类器：dtree和svm
3. 实现两种ensemble learning算法与上述两种分类器的组合
4. 对代码进行一定程度的重构，更好地支持模块化和命令行参数式的参数调节
5. 通过不同参数下的实验收集数据，比较效果
6. 尝试Naive Bayes分类器
7. 尝试使用sklearn提供的tfidf特征

详细各阶段的设计实现见后文。

### 基本任务

#### 1. 数据读取与预处理

数据读取中，使用了正则表达式进行信息抽取，即从原始文本中提取评价信息。需要说明的是，对于训练集，需要同时返回评价内容及其标签，对验证集则只需返回评价内容。

预处理中，我主要对否定词做了处理。具体做法是：在否定词的到最近的标点之间的所有词添加NOT_前缀（所有否定词后的词极性取反，最早由Das, Mike Chen et.al 在Yahoo! For Amazon: Extracting market sentiment from stock message boards. 2001. APFA中提出），参考[Stanford NLP学习笔记：7. 情感分析（Sentiment）](https://www.cnblogs.com/arkenstone/p/6064196.html)

另外，还考虑了停用词以及常见的标点符号。在预处理中，一方面完成了词性的处理，一方面提出了标点符号，最后每条评论转化为以词为单位的列表。

#### 2. 特征生成

如前所述，在这里使用的特征为，在特定vocabulary上，统计评价中各词语的频率，并归一化。vocabulary通过统计所有词语的出现频率，取出指定数量的最高频词。同时还考虑了三种标签下特征词语数量的均衡，对三种标签对应样本分别取相同数量的最高频词，最后的vocabulary为三者的并集。

计算特征时，在上面生成的vocabulary上，直接对词频进行累加，最后除以评价中总词数，相当于认为vocabulary中各词语的权重相等。

取一定数量的最高频词，一方面是考虑性能，如果取全部词汇表运行速度极慢（可能由于我实现中优化不足），另外受限于虚拟机内存，若特征过多可能出现Memory error问题。另一方面，也是考虑到应该更多侧重于“重要”的词，这样相当于二分词汇表：“重要”的词赋权重1，其他词赋权重0.

最终生成的特征，每条评论对应一个list，元素为对应词汇的归一化词频（元素顺序需要一致，通过在统计词频生成最高频词列表时确定）。

#### 3. 基础分类器实现

直接使用了sklearn中提供的Decision Tree和LinearSVC。

值得一提的是，在训练时，通过sklearn的`train_test_split()`将训练集划分为训练集和测试集，测试集大小可以指定. 训练结果（在测试集上的表现）被写入log文件中以备查阅。

另外，我在实现时定义了统一接口：`train`, `validate`, 这样方便了之后在ensemble learning中的复用。而且基础分类器的参数也可指定，如DTree的criterion等。

#### 4. ensemble learning算法实现

在实现时，利用基础分类器统一接口的优势，可以直接将需要使用的分类器module（自己定义在.py文件中）名作为参数传入，避免了重复代码，结构化良好。

另外，使用getopt库，支持命令行参数调节，可以指定如bagging中分类器数量等参数，便于之后利用脚本编写批量测试。

算法均以PPT上的描述为准实现。

##### Bagging

Bagging的要点在于，对不同的分类器，在同一训练集上通过bootstrap采样进行训练。那么，需要确定一个参数，即采样数量，这里用`sampleRate`描述，意为采样数量占训练集比例。另一个参数是要使用的分类器数量，考虑到投票方式，宜选用奇数。

采样实现如下，通过调用random库的`sample`函数生成随机序号列表，利用这个列表中的序号取出原训练集的子集即完成采样：

```python
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
```

此外，得到最终结果的方法为公平投票，即对每个待预测样本，对所有分类器结果在各输出累加，取出累加结果最大的标签（argmax），实现如下，其中参数`result[j][i]`为第j个分类器对第i个样本给出的结果：

```python
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
```

ensemble分类器建立过程为，对每个分类器，调用`sampling`进行采样，并用采样得到的子集进行训练。为了增加随机性，在确定采样大小时，我还使其有一定变化范围，即给定采样率的一半到采样率的区间，当然，丢弃部分样本也有助于防止过拟合。而在测试&验证时，用每个分类器给出各自的结果，再按照上述公平投票得到最终结果。相关实现参考如下，其中`clfType`为分类器module名，如dtree：

```python
    # generate
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
    baggingRes = maxCntRes(res)
```

##### AdaBoost.M1
