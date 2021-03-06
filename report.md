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
- 分类器库：[sklearn 0.19.1](http://scikit-learn.org/stable/index.html)

## 设计实现

### 概述

整个设计实现全部独立完成。大致经过了以下阶段：

1. 读取原始数据，数据处理，特征提取（使用了词频作为特征，预处理中对否定做了处理）
2. 在已有库的基础上实现基础分类器：dtree和svm
3. 实现两种ensemble learning算法与上述两种分类器的组合
4. 对代码进行一定程度的重构，更好地支持模块化和命令行参数式的参数调节
5. 通过不同参数下的实验收集数据，比较效果
6. 尝试其他分类器：Naive Bayes
7. 尝试其他特征：使用sklearn提供的tfidf特征

详细各阶段的设计实现见后文。

### 基本任务实现

#### 1. 数据读取与预处理

数据读取中，使用了正则表达式进行信息抽取，即从原始文本中提取评价信息。需要说明的是，对于训练集，需要同时返回评价内容及其标签，对验证集则只需返回评价内容。

预处理中，我主要对否定词做了处理。具体做法是：在否定词的到最近的标点之间的所有词添加NOT_前缀（所有否定词后的词极性取反，最早由Das, Mike Chen et.al 在Yahoo! For Amazon: Extracting market sentiment from stock message boards. 2001. APFA中提出），参考[Stanford NLP学习笔记：7. 情感分析（Sentiment）](https://www.cnblogs.com/arkenstone/p/6064196.html)。处理效果的例子如下：

```txt
    before:
        不 贵 ，
    after:
        NOT_贵 ，
```

另外，还考虑了停用词以及常见的标点符号。在预处理中，一方面完成了词性的处理，一方面提出了标点符号，最后每条评论转化为以词为单位的列表。如：

```py
    ['六十多块', '钱', 'NOT_贵', '质量', '不错', '！']
```

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

在实现时，利用基础分类器统一接口的优势，可以直接将需要使用的分类器module（自己定义在.py文件中）名作为参数传入，避免了重复代码，结构化良好。如：

```python
    def bagging(clfType, topSet, X, Y)
    def validate(clfType, bag, topSet, X)
```

其中，clfType即为基础分类器的类型，由于我自己对sklearn提供的分类器进行了一次封装，产生了svm和dtree两个module，因此在这里，调用时直接将module名作为参数传入即可。

另外，使用getopt库，支持命令行参数调节，可以指定如bagging中分类器数量等参数，便于之后利用脚本编写批量测试。如：

```python
    opts, args = getopt.getopt(sys.argv[1:], "hc:n:t:r:")
    for name, value in opts:
        if name == '-h':
            print("Usage: %s -c [dtree|svm|bayes] -t topWordSize -n clfNum -r sampleRate" % sys.argv[0])
            exit(-1)
        # elif other options...
```

具体的算法均以PPT上的描述为准实现。

##### Bagging

###### 1. 原理

Bagging的原理是，利用在同一训练集上得到的基于同一模型的若干分类器，在预测时，通过公平投票机制，确定最终的分类结果。

Bagging的要点在于，对不同的分类器，在同一训练集上通过bootstrap采样进行训练。那么，需要确定一个参数，即采样数量，这里用`sampleRate`描述，意为采样数量占训练集比例。另一个参数是要使用的分类器数量，考虑到投票方式，宜选用奇数。

###### 2. 实现

采样实现在`sampling()`函数中，通过调用random库的`sample`函数生成随机序号列表:

```python
    splNum = random.sample(range(0, total), size)
```

利用这个列表中的序号，依次取出原训练集的中对应编号的元素构成子集，即完成采样。

此外，得到最终结果的方法实现在`maxCntRes()`函数中，为公平投票，即对每个待预测样本，对所有分类器结果在各输出累加，取出累加结果最大的标签（argmax），实现如下，其中参数`result`为所有单个分类器给出的对每个样本的预测结果，也即，`result[j][i]`为第j个分类器对第i个样本给出的结果：

```python
        finRes = []
        # for each sample
        for i in range(0, resSize):
            dic = {}
            # for result given by seperate clf
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

bagging分类器建立过程为，对每个分类器，调用`sampling()`进行采样，并用采样得到的子集进行训练。为了增加随机性，在确定采样大小时，我还使其有一定变化范围，即给定采样率的一半到采样率的区间，当然，丢弃部分样本也有助于防止过拟合。而在测试&验证时，用每个分类器给出各自的结果，再按照上述公平投票得到最终结果。相关实现参考如下，其中`clfType`为分类器module名，如dtree：

```python
    # generate
    sampleSize = int(trainLen * sampleRate)
    forest = []
    # for each clf
    for i in range(0, dtNum):
        # sampling
        X_smpl, y_smpl = sampling(X_train, y_train, random.randint(int(sampleSize / 2), sampleSize))
        tree = clfType.train(topSet, X_smpl, y_smpl, test_size=0.1)
        forest.append(tree)

    # test
    testLen = len(y_test)
    res = []
    for i in range(0, dtNum):
        res.append(clfType.validate(forest[i], topSet, X_test))
    baggingRes = maxCntRes(res)
```

##### AdaBoost.M1

###### 1. 原理

AdaBoost.M1是AdaBoost的多分类版本。其原理是通过调节样本权重，使分类器更专注于容易出错的训练样本，即增大“更难”分类的训练样本的权重，由此经过多轮迭代，得到若干个分类器。其具体过程为，从各样本权重相等为 $1 / N$ 开始，第 $t$ 轮训练后计算错误率$ε_t$值为所有错误分类样本的权重之和， 若$ε_t > 0.5$则直接退出，否则，计算因子 $β_t = \frac{ε_t}{1 - ε_t}$，然后对所有训练样本重新赋权重：

- 若分类正确，则 $w_{new} = w_{old} * β_t$
- 若分类错误，则 $w_{new} = w_{old}$

最后对全部训练样本的权值再次归一化。并将新的权值用于第 $t+1$ 轮训练。注意，每轮训练完成后，需要保存该轮的分类器和对应的$β_t$。

在预测时，使用“不公平投票”方法。每个分类器给出各自结果，各结果以各分类器对应的 $β_t$ 为权重进行投票累加（作为对比，bagging中权重均为1），最终取得到票数最多的类别为预测结果。

###### 2. 实现

Boosting的总体实现方法与Bagging十分类似，其核心在于，对于每个（或者说每轮）分类器，在训练时训练样本的权重是不同的。与Bagging每次的重采样相应的，Boosting对每个分类器，都需要进行re-weighting。在re-weighting中，通过比对输出结果与原始label，累加错误结果的权重得到`err`，若`err` > 0.5则直接退出，否则执行 re-weight 。re-weight原理如前所述，具体实现如下：

```python
    # reWeight
    beta = err / (1.0 - err)
    # new weight for samples
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
    for i in range(0, total):
        neoWeight[i] = neoWeight[i] / weightSum

    return neoWeight, beta
```

至于权值如何在训练中使用，这里直接调用了sklearn提供的分类器接口中的`sample_weight`参数，将上面得到的权值list传入即可。

### 拓展任务实现

#### 1. Ensemble Learning中各参数的影响

##### Bagging

Bagging中使用的参数主要包括：分类器数量`clf num`，特征截断后的特征大小`feature size`（与单个分类器相关，本身不是Ensemle Learning算法的参数），和bootstrap采样时的采样率`sample rate`。以下基于前文使用的特征和实现方法，进行了批量测试，得到如下统计结果：

【注意1】如前文对特征的介绍，特征采用的是对三种标签下高频词集合的并集，因此可能会出现“不够整”的数字，如设置各标签下取top 50，最终特征维度可能为82

【注意2】以下结果为在测试集上运行的结果，而非使用validation集得到的结果。

###### 1. DTree

1. `feature size`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|1|3|82|0.50|0.7542|
|2|3|158|0.50|0.6913|
|3|3|308|0.50|0.6599|
|4|3|724|0.50|0.6840|
|5|3|1069|0.50|0.6491|
|6|3|1416|0.50|0.6531|
|7|3|2903|0.50|0.6597|

可以看到，随着特征数量增加，总体而言RMSE降低，效果是在提升的。但从第6次实验开始，RMSE开始增加，可能过多的feature造成了重要特征被稀释。

2. `clf num`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|8|3|1416|0.50|0.6498|
|9|5|1416|0.50|0.6509|
|10|7|1416|0.50|0.6305|
|11|9|1416|0.50|0.6200|
|12|11|1416|0.50|0.6260|
|13|13|1416|0.50|0.6147|

从测试结果来看，基本可以得出结论，随着使用的分类器数量增加，预测准确率呈上升趋势。

3. `sample rate`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|14|5|1416|0.10|0.6949|
|15|5|1416|0.20|0.6526|
|16|5|1416|0.30|0.6260|
|17|5|1416|0.50|0.6425|
|18|5|1416|0.75|0.6282|

随着采样率上升，总体而言效果提升，这是因为对每个分类器，所看到的样本数量增加。但存在的问题是，采样率上升将导致各分类器有更大的概率看到相同的样本，分类器之间逐渐趋同，可能会削弱Ensemble Learning的效果。

###### 2. SVM

1. `feature size`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|1|3|82|0.50|0.7012|
|2|3|158|0.50|0.6633|
|3|3|308|0.50|0.6238|
|4|3|724|0.50|0.5779|
|5|3|1069|0.50|0.5600|
|6|3|1416|0.50|0.5702|
|7|3|2903|0.50|0.5817|

趋势与DTree类似，随着特征数量增加，总体而言效果提升，但当特征是了超过1416后，效果反而有所下降。

2. `clf num`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|8|3|1416|0.50|0.5724|
|9|5|1416|0.50|0.5779|
|10|7|1416|0.50|0.5710|
|11|9|1416|0.50|0.5676|
|12|11|1416|0.50|0.5618|
|13|13|1416|0.50|0.5656|

与DTree类似，总体而言，随着使用的分类器数量增加，预测准确率呈上升趋势。

3. `sample rate`

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|14|5|1416|0.10|0.6391|
|15|5|1416|0.20|0.5900|
|16|5|1416|0.30|0.5846|
|17|5|1416|0.50|0.5636|
|18|5|1416|0.75|0.5448|

随着采样率上升，总体而言效果提升。不过需要注意，随着采样率提高，SVM的同质化将比DTree更严重。

另外，根据上面两组数据对比，可以看到在相同参数下，SVM有着相较DTree更好的表现。


##### AdaBoost.M1

Boost中使用的参数主要包括：最大迭代轮数`max iteration`，特征截断后的特征大小`feature size`（与单个分类器相关，本身不是Ensemle Learning算法的参数）。但是需要注意，实际迭代轮数`real iteration`也需要考虑，因为在分类器本身较弱时，经常出现未达到最大迭代轮数就已经退出的情况。测试结果如下：

###### 1. DTree

1. `feature size`

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|1|5|2|82|0.7792|
|2|5|3|158|0.7209|
|3|5|2|308|0.6939|
|4|5|4|724|0.6686|
|5|5|1|1069|0.6775|
|6|5|3|1416|0.6730|
|7|5|2|2903|0.6550|

可以看到，随着特征数量增加，总体而言RMSE降低，效果是在提升的。

2. `max iteration`

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|8|2|2|1416|0.6839|
|9|3|2|1416|0.6655|
|10|4|2|1416|0.7022|
|11|5|2|1416|0.6888|
|12|6|1|1416|0.6550|

很遗憾，由于分类器本身较弱，除了第8号测试达到了最大迭代轮数，其他均未超过2轮就已经退出。

###### 2. SVM

1. `feature size`

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|1|5|2|82|0.6882|
|2|5|1|158|0.6476|
|3|5|1|308|0.6158|
|4|5|2|724|0.5628|
|5|5|1|1069|0.5339|
|6|5|1|1416|0.5427|
|7|5|1|2903|0.5356|

随着特征数量增加，总体而言RMSE降低，但当特征数超过1416时，RMSE有回升趋势。

2. `max iteration`

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|8|2|2|1416|0.5394|
|9|3|2|1416|0.5477|
|10|4|1|1416|0.5366|
|11|5|1|1416|0.5477|
|12|6|1|1416|0.5462|

与DTree类似，由于分类器本身较弱，大多均未超过2轮就已经退出。这几乎完全摧毁了Boost本身希望得到的效果。

#### 2. 尝试其他分类器：Naive Bayes

如前所述，通过实现统一的接口，可以很方便地拓展使用其他基础分类器。在这里，由于使用了词频特征，因此尝试了Naive Bayes分类器，实现上直接调用了sklearn提供的MultinomialNB模型。

类似地，使用NB模型进行两种Ensemble Learning的测试，得到如下结果：

##### Bagging

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|1|3|82|0.50|1.0216|
|2|3|158|0.50|1.0009|
|3|3|308|0.50|0.9744|
|4|3|724|0.50|0.9443|
|5|3|1069|0.50|0.9676|
|6|3|1416|0.50|0.9663|
|7|3|2903|0.50|1.0009|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|8|3|1416|0.50|0.9634|
|9|5|1416|0.50|0.9440|
|10|7|1416|0.50|0.9689|
|11|9|1416|0.50|0.9660|
|12|11|1416|0.50|0.9760|
|13|13|1416|0.50|0.9467|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|14|5|1416|0.10|1.0745|
|15|5|1416|0.20|1.0209|
|16|5|1416|0.30|1.0003|
|17|5|1416|0.50|0.9712|
|18|5|1416|0.75|0.9259|

##### AdaBoost.M1

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|1|5|2|82|1.1210|
|2|5|1|158|1.1058|
|3|5|1|308|1.1080|
|4|5|1|724|1.1367|
|5|5|1|1069|1.1092|
|6|5|1|1416|1.0970|
|7|5|1|2903|1.0993|

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|8|2|1|1416|1.1123|
|9|3|1|1416|1.1076|
|10|4|1|1416|1.1058|
|11|5|2|1416|1.0995|
|12|6|4|1416|1.0818|

##### 分析

首先，可以看到NB模型的效果远不如之前的两种模型，RMSE非常高，且调节参数也很难降低。

其次，针对各项参数的调节，虽然由于基础分类器本身的原因，效果不显著，但总体趋势仍与之前分析的相同。

#### 3. 尝试其他特征：TF-IDF

##### 原理

前文中，使用的特征是基于词频高低得到的词频最高的词，但事实上，在搜索引擎等领域，tf-idf是一种常用的文本特征，它与前面使用的单纯的词频有所不同。

tf（erm frequency）就是一般意义上的词频，即一个词在文档中出现的频率:
$$tf_{d,t}=\frac{n_{d,t}}{∑_kn_{d,k}}$$
其中，$tf_{d,t}$表示词语 t 在文档d中出现的频率。$n_{d,t}$表示词语 t 在文档d中出现的次数

idf（inverse document frequency）则衡量一个词语的“普遍重要性”，也就是它在整体所有文档中的重要程度，而词项t的idf的度量是
$$idf_t=1+\frac{log|D|}{df_t}$$
其中，$df_t$ 表示包含词语t的document个数，$|D|$ 是语料库中的document总数.

##### 实现

在实现上，我本来试图自行实现TF-IDF特征提取和生成，但在完成相应代码后，实际运行却发现，由于特征矩阵过大，导致运行中经常出现Memory Error，可用性很低，难以测试。这部分代码我予以了保留，位于`feature.py`中，以注释`# self-implementation`标记。

于是考虑使用sklearn提供的`TfidfVectorizer`进行特征提取。作为库，sklearn对相应的算法有着更好的优化，避免了过大的特征矩阵导致的问题。实现中，停用词直接调用sklearn提供的参数`stop_words`指定为前文提到的停用词列表，并且可以通过参数`max_features`指定需要的最大特征数量，与前文的指定topWord大小类似。

通过`TfidfVectorizer`，对训练集应用`fit_transform()`，得到词汇表vocabulary，所有特征词语列表words（用于确保顺序），和训练集的特征feature。feature可以直接用于分类器的训练。而vocabulary用于对验证集的特征抽取，只需要在生成`TfidfVectorizer`对象时指定`vocabulary`参数即可。

##### 测试结果

使用类似前面的测试，得到如下结果：

###### Bagging

**DTree**

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|1|3|50|0.50|0.7904|
|2|3|100|0.50|0.7139|
|3|3|200|0.50|0.6619|
|4|3|500|0.50|0.6587|
|5|3|750|0.50|0.6435|
|6|3|1000|0.50|0.6433|
|7|3|2000|0.50|0.6361|
|8|3|5000|0.50|0.6477|
|9|3|10000|0.50|0.6481|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|10|3|1000|0.50|0.6371|
|11|5|1000|0.50|0.6350|
|12|7|1000|0.50|0.6262|
|13|9|1000|0.50|0.6183|
|14|11|1000|0.50|0.5902|
|15|13|1000|0.50|0.6280|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|16|5|1000|0.10|0.6728|
|17|5|1000|0.20|0.6396|
|18|5|1000|0.30|0.6462|
|19|5|1000|0.50|0.6211|
|20|5|1000|0.75|0.6167|

在各类参数情况下，使用tf-idf特征的结果略好于之前直接使用词频的结果。

**SVM**

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|1|3|50|0.50|0.7709|
|2|3|100|0.50|0.6785|
|3|3|200|0.50|0.6476|
|4|3|500|0.50|0.5822|
|5|3|750|0.50|0.5475|
|6|3|1000|0.50|0.5569|
|7|3|2000|0.50|0.5608|
|8|3|5000|0.50|0.5724|
|9|3|10000|0.50|0.5498|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|10|3|1000|0.50|0.5557|
|11|5|1000|0.50|0.5456|
|12|7|1000|0.50|0.5684|
|13|9|1000|0.50|0.5650|
|14|11|1000|0.50|0.5704|
|15|13|1000|0.50|0.5479|

|No.|clf num|feature size|sample rate|RMSE|
|-:|-:|-:|-:|-:|
|16|5|1000|0.10|0.5925|
|17|5|1000|0.20|0.5931|
|18|5|1000|0.30|0.5793|
|19|5|1000|0.50|0.5682|
|20|5|1000|0.75|0.5311|

经过对比，同样可以看出使用该特征优于之前的特征，且在SVM上表现比DTree上更明显。

###### AdaBoost.M1

**DTree**

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|1|5|2|50|0.8076|
|2|5|2|100|0.7373|
|3|5|2|200|0.6877|
|4|5|3|500|0.6713|
|5|5|3|750|0.6583|
|6|5|2|1000|0.6391|
|7|5|2|2000|0.6657|
|8|5|4|5000|0.6757|
|9|5|2|10000|0.6389|

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|10|2|2|1000|0.6616|
|11|3|3|1000|0.6540|
|12|4|2|1000|0.6573|
|13|5|3|1000|0.6618|
|14|6|2|1000|0.6398|

TF-IDF特征效果仍然较优，但在boost下迭代轮数仍然较少，不够令人满意。

**SVM**

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|1|5|1|50|0.7775|
|2|5|1|100|0.7056|
|3|5|1|200|0.6074|
|4|5|1|500|0.5758|
|5|5|1|750|0.5555|
|6|5|1|1000|0.5458|
|7|5|1|2000|0.5522|
|8|5|1|5000|0.5522|
|9|5|2|10000|0.5345|

|No.|max iteration|real iteration|feature size|RMSE|
|-:|-:|-:|-:|-:|
|10|2|1|1000|0.5598|
|11|3|1|1000|0.5632|
|12|4|1|1000|0.5799|
|13|5|1|1000|0.5520|
|14|6|2|1000|0.5563|

在Boost+SVM下，TF-IDF特征与之前使用的词频特征不相上下，甚至迭代轮数还少于之前使用的特征，效果不是很好。

## 结果报告

必须承认，可能由于特征选择等原因，导致我实现的弱分类器本身性能一般，因此Ensemble Learning算法的结果也很难提高太多。现报告各组合在Kaggle上提交的结果：

- Bagging + DTree: 0.57857
- Bagging + SVM: 0.51549
- AdaBoost.M1 + DTree: 0.62823
- AdaBoost.M1 + SVM: 0.51491 (最好结果)

## 讨论

### 1. 关于基础分类器

从测试结果来看，基础分类器中，SVN效果相对较优，而DTree不如SVM。至于原因，可能由于情感分析中真正重要的、表达感情的词其实不多，决策树对于大量特征和大量样本，可能会导致随着深度增加而稀释了那些真正重要的词。使用支持向量机，则有可能更多关注于作为支持向量的少量样本，从而效果更好。

### 2. 关于Ensemble Learning算法

在实验中，由于弱分类器本身性能较差，导致Boost算法不能达到较高的迭代轮数，这直接影响了Boost的性能，可能退化到了单一分类器。但是，这其实也反映了Boost算法的一个特点，即对弱分类器本身的要求：弱分类器本身性能不能太差。

而Bagging则不受到这种影响，因为它保证一定能实现给定数目的弱分类器。

至于两种算法的异同，首先就相同点而言，都是Ensemble Learning算法，都是基于若干弱分类器，根据它们各自给出的结果通过投票方式，选出最终结果；就不同点而言，Bagging是相同模型在相同训练集（的子集）上得到若干弱分类器，Boost是同一模型通过赋予不同的样本权值进行多轮迭代，得到若干弱分类器；另外，就得到结果的方法而言，Bagging为公平投票，各弱分类器投票权值相同，而Boost为不公平投票，各弱分类器投票权值为各自β值。

### 3. 关于组合

根据之前的分析，理论上最优的是Bagging + SVM。如前所述，由于弱分类器较弱，Boost无法实现应有的性能，而对于弱分类器本身，SVM强于DTree，因此该组合应该表现最好。

但事实上根据实际提交结果，最优结果为AdaBoost.M1 + SVM得到的。分析原因，一方面其实二者相差不多，有一定随机成分在其中；另一方面，还是由于弱分类器本身性能一般，因此Bagging可能出现“一群臭皮匠顶不上诸葛亮”的情况，但Boost虽然迭代轮数少，却因为专注于较“难”的样本，有可能有较好的表现。

## 参考文献

1. [sklearn.svm.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn-svm-linearsvc)
2. [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
3. [sklearn.naive_bayes.MultinomialNB](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB)
4. [sklearn.model_selection.train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
5. [sklearn.feature_extraction.text.TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
6. [TF-IDF特征提取 用sklearn提取tfidf特征](https://blog.csdn.net/Techmonster/article/details/74905668)
7. [bootstrap, boosting, bagging 几种方法的联系](https://blog.csdn.net/chenhongc/article/details/9404583)
8. [Stanford NLP学习笔记：7. 情感分析（Sentiment）](https://www.cnblogs.com/arkenstone/p/6064196.html)
9. [python scikit-learn计算tf-idf词语权重](https://blog.csdn.net/liuxuejiang158blog/article/details/31360765)
