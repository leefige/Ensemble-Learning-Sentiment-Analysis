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

#### 1. 数据读取


