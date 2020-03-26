#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:11:22 2020

@author: rui
"""

import numpy as np
 
def dataSet():
    # 西瓜数据集离散化
    X = np.mat('2,3,3,2,1,2,3,3,3,2,1,1,2,1,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,1,2,2,2,1,1;\
            2,3,2,3,2,2,2,2,3,1,1,2,2,3,2,2,3;\
            3,3,3,3,3,3,2,3,2,3,1,1,2,2,3,1,2;\
            1,1,1,1,1,2,2,2,2,3,3,3,1,1,2,3,2;\
            1,1,1,1,1,2,2,1,1,2,1,2,1,1,2,1,1;\
            0.697,0.774,0.634,0.668,0.556,0.403,0.481,0.437,0.666,0.243,0.245,0.343,0.639,0.657,0.360,0.593,0.719;\
            0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,0.057,0.099,0.161,0.198,0.370,0.042,0.103\
            ').T
    X = np.array(X)         # 样本属性集合，行表示一个样本，列表示一个属性
    Y = np.mat('1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0')
    Y = np.array(Y).T          # 每个样本对应的标签
    return X, Y
 
 
def sigmod(x):
    return 1.0/(1.0+np.exp(-x))
 
 
def bpstand(hideNum):                         # 标准的反向传播算法
    X,Y = dataSet()
    V = np.random.rand(X.shape[1],hideNum)      # 权值及偏置初始化
    V_b = np.random.rand(1,hideNum)
    W = np.random.rand(hideNum,Y.shape[1])
    W_b = np.random.rand(1,Y.shape[1])
 
    rate = 0.1
    error = 0.001
    maxTrainNum = 1000000
    trainNum = 0
    loss = 10
 
    while (loss>error) and (trainNum < maxTrainNum):
        for k in range(X.shape[0]):               # 标准bp方法一次只处理一个样本
            H = sigmod(X[k,:].dot(V)-V_b)          # 因为书上一直给出的是减去阈值，所以这里用减号。
            Y_ = sigmod(H.dot(W)-W_b)              # 其实大部分情况下人们都用的是加上偏置b这种表达方式
            loss = sum((Y[k]-Y_)**2)*0.5           # 改成加号后只需要在下面更新参数时也用加号即可
 
            g = Y_*(1-Y_)*(Y[k]-Y_)        #  计算相应的梯度，及更新参数。 此处特别注意维度的正确对应关系
            e = H*(1-H)*g.dot(W.T)
            W += rate*H.T.dot(g)
            W_b -= rate*g
            V += rate*X[k].reshape(1,X[k].size).T.dot(e)
            V_b -= rate*e
            trainNum += 1
    print("标准bp算法:")
    print("总训练次数：",trainNum)
    print("最终损失：",loss)
 
 
def bpAccum(hideNum):                   # 累积bp算法
    X,Y = dataSet()
    V = np.random.rand(X.shape[1],hideNum)
    V_b = np.random.rand(1,hideNum)
    W = np.random.rand(hideNum,Y.shape[1])
    W_b = np.random.rand(1,Y.shape[1])
 
    rate = 0.1
    error = 0.001
    maxTrainNum = 1000000
    trainNum = 0
    loss = 10
 
    while (loss>error) and (trainNum<maxTrainNum):
        H = sigmod(X.dot(V)-V_b)
        Y_ = sigmod(H.dot(W)-W_b)
        loss = 0.5*sum((Y-Y_)**2)/X.shape[0]
 
        g = Y_*(1-Y_)*(Y-Y_)                 # 对应元素相乘，类似于matlab中的点乘
        e = H*(1-H)*g.dot(W.T)
        W += rate*H.T.dot(g)
        W_b -= rate*g.sum(axis=0)
        V += rate*X.T.dot(e)
        V_b -= rate*e.sum(axis=0)
        trainNum += 1
    print("累积bp算法:")
    print("总训练次数：",trainNum)
    print("最终损失：",loss)
 
    