#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:34:30 2020

@author: rui
"""

import numpy as np
import pandas as pd
 
def readData():          # 读取数据，只取离散属性
    dataset = pd.read_excel('./WaterMelon_3.0.xlsx',encoding = 'gbk')  # 读取数据
    Attributes = np.hstack((np.array(dataset.columns[1:-3]),np.array(dataset.columns[-1])))   # 属性名称列表
    dataset = np.array(dataset)
    dataset = np.hstack((dataset[:,1:-3],np.reshape(dataset[:,-1],newshape=(len(dataset[:,-1]),1))))
    m,n = np.shape(dataset)
    dataList = []
    for i in range(m):      # 生成数据列表，列表元素是集合类型
        curset = {}
        for j in range(n):
            curset[Attributes[j]] = dataset[i,j]
        dataList.append(curset)
 
    attrNum = {}            # 统计每个属性的可取值个数
    for i in range(n):
        curSet = set()      # 使用集合是利用了集合里面元素不可重复的特性，从而提取出了每个属性的取值
        for j in range(m):
            curSet.add(dataset[j,i])
        attrNum[Attributes[i]] = len(curSet)
    return dataList,attrNum
 
 
def getClassPrior(classname1,classvalue1,classname2,classvalue2,dataset,attrNum):     # 得到类先验概率，经过拉普拉斯平滑
    count = 0
    for i in range(len(dataset)):
        if dataset[i][classname1] == classvalue1 and dataset[i][classname2] == classvalue2 : count += 1
    return (count+1)/(len(dataset) + attrNum[classname1]*attrNum[classname2])
 
 
def getClassCondition(classname1,classvalue1,classname2,classvalue2,classname,classvalue,dataset,attrNum):   # 得到类条件概率
    count = 0
    count_ = 0
    for i in range(len(dataset)):
        if dataset[i][classname1]==classvalue1 and dataset[i][classname2] == classvalue2 and dataset[i][classname]==classvalue:
            count += 1
        if dataset[i][classname1]==classvalue1 and dataset[i][classname2] == classvalue2 : count_ += 1
    return (count+1)/(count_+attrNum[classname])
 
 
def main():
    test1 = {'色泽':'青绿','根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑'}
    dataset,attrNum = readData()
    good = 0
    bad = 0
    for j in test1:
        Pgood = getClassPrior('好瓜','是',j,test1[j],dataset,attrNum)
        Pbad = getClassPrior('好瓜','否',j,test1[j],dataset,attrNum)
        for i in test1:
            Pgood *= getClassCondition(j,test1[j],'好瓜','是',i,test1[i],dataset,attrNum)
            Pbad *= getClassCondition(j,test1[j],'好瓜','否',i,test1[i],dataset,attrNum)
    good += Pgood
    bad += Pbad
    print(good,bad)
    print('该西瓜是%s'%('好瓜' if good>bad else '坏瓜'))
