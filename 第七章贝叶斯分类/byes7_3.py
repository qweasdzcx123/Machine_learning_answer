#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 16:44:18 2020

@author: rui
"""
import numpy as np
import pandas as pd
import math
 
def readData():          # 读取数据
    dataset = pd.read_excel('./WaterMelon_3.0.xlsx',encoding = 'gbk')  # 读取数据
    Attributes = dataset.columns[1:]    # 属性名称列表
    dataset = np.array(dataset)
    dataset = dataset[:,1:]
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
 
 
def getClassPrior(classname,classvalue,dataset,attrNum):     # 得到类先验概率，经过拉普拉斯平滑
    count = 0
    for i in range(len(dataset)):
        if dataset[i][classname] == classvalue : count += 1
    return (count+1)/(len(dataset) + attrNum[classname])
 
def getClassCondition(classname,classvalue,classCondname,classCondvalue,dataset,attrNum):   # 得到类条件概率
    if classname=='密度'or classname=='含糖率':      # 若是连续属性，则用概率密度进行计算
        value = []
        for i in range(len(dataset)):
            if dataset[i][classCondname]==classCondvalue:
                value.append(dataset[i][classname])
        mean = np.mean(value)
        delt = np.std(value)
        return (1/(math.sqrt(2*math.pi)*delt))*math.exp(-(classvalue-mean)**2/(2*delt**2))
    else:                                             # 离散属性用频率代替概率，并进行拉普拉斯平滑
        count = 0
        count_ = 0
        for i in range(len(dataset)):
            if dataset[i][classname]==classvalue and dataset[i][classCondname]==classCondvalue:
                count += 1
            if dataset[i][classCondname]==classCondvalue : count_ += 1
        return (count+1)/(count_+attrNum[classname])
 
def main():
    test1 = {'色泽':'青绿','根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑',\
         '密度':0.697,'含糖率':0.460}
    dataset,attrNum = readData()
    Pgood = getClassPrior('好瓜','是',dataset,attrNum)
    Pbad = getClassPrior('好瓜','否',dataset,attrNum)
    for i in test1:
        Pgood *= getClassCondition(i,test1[i],'好瓜','是',dataset,attrNum)
        Pbad *= getClassCondition(i,test1[i],'好瓜','否',dataset,attrNum)
    print(Pgood,Pbad)
    print('该西瓜是%s'%('好瓜' if Pgood>Pbad else '坏瓜'))
