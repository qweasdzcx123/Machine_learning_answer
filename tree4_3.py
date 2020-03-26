#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 17:12:11 2020

@author: rui
"""


import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['AR PL UKai CN'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
 
dataset = pd.read_excel('./WaterMelon_3.0.xlsx',encoding = 'gbk')  # 读取数据
Attributes = dataset.columns        #所有属性的名称
#print(Attributes)
m,n = np.shape(dataset)   # 得到数据集大小
dataset = np.matrix(dataset)
for i in range(m):      # 将标签替换成 好瓜 和 坏瓜
    if dataset[i,n-1]=='是': dataset[i,n-1] = '好瓜'
    else : dataset[i,n-1] = '坏瓜'
attributeList = []       # 属性列表，每一个属性的取值，列表中元素是集合
for i in range(n):
    curSet = set()      # 使用集合是利用了集合里面元素不可重复的特性，从而提取出了每个属性的取值
    for j in range(m):
        curSet.add(dataset[j,i])
    attributeList.append(curSet)
#print(attributeList)
D = np.arange(0,m,1)     # 表示每一个样本编号
A = list(np.ones(n))    # 表示每一个属性是否被使用，使用过了标为 -1
A[-1] = -1              # 将数据里面的标签和编号列标记为 -1
A[0] = -1
#print(A)
#print(D)
EPS = 0.000001
 
 
class Node(object):            # 创建一个类，用来表示节点的信息
    def __init__(self,title):
        self.title = title     # 上一级指向该节点的线上的标记文字
        self.v = 1              # 节点的信息标记
        self.children = []     # 节点的孩子列表
        self.deep = 0         # 节点深度
        self.ID = -1         # 节点编号
 
def isSameY(D):                  # 判断所有样本是否属于同一类
    curY = dataset[D[0],n-1]
    for i in range(1,len(D)):
        if dataset[D[i],n-1] != curY:
            return  False
    return True
 
def isBlankA(A):              # 判断 A 是否是空，是空则返回true
    for i in range(n):
        if A[i]>0: return False
    return True
 
def isSameAinD(D,A):       # 判断在D中，是否所有的未使用过的样本属性均相同
    for i in range(n):
        if A[i]>0:#未使用属性
            for j in range(1,len(D)):
                if not isSameValue(dataset[D[0],i],dataset[D[j],i],EPS):  ###D代表样本列表
                    return False
    return True
 
def isSameValue(v1,v2,EPS):            # 判断v1、v2 是否相等
    if type(v1)==type(dataset[0,8]):  #如果为数值型则相减
        return abs(v1-v2)<EPS
    else: return  v1==v2
 
def mostCommonY(D):             # 寻找D中样本数最多的类别
    res = dataset[D[0],n-1]     # D中第一个样本标签
    maxC = 1
    count = {}
    count[res] = 1              # 该标签数量记为1
    for i in range(1,len(D)):
        curV = dataset[D[i],n-1]   # 得到D中第i+1个样本的标签
        if curV not in count:      # 若之前不存在这个标签
            count[curV] = 1        # 则该标签数量记为1
        else:count[curV] += 1      # 否则 ，该标签对应的数量加一
        if count[curV]>maxC:       # maxC始终存贮最多标签对应的样本数量
            maxC = count[curV]     # res 存贮当前样本数最多的标签类型
            res = curV
    return res             # 返回的是样本数最多的标签的类型
 
def entropyD(D):       # 参数D中所存的样本的交叉熵　信息熵算法
    types = []         # 存贮类别标签
    count = {}         # 存贮每个类别对应的样本数量
    for i in range(len(D)):           # 统计D中存在的每个类型的样本数量
        curY = dataset[D[i],n-1]
        if curY not in count:
            count[curY] = 1
            types.append(curY)
        else:
            count[curY] += 1
    ans = 0
    total = len(D)              # D中样本总数量
    for i in range(len(types)): # 计算交叉熵
        ans -= count[types[i]]/total*math.log2(count[types[i]]/total)
    return ans   ##返回信息熵
 
def gain(D,p):        # 属性 p 上的信息增益
    if type(dataset[0,p])== type(dataset[0,8]):   # 判断若是连续属性，则调用另一个函数
        res,divideV = gainFloat(D,p)
    else:
        types = []
        count = {}
        for i in range(len(D)):  # 得到每一个属性取值上的样本编号
            a = dataset[D[i],p]
            if a not in count:
                count[a] = [D[i]]
                types.append(a)
            else:
                count[a].append(D[i])
        res = entropyD(D)              # D的交叉熵
        total = len(D)
        for i in range(len(types)):    # 计算出每一个属性取值分支上的交叉熵，再计算出信息增益
            res -= len(count[types[i]])/total*entropyD(count[types[i]])  ## Ent(D)-(1-len(types))(D^v/D)*Ent(D^v)
        divideV = -1000              # 这个只是随便给的一个值，没有实际意义
    return res,divideV
 
def gainFloat(D,p):            # 获得在连续属性上的最大信息增益及对应的划分点　　　每两个连续值划定一个划分点，将数据划分为两个属性,计算得到此划分下属性的信息增益,选出信息增益最大值的,返回其信息增益和划分点
    a = []
    for i in range(len(D)):    # 得到在该属性上的所有取值
        a.append(dataset[D[i],p])
    a.sort()     # 排序
    T = []
    for i in range(len(a)-1):       # 计算每一个划分点 　
        T.append((a[i]+a[i+1])/2)
    res = entropyD(D)               # D的交叉熵
    ans = 0
    divideV = T[0]
    for i in range(len(T)):         # 循环根据每一个分割点进行划分
        left = []
        right = []
        for j in range(len(D)):     # 根据特定分割点将样本分成两部分
            if (dataset[D[j],p]<=T[i]):
                left.append(D[j])
            else:right.append(D[j])
        temp = res-entropyD(left)-entropyD(right)    # 计算特定分割点下的信息增益
        if temp>ans:
            divideV = T[i]     # 始终存贮产生最大信息增益的分割点
            ans = temp         # 存贮最大的信息增益
    return ans,divideV
 
def treeGenerate(D,A,title):
    #　D总样本数　A包含所有属性，每个属性包含不同的属性
    #1.生成节点
    node = Node(title)
    # 情况１　D中所有样本是否属于同一类　返回此节点　
    if isSameY(D):             
        node.v = dataset[D[0],n-1]##返回标签
        return node
 
    # 情况２　是否所有属性全部使用过  或者  D中所有样本的未使用的属性均相同
    if isBlankA(A) or isSameAinD(D,A):
        node.v =mostCommonY(D) # 此时类别标记为样本数最多的类别（暗含可以处理存在异常样本的情况）
        return node              # 否则所有样本的类别应该一致
    
    #情况３　　　　对当前样本D进行信息增益计算
    entropy = 0
    floatV = 0
    p = 0
    for i in range(len(A)):      # 循环遍历A,找可以获得最大信息增益的属性
        if(A[i]>0): ###未使用过的特征　进行Entropy计算　取最大值的特征并记录此特征
            curEntropy,divideV = gain(D,i)
            if curEntropy > entropy:
                p = i                     # 存贮属性编号
                entropy = curEntropy
                floatV = divideV
'''
             2.递归节点
            如果是离散属性　如瓜表面特征
            ①定义当前节点v和获取所有特征,对样本D进行遍历，找到含有该属性中某特征的样本集Dv
            ②如果未找到Dv,则将该分支作为叶子，类别是D中样本数最多的类别
            ③如果找到Dv,将A中对应属性赋值－1表示已经读取过，将当前节点指向递归传上来的节点
            如果是连续属性　已找到信息增益最大值的划分点
            ①定义当前节点v
            ②判断特征值是否大于划分点从而获取到Dleft,Dright样本子集
            ③将当前节点指向左右传上来的节点
'''
    if isSameValue(-1000,floatV,EPS):
        node.v = Attributes[p]+"=?"   # 节点信息特征
        curSet = attributeList[p]    # 该属性的所有取值
        for i in curSet:
            Dv = []
            for j in range(len(D)):     # 获得该属性取某一个值时对应的样本标号
                if dataset[D[j],p]==i:
                    Dv.append(D[j]) ###取值为某属性值的所有样本Dv
 
            # 若该属性取值对应没有符合的样本，则将该分支作为叶子，类别是D中样本数最多的类别
            # 其实就是处理在没有对应的样本情况下的问题。那就取最大可能性的一类。
            if Dv==[]:
                nextNode = Node(i)
                nextNode.v = mostCommonY(D)
                node.children.append(nextNode)
            else:     # 若存在对应的样本，则递归继续生成该节点下的子树
                newA = copy.deepcopy(A)    # 注意是深度复制，否则会改变A中的值
                newA[p]=-1  #用过的属性值标记为-1
                node.children.append(treeGenerate(Dv,newA,i))
    else:   # 若对应的是连续的属性 如数字
        Dleft = []
        Dright = []
        node.v = Attributes[p]+"<="+str(floatV)+"?"     # 节点信息
        for i in range(len(D)):       # 根据划分点将样本分成左右两部分
            if dataset[D[i],p]<=floatV: Dleft.append(D[i])
            else: Dright.append(D[i])
        node.children.append(treeGenerate(Dleft,A[:],"是"))    # 左边递归生成子树，是 yes 分支
        node.children.append(treeGenerate(Dright,A[:],"否"))    # 同上。 注意，在此时没有将对应的A中值变成 -1
    return node                                                # 因为连续属性可以使用多次进行划分
 
def countLeaf(root,deep):
    root.deep = deep
    res = 0
    if root.v=='好瓜' or root.v=='坏瓜':   # 说明此时已经是叶子节点了，所以直接返回
        res += 1
        return res,deep
    curdeep = deep             # 记录当前深度
    for i in root.children:    # 得到子树中的深度和叶子节点的个数　广度优先
        a,b = countLeaf(i,deep+1)
        res += a
        if b>curdeep: curdeep = b
    return res,curdeep
 
def giveLeafID(root,ID):         # 给叶子节点编号
    if root.v=='好瓜' or root.v=='坏瓜':
        root.ID = ID
        ID += 1
        return ID
    for i in root.children:
        ID = giveLeafID(i,ID)
    return ID
 
def plotNode(nodeTxt,centerPt,parentPt,nodeType):     # 绘制节点
    plt.annotate(nodeTxt,xy = parentPt,xycoords='axes fraction',xytext=centerPt,
                 textcoords='axes fraction',va="center",ha="center",bbox=nodeType,
                 arrowprops=arrow_args)
 
def dfsPlot(root,cnt,leafnode,decisionnode,deep):
    if root.ID==-1:          # 说明根节点不是叶子节点
        childrenPx = []
        meanPx = 0
        for i in root.children:
            cur = dfsPlot(i,cnt,leafnode,decisionnode,deep)
            meanPx += cur
            childrenPx.append(cur)
        meanPx = meanPx/len(root.children)
        c = 0
        for i in root.children:
            nodetype = leafnode
            if i.ID<0: nodetype=decisionnode
            plotNode(i.v,(childrenPx[c],0.9-i.deep*0.8/deep),(meanPx,0.9-root.deep*0.8/deep),nodetype)
            plt.text((childrenPx[c]+meanPx)/2,(0.9-i.deep*0.8/deep+0.9-root.deep*0.8/deep)/2,i.title)
            c += 1
        return meanPx
    else:
        return 0.1+root.ID*0.8/(cnt-1)
 
 
decisionNode = dict(boxstyle = "sawtooth",fc = "0.9",color='blue')
leafNode = dict(boxstyle = "round4",fc="0.9",color='red')
arrow_args = dict(arrowstyle = "<-",color='green')