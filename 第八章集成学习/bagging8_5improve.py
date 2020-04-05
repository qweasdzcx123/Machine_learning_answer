# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
 
class Bagging:
    # 导入数据
    def loadData(self):
        dataset = pd.read_excel('./WaterMelon_3.0.xlsx',encoding = 'gbk')  # 读取数据
        Attributes = dataset.columns         # 所有属性的名称
        m,n = np.shape(dataset)              # 得到数据集大小
        dataset = np.matrix(dataset)
        for i in range(m):                  # 将标签替换成 好瓜 1 和 坏瓜 -1
            if dataset[i,n-1]=='是': dataset[i,n-1] = 1
            else : dataset[i,n-1] = -1
        self.future = Attributes[1:n-1]      # 特征名称（属性名称）
        self.x = dataset[:,1:n-1]            # 样本
        self.y = dataset[:,n-1].flat         # 实际标签
        self.m = m                           # 样本个数
 
    def __init__(self,T):
        self.loadData()
        self.T = T                  # 迭代次数
        self.seg_future = list()    # 存贮每一个基学习器用来划分的属性
        self.seg_value = list()     # 存贮每一个基学习器的分割点
        self.flag = list()          # 标志每一个基学习器的判断方向。
                                    # 取0时 <= value 的样本标签为1，取1时 >value 的样本标签为1
        self.w = 1.0/self.m * np.ones((self.m,))     # 初始的权重
 
    def booststrap(self):      # 自助采样
        b = []
        for i in range(self.m):
            b.append(int(np.floor(np.random.uniform(0,17))))
        for i in range(self.m):
            count = b.count(i)
            self.w[i] = count/self.m
 
    # 计算交叉熵
    def entropyD(self,D):          # D 表示样本的编号，从0到16
        pos = 0.0000000001
        neg = 0.0000000001
        for i in D:
            if self.y[i]==1: pos = pos + self.w[i]      # 标签为1的权重
            else: neg = neg + self.w[i]                 # 标签为-1的权重
        P_pos = pos/(pos+neg)                           # 标签为1占的比例
        P_neg = neg/(pos+neg)                           # 标签为-1占的比例
        ans = - P_pos * math.log2(P_pos) - P_neg * math.log2(P_neg)      # 交叉熵
        return ans
 
    # 获得在连续属性上的最大信息增益及对应的划分点
    def gainFloat(self,p):            # p为对应属性编号（0表示密度，1表示含糖率）
        a = []
        for i in range(self.m):      # 得到所有属性值
            a.append(self.x[i,p])
        a.sort()                      # 排序
        T = []
        for i in range(len(a)-1):    # 计算每一个划分点
            T.append(round((a[i]+a[i+1])/2,4))
        res = self.entropyD([i for i in range(self.m)])     # 整体交叉熵
        ans = 0
        divideV = T[0]
        for i in range(len(T)):         # 循环根据每一个分割点进行划分
            left = []
            right = []
            for j in range(self.m):     # 根据特定分割点将样本分成两部分
                if(self.x[j,p] <= T[i]):
                    left.append(j)
                else:
                    right.append(j)
            temp = res-self.entropyD(left)-self.entropyD(right)    # 计算特定分割点下的信息增益
            if temp>ans:
                divideV = T[i]     # 始终存贮产生最大信息增益的分割点
                ans = temp         # 存贮最大的信息增益
        return ans,divideV
 
    # 进行决策，选择合适的属性进行划分
    def decision_tree(self):
        gain_1,devide_1 = self.gainFloat(0)           # 得到对应属性上的信息增益及划分点
        gain_2,devide_2 = self.gainFloat(1)
        if gain_1 >= gain_2:                          # 选择信息增益大的属性作为划分属性
            self.seg_future.append(self.future[0])
            self.seg_value.append(devide_1)
            V = devide_1
            p = 0
        else:
            self.seg_future.append(self.future[1])
            self.seg_value.append(devide_2)
            V = devide_2
            p = 1
        left_total = 0
        right_total = 0
        for i in range(self.m):                    # 计算划分之后每一部分的分类结果
            if self.x[i,p] <= V:
                left_total = left_total + self.y[i]*self.w[i]        # 加权分类得分
            else:
                right_total = right_total + self.y[i]*self.w[i]
        if left_total > right_total:
            flagg = 0
        else:
            flagg = 1
        self.flag.append(flagg)                  # flag表示着分类的情况
 
    # 得到样本在当前基学习器上的预测
    def pridect(self):
        hlist = np.ones((self.m,))
        if self.seg_future[-1]=='密度': p = 0
        else: p = 1
        if self.flag[-1]==0:                  # 此时小于等于V的样本预测为1
            for i in range(self.m):
                if self.x[i,p] <= self.seg_value[-1]:
                    hlist[i] = 1
                else: hlist[i] = -1
        else:                                # 此时大于V的样本预测是1
            for i in range(self.m):
                if self.x[i,p] <= self.seg_value[-1]:
                    hlist[i] = -1
                else:
                    hlist[i] = 1
        return hlist
 
    def mysign(self,H):    # 改进sign函数
        h = H
        for i in range(len(H)):
            if H[i] < 0: h[i] = -1
            elif H[i]>0: h[i] = 1
            else: h[i] = int(1-2*np.round(np.random.rand()))    # 0的时候随机取值
        return h
 
    # 训练过程，进行集成
    def train(self):
        H = np.zeros(self.m)
        self.H_predict = []                        # 存贮每一个集成之后的分类结果
        for t in range(self.T):
            self.booststrap()                      # 自助采样
            self.decision_tree()                   # 得到基学习器分类结果
            hlist = self.pridect()                 # 计算该基学习器的预测值
            H = np.add(H,hlist)                # 得到 t 个分类器集成后的分类结果（加权集成）
            self.H_predict.append(self.mysign(H))
 
    # 打印相关结果
    def myPrint(self):
        tplt_1 = "{0:<10}\t{1:<10}\t{2:<10}\t{3:<10}"
        print(tplt_1.format('轮数','划分属性','划分点','何时取1？'))
        for i in range(self.T):
            if self.flag[i]==0:
                print(tplt_1.format(str(i),self.seg_future[i],str(self.seg_value[i]),
                                    'x <= V'))
            else:
                print(tplt_1.format(str(i),self.seg_future[i],str(self.seg_value[i]),
                                    'x > V'))
        print()
        print('------'*10)
        print()
        print('%-6s'%('集成个数'),end='')
        self.print_2('样本',[i+1 for i in range(17)])
        print()
        print('%-6s'%('真实标签'),end='')
        self.print_1(self.y)
        print()
        for num in range(self.T):
            print('%-10s'%(str(num+1)),end='')
            self.print_1(self.H_predict[num])
            print()
 
    def print_1(self,h):
        for i in h:
            print('%-10s'%(str(np.int(i))),end='')
 
    def print_2(self,str1,h):
        for i in h:
            print('%-8s'%(str1+str(i)),end='')
 
    # 绘图
    def myPlot(self):
        Rx = []
        Ry = []
        Bx = []
        By = []
        for i in range(self.m):
            if self.y[i]==1:
                Rx.append(self.x[i,0])
                Ry.append(self.x[i,1])
            else:
                Bx.append(self.x[i,0])
                By.append(self.x[i,1])
        plt.figure(1)
        l1, = plt.plot(Rx,Ry,'r+')
        l2, = plt.plot(Bx,By,'b_')
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.legend(handles=[l1,l2],labels=['好瓜','坏瓜'],loc='best')
        for i in range(len(self.seg_value)):
            if self.seg_future[i]=='密度':
                plt.plot([self.seg_value[i],self.seg_value[i]],[0.01,0.5])
            else:
                plt.plot([0.2,0.8],[self.seg_value[i],self.seg_value[i]])
        plt.show()
 
def main():
    bag = Bagging(11)
    bag.train()
    bag.myPrint()
    bag.myPlot()
 