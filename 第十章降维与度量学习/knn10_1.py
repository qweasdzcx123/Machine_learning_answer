# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import warnings
import heapq
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['AR PL UKai CN'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
warnings.filterwarnings("ignore")
class KNN:
    #读取数据
    def loadData(self):            
        dataset = pd.read_excel('./WaterMelon_3.0.xlsx',encoding = 'gbk')
        Attributes = dataset.columns
        m,n = np.shape(dataset)
        dataset = np.matrix(dataset)
        for i in range(m):
            if dataset[i,n-1] == '是': dataset[i,n-1] = 1
            else: dataset[i,n-1] = -1
        self.future = Attributes[1:n-1]
        self.x = dataset[:,1:n-1]
        self.y = dataset[:,n-1]
        self.m = m
 
    def getDistance(self):        # 计算出每一个样本之间的距离，用二维矩阵存贮
        self.distance = np.zeros((self.m,self.m))      #m*m矩阵
        for i in range(self.m):
            self.distance[i,i] = np.inf #对角线元素为无穷大
            for j in range(i+1,self.m):
                d = self.Edist(self.x[i,:],self.x[j,:])
                self.distance[i,j] = d
                self.distance[j,i] = d
 
    def Edist(self,x1,x2):    # 欧式距离
        x1 = np.array(x1)
        x2 = np.array(x2)
        #print(x1)
        return np.linalg.norm(x1-x2)   #而范数即向量欧式距离
 
    def train(self,k):        # 进行knn训练
        label = np.ones((self.m,))
        for i in range(self.m):
            index = self.getindex(self.distance[i,:],k)    # 得到最小k个样本的索引
            label[i] = self.getLabel(index)  #根据这k个样本标记的和作为返回值 # 根据最近的样本得到该样本的预测标签
        return label
 
    def getindex(self,dist,k):          # 找最小值对应的索引
        dist = dist.tolist()
        index = map(dist.index,heapq.nsmallest(k,dist))#从列表中找到值最小的k个索引值
        return list(index)
 
    def getLabel(self,index):           # 根据索引对应的样本标签进行分类
        labellist = self.y[index]
        #print(labellist)
        #print(np.sum(labellist))
        if np.sum(labellist)>0: return 1
        else: return -1
 
    # 绘图
    def myPlot(self,label):
        #print(type(self.y[1]))
        Y = np.array(self.y).tolist()
        Y = np.reshape(Y,(self.m,)).tolist()
        #print(Y)
        #print(type(Y))
        #print(np.shape(Y))
        Tgoodin = [i for i,x in enumerate(Y) if x==1]
        #print(Tgoodin)
        Tbadin = [i for i,x in enumerate(Y) if x==-1]
        #print(Tbadin)
        Tgood = self.x[Tgoodin,:]
        Tbad = self.x[Tbadin,:]
        #print(Tgood)
 
        label = label.tolist()
        Pgoodin = [i for i,x in enumerate(label) if x==1]
        Pbadin = [i for i,x in enumerate(label) if x==-1]
        Pgood = self.x[Pgoodin,:]
        Pbad = self.x[Pbadin,:]
 
        plt.figure()
        l1, = plt.plot(Tgood[:,0],Tgood[:,1],'r+')
        l2, = plt.plot(Tbad[:,0],Tbad[:,1],'r_')
        l3, = plt.plot(Pgood[:,0],Pgood[:,1],'bx')
        l4, = plt.plot(Pbad[:,0],Pbad[:,1],'gx')
        plt.legend(handles=[l1,l2,l3,l4],labels=['真好瓜','真坏瓜','预测为好瓜','预测为坏瓜'],loc='best')
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.show()
 
def main():
    knn = KNN()
    knn.loadData()
    knn.getDistance()
    label = knn.train(3)
    knn.myPlot(label)