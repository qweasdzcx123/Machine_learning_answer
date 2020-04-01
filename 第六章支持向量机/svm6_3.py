#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 09:13:47 2020

@author: rui
"""

import numpy as np
from libsvm.svmutil import *
 
def readData():    # 读取数据
    """
    Read data from txt file.
    Return:
        X1, y1, X2, y2, X3, y3: X is list with shape [50, 4],
                                y is list with shape [50,]
    """
    X1 = []
    y1 = []
    X2 = []
    y2 = []
    #read data from txt file
    with open("./iris.data", "r") as f:
        for line in f:
            x = []
            iris = line.strip().split(",")
            for attr in iris[0:4]:
                aa = float(attr)
                x.append(aa)
            if iris[4]=="Iris-setosa":
                X1.append(x)
                y1.append(1)
            else:
                X2.append(x)
                y2.append(-1)
    return X1, y1, X2, y2
 
def getX(X):              # 生成LIBSVM需要的固定数据格式
    A = []
    X = np.array(X)
    m,n = np.shape(X)
    for i in range(m):
        dic = {}
        for j in range(n):
            dic[j+1] = X[i,j]
        A.append(dic)
    return A
 
def svm_linear(x,y):       # 线性核训练
    lina_options = '-t 0 -c 1 -b 1'      #线性核
    model = svm_train(y,x,lina_options)
    svm_save_model('./problem_6.3/UCI_linear',model)
 
def svm_guass(x,y):      # 高斯核训练
    guass_options = '-t 2 -c 4 -b 1'     # 高斯核
    model = svm_train(y,x,guass_options)
    svm_save_model('./problem_6.3/UCI_guass',model)
 
def predic_linear(x,y):     # 线性核预测
    model = svm_load_model('./problem_6.3/UCI_linear')
    p_label,p_acc,p_val = svm_predict(y,x,model)
    return p_label,p_acc
 
def predic_guass(x,y):      # 高斯核预测
    model = svm_load_model('./problem_6.3/UCI_guass')
    p_label,p_acc,p_val = svm_predict(y,x,model)
    return p_label,p_acc
 
def main():
    X1,y1,X2,y2 = readData()
 
    x1 = getX(X1)
    x2 = getX(X2)
 
    x_train = x1[:40]
    y_train = y1[:40]
    x_train.extend(x2[:40])
    y_train.extend(y2[:40])
 
    x_test = x1[40:]
    x_test.extend(x2[40:])
    y_test = y1[40:]
    y_test.extend(y2[40:])
 
    svm_linear(x_train,y_train)
    svm_guass(x_train,y_train)
    p_label_lin,p_acc_lin = predic_linear(x_test,y_test)
    p_label_gua,p_acc_gua = predic_guass(x_test,y_test)
    print("线性预测结果")
    print(p_label_lin,p_acc_lin)
    print("高斯预测结果")
    print(p_label_gua,p_acc_gua)
 
