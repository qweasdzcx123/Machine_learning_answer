import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
class Cluster:
    def loadData(self):            # 读入数据
        dataset = pd.read_excel('./WaterMelon_4.0.xlsx',encoding = 'gbk')  # 读取数据
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
 
    # 执行聚类过程
    def cluster(self,k,U):      # U是初始均值向量，k为聚类数
        C = {}
        d = np.ones((k,))
        for i in range(self.m):
            for j in range(k):
                d[j] = self.distance(U[j,:],self.x[i,:])    # 计算每一个均值向量到 xi 的距离
            lamda = np.argmin(d)  ##找到距离λ最小的均值向量
            if (lamda+1) in C:
                C[lamda+1] = np.vstack((C[lamda+1],self.x[i,:]))##　将此xi划入相应族
            else:
                C[lamda+1] = self.x[i,:]
        return C             #返回族集合
 
    # 计算距离函数  即两个向量的二范数
    def distance(self,p,q):
        p = np.array(p)
        q = np.array(q)
        return np.linalg.norm(q-p)
 
    # 更新均值向量　W是族集合　U为初始选定的均值向量
    def getNewMean(self,W,U):
        flag = 0
        keys = W.keys()#每一类族
        nums = 0
        for key in keys:
            X = W[key]
            u = np.mean(X,0)
            if u == U:
                nums+=1
                continue
            else:
                
                flag = 1
                U[nums]=u
                nums+=1
                
        return flag         # flag为1说明进行了更新
 
    # 迭代执行整个聚类更新过程
    def train(self,k,U):
        flag = 1
        count = 1
        while (flag == 1 & count <= 15):
            C = self.cluster(k,U)
            print(np.shape(C[1]))
            flag = self.getNewMean(C,U)
            count = count + 1
        print(count)
        return C,U
 
    # 绘制分类结果图
    def myplot(self,C,U):
        keys = C.keys()
        for key in keys:
            x1 = C[key][:,0]
            x2 = C[key][:,1]
            #print(C[key][1])
            #hull = cv2.convexHull(C[key])
            plt.plot(x1,x2,'.')
            plt.plot(x1,x2,)
            #plt.plot(hull[:,0,0],hull[:,0,1],'g-.')
        U = np.array(U)
        print(U[0,0])
        plt.plot(U[:,:,0],U[:,:,1],'+')
        plt.xlabel('密度')
        plt.ylabel('含糖率')
        plt.show()
 
    # 聚类
    def myTest(self,k,index):   # k为聚类的数量，index为初始的均值向量的索引
        self.loadData()
        #space = np.floor(self.m/k)
        #index = [int(np.random.uniform(i*space,i*space+space)) for i in range(k)]
        U = list()
        for i in index:
            U.append(self.x[i,:])
        C,U = self.train(k,np.array(U))
        self.myplot(C,U)
 
def main():
    clust = Cluster()
    clust.myTest(2,[5,15])         # 两类
    clust.myTest(3,[5,15,25])      # 三类
    clust.myTest(4,[5,15,20,25])   # 四类
 
if __name__ == '__main__':
    main()