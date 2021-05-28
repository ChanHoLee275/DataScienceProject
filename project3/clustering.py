import numpy as np
import pandas as pd
import os
import sys
import copy
import matplotlib.pyplot as plt

np.random.seed(0)

class DBSCAN:

    def __init__(self,data,n,eps,minPts):
        self.data = data
        self.NumberOfCluster = n
        self.eps = eps
        self.minPts = minPts
        self.cluster = np.zeros(self.data.shape[0]).reshape(self.data.shape[0])
        self.checked = np.zeros(self.data.shape[0]).reshape(self.data.shape[0])

    def countdistance(self,point):
        # 모든 점과 밀도를 계산하는 한 점 사이의 거리를 계산하고 점을 기준으로 반경 안에 얼마나 점이 있는지 계산하는 메소드
        minus = self.data[:,1:3] - point
        distance = (minus[:,0]**2 + minus[:,1]**2)**0.5
        count = np.less_equal(distance,self.eps)
        return np.sum(count), count

    def unionCluster(self,clusterNumber):
        _index = self.cluster == clusterNumber
        candidate = self.data[_index]
        for i in range(candidate.shape[0]):
            (count,index) = self.countdistance(candidate[i,1:3])
            cluster = self.cluster[index]
            if count >= self.minPts:
                for j in range(count):
                    if cluster[j] != clusterNumber:
                        self.cluster[_index] = cluster[j]
                        return 1
            else :
                continue
        return 0

    def plot(self):
        unique_labels = set(model.cluster)
        colors = [plt.cm.gist_rainbow(each) for each in np.linspace(0, 1, len(unique_labels))]
        plt.figure(figsize=[8, 8])
        model.data = pd.DataFrame(model.data)
        for cluster_index, col in zip(unique_labels, colors):
            if cluster_index == -1:
                col = [0, 0, 0, 1]
            class_mask = (model.cluster == cluster_index)
            plt.plot(model.data.values[class_mask][:, 1], 
                    model.data.values[class_mask][:, 2], 
                    'o', markerfacecolor=tuple(col), markeredgecolor=tuple(col), 
                    markersize=1)

    def expanding(self,candidate,clusterNumber):
        # 특정 점들을 받아드려서, cluster인지 아닌지 확인하는 메소드
        row = candidate.shape[0]
        for i in range(row):
            if self.checked[int(candidate[i,0])] == 1:
                continue
            (count,index) = self.countdistance(candidate[i,1:3])
            self.checked[int(candidate[i,0])] = 1
            if count >= self.minPts:
                self.cluster[index] = clusterNumber
    
    def checking(self,clusterNumber):
        # 특정 클러스터의 모든 점을 조사하여 더 이상 확장이 불가능하다고 판단하면, 빈 배열을 반환, 확장 가능하면 고려해야 할 점들을 반환
        cluster = (self.cluster == clusterNumber)
        checked = self.checked[cluster]
        index = np.logical_and(cluster,(np.logical_not(self.checked)))
        return self.data[index]
    
    def training(self):
        # expanding 메소드와 checking 메소드를 활용하여 clustering 하는 메소드
        clusterNumber = 1
        while True:

            i = np.random.randint(0,self.data.shape[0])
            
            while self.checked[i] != 0:
                i = np.random.randint(0,self.data.shape[0])

            self.checked[i] = 1
            
            (count,index) = self.countdistance(self.data[i,1:3])

            if count >= self.minPts:

                if np.sum(self.cluster[index]) == 0:

                    self.checked[int(self.data[i,0])] = 1
                    self.cluster[index] = clusterNumber
                    candidate = self.checking(clusterNumber)
                    
                    while candidate.size != 0:
                        self.expanding(candidate,clusterNumber)
                        candidate = self.checking(clusterNumber)

                    if clusterNumber != 1 and self.unionCluster(clusterNumber):
                        clusterNumber -= 1
                    
                    else :
                        print("cluster",clusterNumber,"is done!")
                    clusterNumber += 1
            
            if np.sum(self.checked) == self.data.shape[0]:
                counts = list()
                for j in range(1,clusterNumber+1):
                    counts.append(np.sum(self.cluster == j))
                        
                while len(counts) != self.NumberOfCluster:
                    minimum = min(counts)
                    idx = counts.index(minimum)
                    self.cluster[self.cluster == idx + 1] = 0
                    counts.pop(idx)
                    self.data = self.data[self.cluster != 0]
                    self.cluster = self.cluster[self.cluster != 0]
                break

        return self.cluster
                


dir = os.getcwd()

command = sys.argv

try :
    if len(command) != 5:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용하여 주세요. ex. clustering.py input1.txt 8 15 22")
    exit()

Inputfile = os.path.join(dir,command[1])
n = int(command[2])
eps = int(command[3])
minPts = int(command[4])
Outputfile = command[1][:-4]+"_cluster_"

# read Input file
Input = pd.read_csv(Inputfile,sep="\t",header=None).to_numpy()

model = DBSCAN(Input,n,eps,minPts)
model.training()

# write result
for i in range(0,n):
    fileName = Outputfile+str(i)+".txt"
    clusterData = model.data[model.cluster == i+1]
    clusterData = np.array(clusterData[:,0],dtype=np.int64)
    clusterData = pd.DataFrame(clusterData)
    path = os.path.join(dir,fileName)
    clusterData.to_csv(path,index=False,header=None)