from typing import ContextManager
import numpy as np
import pandas as pd
import os
import sys
import copy

def countdistance(point,array2d,minimum):
    minus = array2d - point
    distance = (minus[:,0]**2 + minus[:,1]**2)**0.5
    count = np.less_equal(distance,minimum)
    return np.sum(count), count

def training(candidate, array2d,cluster,minimum,minPts,clusterNumber):
    row = candidate.shape[0]
    for i in range(row):
        (count,index) = countdistance(candidate[i,:],array2d,minimum)
        if np.sum(cluster[index] == 0 ) and count >= minPts:
            cluster[index] = clusterNumber
            cluster = training(array2d[index], array2d,cluster,minimum,minPts,clusterNumber)
    return cluster

dir = os.getcwd()

command = sys.argv = ["clustering.py" ,"input1.txt", "8", "15", "22"]

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
Outputfile = command[1]+"_cluster_"

# read Input file
Input = pd.read_csv(Inputfile,sep="\t",header=None).to_numpy()
cluster = np.zeros(Input.shape[0]).reshape(Input.shape[0],1)

# clustering start
i = np.random.randint(0,Input.shape[0])
before_cluster = copy.deepcopy(cluster)
clusterNumber = 1

while True:
    (count,index) = countdistance(Input[i,:],Input,eps)
    if count >= minPts:
        if np.sum(cluster[index] == 0) == count:
            cluster[index] = clusterNumber
            cluster = training(Input[index],Input,cluster,eps,minPts,clusterNumber)
            clusterNumber += 1
    else :
        i = np.random.randint(0,Input.shape[0])
    if np.sum(before_cluster == cluster) == Input.shape[0]:
        break
    else :
        before_cluster = copy.deepcopy(cluster)

Input[3] = cluster

for i in range(n):
    Outputfile += str(i) + ".txt"
    index = Input[:,3] == i
    Output = Input[index,0]
    Output[1] = Input[index,3]
    Output.to_csv(Outputfile,sep="\t")

"""
for i in range(Input.shape[0]):
    (count,index) = countdistance(Input[i,:],Input,eps)
    if count >= minPts:
        if np.sum(cluster[index] == 0) == count:
            cluster[index] = clusterNumber
            clusterNumber += 1
        else :
            clustered = cluster[cluster[index] != 0]
            cluster[index] = clustered[0]
    else :
        continue
    if np.sum(cluster != 0) == Input.shape[0]:
        break
"""