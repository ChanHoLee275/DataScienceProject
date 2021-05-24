from typing import ContextManager
import numpy as np
import pandas as pd
import os
import sys

def countdistance(point,array2d,minimum):
    minus = array2d - point
    distance = (minus[:,0]**2 + minus[:,1]**2)**0.5
    count = np.less_equal(distance,minimum)
    return np.sum(count), count

dir = os.getcwd()

command = sys.argv

try :
    if len(command) != 5:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용하여 주세요. ex. apriori.py 5 input.txt output.txt")
    exit()

Inputfile = os.path.join(dir,command[1])
n = int(command[2])
eps = int(command[3])
minPts = int(command[4])

# read Input file
Input = pd.read_csv(Inputfile,sep="\t",header=None)
cluster = pd.DataFrame(np.zeros(Input.shape[0]).reshape(Input.shape[0],1))

# clustering start
# 다시 한번 생각해보기, 이렇게 진행하면 잘 안될듯

clusterNumber = 1

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
