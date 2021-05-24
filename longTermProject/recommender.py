import os
import sys
import time
import numpy as np
import pandas as pd

start = time.time()
path = os.getcwd()

command = sys.argv
try :
    if len(command) != 3:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용해주세요. ex. recommander.py u1.base u1.test")
    exit()

# read data

TrainDataFileName = command[1]
TestDataFileName = command[2]

TrainDataPath = os.path.join(path,TrainDataFileName)
TestDataPath = os.path.join(path,TestDataFileName)

train = pd.read_csv(TrainDataPath,sep='\t',header=None).to_numpy()
test = pd.read_csv(TestDataPath,sep='\t',header=None).to_numpy()

# convert raw data to rating matrix (post-use matrix) and make pre-use matrix
maximum = np.amax(train,axis=0)

postUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])
preUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])


for i in range(len(train)):
    postUseMatrix[train[i,0]-1,train[i,1]-1] = train[i,2]
    preUseMatrix[train[i,0]-1,train[i,1]-1] = 1

# make pre-use matrix ( 0 - 1 rating matrix and fill in the blank use WRMF method ) // WRMF 조사

# conversion pre-use matrix to post-use matrix by specific method // pre-use matrix를 어떻게 반영할 것인지에 대해서 조사

# predicting the data using SVD method // SVD를 어떻게 진행하는지 조사

# save the data to txt file format