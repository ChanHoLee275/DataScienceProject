import os
import sys
import time
import numpy as np
import pandas as pd
import WRMF
import copy
import GradientDescent

start = time.time()
path = os.getcwd()

command = sys.argv = ["a",'u4.base','u4.test']

try :
    if len(command) != 3:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용해주세요. ex. recommander.py u1.base u1.test")
    exit()

# function for RMSE

def RMSE(post,label):
    output = 0
    (row,column) = label.shape
    array = np.zeros(row*3).reshape(row,3)
    for i in range(row):
        predict = post[label[i,0]-1,label[i,1]-1]
        array[i,0] = label[i,0]
        array[i,1] = label[i,1]
        array[i,2] = predict
        answer = label[i,2]
        output += np.power((predict-answer),2)
    return np.sqrt(output/row), array


# read data

TrainDataFileName = command[1]
TestDataFileName = command[2]

TrainDataPath = os.path.join(path,TrainDataFileName)
TestDataPath = os.path.join(path,TestDataFileName)

train = pd.read_csv(TrainDataPath,sep='\t',header=None).to_numpy()
test = pd.read_csv(TestDataPath,sep='\t',header=None).to_numpy()

# convert raw data to rating matrix (post-use matrix) and make pre-use matrix
maximum = np.amax(train,axis=0)
start = time.time()
postUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])
preUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])


for i in range(len(train)):
    postUseMatrix[train[i,0]-1,train[i,1]-1] = train[i,2]
    preUseMatrix[train[i,0]-1,train[i,1]-1] = 1

# make pre-use matrix ( 0 - 1 rating matrix and fill in the blank use WRMF method ) // WRMF 조사 // 완료

model1 = WRMF.WRMF(preUseMatrix)
model1.train()

# pre-use matrix trancate
(row,column) = model1.model.shape

for i in range(row):
    index1 = model1.model[i,:] < 0
    index2 = model1.model[i,:] > 1
    model1.model[i,:][index1] = 0
    model1.model[i,:][index2] = 1
index = np.where(preUseMatrix == 1)
print(abs(model1.model[index] - preUseMatrix[index]).sum()/(row*column))

# conversion pre-use matrix to post-use matrix by specific method
index = np.where(preUseMatrix != 1)
vector = copy.deepcopy(model1.model[index])
vector = np.reshape(vector,row*column)
threshold1 = np.sort(vector)[::-1][int(0.3*len(vector))]
threshold2 = np.sort(vector)[::-1][int(0.5*len(vector))]

print(threshold1, threshold2)
# predicting the data using WRMF

for i in range(row):
    index1 = np.logical_and(model1.model[i,:] >= threshold1, model1.model[i,:] != 1)
    index2 = np.logical_and(model1.model[i,:] >= threshold2, model1.model[i,:] < threshold1)
    postUseMatrix[i,:][index1] = 2
    postUseMatrix[i,:][index2] = 1

model2 = GradientDescent.GradientDescent(postUseMatrix)

model2.train()

end = time.time()

(row,column) = model2.model.shape

for i in range(row):
    index1 = model2.model[i,:] < 0
    index2 = model2.model[i,:] > 5
    model2.model[i,:][index1] = 0
    model2.model[i,:][index2] = 5

print("running time : ",end-start)
output = np.array(model2.model,dtype=np.int64)

(performance,prediction) = RMSE(model2.model,test)
print("Test Result : ",performance)
# save the data to txt file format
OutputFile = os.path.join(path,command[1] + "prediction.txt")
prediction = pd.DataFrame(prediction)
prediction.to_csv(OutputFile,sep="\t",header=None)