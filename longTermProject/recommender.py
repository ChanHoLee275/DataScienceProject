import os
import sys
import time
import numpy as np
import pandas as pd
import GradientDescent

start = time.time()
path = os.getcwd()

command = sys.argv

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
        array[i,2] = np.around(predict)
        if array[i,2] > 5:
            array[i,2] = 5
        elif array[i,2] < 1:
            array[i,2] = 1
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
maximum1 = np.amax(train,axis=0)
maximum2 = np.amax(test,axis=0)
maximum = list()
for i in range(2):
    if maximum1[i] > maximum2[i]:
        maximum.append(maximum1[i])
    else :
        maximum.append(maximum2[i])
start = time.time()
postUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])
preUseMatrix = np.zeros(maximum[0]*maximum[1]).reshape(maximum[0],maximum[1])


for i in range(len(train)):
    postUseMatrix[train[i,0]-1,train[i,1]-1] = train[i,2]
    '''preUseMatrix[train[i,0]-1,train[i,1]-1] = 1'''

# make pre-use matrix ( 0 - 1 rating matrix and fill in the blank use WRMF method ) // WRMF 조사 // 완료
'''
model1 = GradientDescent.GradientDescent(preUseMatrix)
model1.train()

# pre-use matrix trancate
(row,column) = model1.model.shape

for i in range(row):
    index1 = model1.model[i,:] < 0
    index2 = model1.model[i,:] > 1
    model1.model[i,:][index1] = 0
    model1.model[i,:][index2] = 1
'''
'''
# conversion pre-use matrix to post-use matrix by specific method
threshold1 = 0.7
threshold2 = 0.8
'''
'''
for i in range(row):
    for j in range(column):
        if preUseMatrix[i,j] == 0:
            if model1.model[i,j] > threshold2:
                postUseMatrix[i,j] = 2
            elif model1.model[i,j] > threshold1:
                postUseMatrix[i,j] = 1
'''
model2 = GradientDescent.GradientDescent(postUseMatrix,crit=0.001,factor=1)

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
OutputFile = os.path.join(path,command[1] + "_prediction.txt")
prediction = np.array(prediction,dtype=np.int64)
prediction = pd.DataFrame(prediction,dtype=np.int64)
prediction.to_csv(OutputFile,sep="\t",header=None,index=False)