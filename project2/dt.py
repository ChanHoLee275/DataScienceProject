import numpy as np
import pandas as pd
import os

class Node:
    def __init__(self,value):
        self.child = list()
        self.data = value
        self.category = None
        self.flag = None

def entropy(data,target):
    answer = 0
    data = data.loc[:,target]
    elements, count = np.unique(data,return_counts=True)
    for i in range(len(elements)):
        answer += count[i]/np.sum(count)*np.log2(count[i]/np.sum(count)+np.finfo(float).eps)
    return -answer

def InformationGain(data,target1,target2):
    # target1 is label
    # target2 is attribute

    answer = 0
    total_entropy = entropy(data,target1)
    column = data[target2]
    weight,count = np.unique(column,return_counts=True)
    
    for i in range(len(weight)):
        group = data.where(data[target2] == weight[i]).dropna()
        answer += count[i]/np.sum(count)*entropy(group, target2)
    
    answer -= total_entropy
    return -answer
    
def GainRatio(data,target1,target2):
    # target1 is label
    # target2 is attribute
    answer = 0

    gain = InformationGain(data, target1, target2)

    column = data[target2]

    split,count = np.unique(column,return_counts=True)

    for i in range(len(split)):
        answer += count[i]/np.sum(count)*np.log2(count[i]/np.sum(count)+np.finfo(float).eps)
    
    return gain/(-answer)

def gini(data,target):
    answer = 0
    column = data[target]
    elements, count = np.unique(column,return_counts=True)

    for i in range(len(elements)):
        group = data.where(data[target] == elements[i]).dropna()
        group = group[target]
        subelements, subcount = np.unique(group,return_counts=True)
        gini = 1
        for j in range(len(subelements)):
            gini -= np.power(subcount[j]/np.sum(subcount),2)
   
        answer += count[i]/np.sum(count)*gini
    
    return answer

def tree(data,category,parent):
    # 종료 조건
    
    if len(data) == 0:
        return
    
    elif len(data) == np.sum(data[category[-1]]):
        return
    
    else :
        # find feature
        max_measure = 0
        divide = category[-1]
        for i in range(len(category)-1):
            measure = InformationGain(data, category[-1], category[i])
            if measure > max_measure:
                divide = category[i]
                max_measure = measure

        # grow tree
        elements,count = np.unique(data,return_counts=True)

        for i in range(len(elements)):
            group = data.where(data[divide] == elements[i]).dropna()
            child = Node(group)
            parent.child.append(child)
            child.flag = elements[i]
            tree(group,category,child)
    
        parent.category = divide

def fit(data,category,model):
    while(model.child != list()):
        divide = model.category
        for i in model.child:
            if i.flag == data[divide]:
                model = model.child
    column = model.data[-1]
    elements, count = np.unique(column,return_counts=True)
    maximum = 0
    label = None
    for i in range(len(elements)):
        if maximum < count[i]:
            maximum = count[i]
            label = elements[i]
    answer = data
    answer[0][-1] = label
    return answer
       
path = os.getcwd()

command = sys.argv

try :
    if len(command) != 4:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용하여 주세요. ex. dt.py dt_train.txt dt_test.txt dt_result.txt")
    exit()

# set file stream

train = command[1]
test = command[2]
save = command[3]

TrainData = open(os.path.join(path,train),'r')
TestData = open(os.path.join(path,test),'r')
SaveResult = open(os.path.join(path,result),'w')

category = TrainData.readline().replace("\n","").split("\t")

train = pd.read_csv(TrainData,sep="\t")
test = pd.read_csv(TestData,sep="\t")

root = Node(train)
tree(data,category,root)

(row,column) = test.shape
output = np.empty(1,column)
for i in range(row):
    output = np.append(output,fit(test.iloc[i,:],category,root),axis=0)


