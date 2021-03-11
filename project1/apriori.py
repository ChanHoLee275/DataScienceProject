import sys
import os
import numpy as np
import pandas as pd

command = sys.argv = "apriori.py 5 input.txt ouput.txt".split()

currentDictionary = os.getcwd()

try :
    if len(command) != 4:
        raise Exception("명령어가 잘못 입력되었습니다.")
    print("명령어가 올바르게 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용하여 주세요. ex. apriori.py 5 input.txt output.txt")
    exit()

MinimumSupport = command[1]
InputFileName = command[2]
OutputFileName = command[3]

InputData = open(currentDictionary + '/project1/' + InputFileName,'r')
OutputData = open(currentDictionary + '/project1/' + OutputFileName,'w')

## Data 읽어오기
RawData = list()
while 1:
    transaction = InputData.readline().split('\t')
    transaction[-1] = transaction[-1][:-1]
    if transaction == [''] :
        break
    transaction = list(map(int, transaction)) ## str2int in list
    RawData.append(transaction)

## remove ID
data = pd.DataFrame(RawData)
items = data.loc[:,[1,2,3,4,5,6,7,8,9,10,11]]
print(items.head())

## Initially, scan DB once to get frequent 1-itemset ==> hash table을 사용해서, key를 item으로 설정하면 좋을 듯

## Generate candidate itemsets of length (k+1) from frequent itemsets of length k ==> 기존의 데이터를 넣어주면, 이를 활용해서 새로운 데이터들을 hash table에 추가하는 형식으로

## Test the candidates against DB ==> hash table의 key를 기준으로 확인!

## Terminate when no frequent or candidate set can be generated ==> 종료조건

## 결과 저장하기
for i in RawData:
    for j in i:
        result = "{%d}\t{%d}\t%d\t%d\n" %(i[0],i[0],i[0],i[0])
        OutputData.write(result)

InputData.close()
OutputData.close()