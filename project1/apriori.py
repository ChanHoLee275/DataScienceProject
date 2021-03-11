import os
import sys

command = sys.argv

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

InputData = open(InputFileName,'r')
OutputData = open(OutputFileName,'w')

## Data 읽어오기
Data = list()
while 1:
    transaction = InputData.readline().split('\t')
    transaction[-1] = transaction[-1][:-1]
    if transaction == [''] :
        break
    transaction = list(map(int, transaction)) ## str2int in list
    Data.append(transaction)

## 결과 저장하기
for i in Data:
    for j in i:
        result = "{%d}\t{%d}\t%d\t%d\n" %(i[0],i[0],i[0],i[0])
        OutputData.write(result)

InputData.close()
OutputData.close()