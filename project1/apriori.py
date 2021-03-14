import os
import sys
import math
import numpy as np
import pandas as pd
import time

path = os.getcwd()

def frequent(data,itemset):
  # itemset is dictionary and data is 2d-array, key of dictionary is tuple
  for i in list(itemset.keys()):
    count = 0
    for j in data:
      if set(i).issubset(set(j)):
        count += 1
    itemset[i] = count
  return itemset
  
def generate_candidate(itemset,MinimumSupport,numberofdata):
    key = list(itemset.keys())
    new_key = dict()
    past_length = len(key[0])
    now_length = past_length + 1
    low_key = list()
    for i in key:
        if itemset[i]/numberofdata*100 < MinimumSupport:
            low_key.append(i)

    for i in range(len(key)):
        for j in range(len(key)):
            flag = 0
            candidate = set(key[i]).union(set(key[j]))
            candidate = list(candidate)
            candidate.sort()
            for k in low_key:
                if set(k) in candidate:
                    flag = 1
                    break
            if len(candidate) == now_length and flag == 0:
                new_key[tuple(candidate)] = 0

    return new_key

def powerset(itemset):
    element = list(itemset)
    count = int(math.pow(2,len(itemset)) - 1)
    answer = list()
    for i in range(1,count):
        subset = list()
        binary = bin(i)
        binary = binary[2:]
        binary = binary[::-1]
        for j in range(len(binary)):
            if binary[j] == '1':
                subset.append(element[j])
        answer.append(subset)
    return answer

    
command = sys.argv = "apriori.py 5 input.txt ouput.txt".split()

try :
    if len(command) != 4:
        raise Exception("명령어가 잘못 입력되었습니다.")
except Exception as e:
    print("명령어를 문법에 맞게 사용하여 주세요. ex. apriori.py 5 input.txt output.txt")
    exit()

MinimumSupport = int(command[1])
InputFileName = command[2]
OutputFileName = command[3]

InputData = open(path+'/'+InputFileName,'r')
OutputData = open(path+'/'+OutputFileName,'w')

## Data 읽어오기
RawData = list()
while 1:
    transaction = InputData.readline().replace("\n","").split('\t')
    if transaction == [''] :
        break
    transaction = list(map(int, transaction)) ## str2int in list
    RawData.append(transaction)

    ## remove ID
data = pd.DataFrame(RawData)
items = data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]
items = items.dropna(axis=0,how='all')
## make 1-item frequent
items = items.values.tolist()
itemset = dict()
numberofdata = len(items)
for i in items:
    for j in i:
        if not itemset.get((j,),0):
            itemset[(j,)] = 0

# Check frequent 1-itemset
itemset = frequent(items,itemset)
low_key = list()
for i in itemset.keys():
    if itemset[i]/numberofdata*100 < MinimumSupport:
        low_key.append(i)
for i in low_key:
    del itemset[i]

# Generate candidate itemsets of length (k+1) from frequent itemsets of length k ==> 기존의 데이터를 넣어주면, 이를 활용해서 새로운 데이터들을 hash table에 추가하는 형식으로
new_itemset = itemset
while True:
  # generate candidate
    new_itemset = generate_candidate(new_itemset,MinimumSupport,numberofdata)
  # check candidate
    new_itemset = frequent(items,new_itemset)
    low_key = list()
    for i in new_itemset.keys():
        if new_itemset[i]/numberofdata*100 < MinimumSupport:
            low_key.append(i)
    L = new_itemset
    for i in low_key:
        del L[i]

  # if result is NULL, stop the generate candidate
    if L == dict():
        break
    else :
        itemset.update(L)

# calculate support and confidence

for i in itemset.keys():
    element = powerset(i)
    for j in element:
        if len(i) == 1:
            before = i
            support = itemset[i]/numberofdata*100
            support = round(support,2)
            after = ''
            confidence = ''
            string = "{"+','.join(str(int(k)) for k in list(before))+"}\t"+"{"+after+"}\t"+str(support)+"\t"+str(confidence)+"\n"
        else : 
            before = set(i).difference(j)
            before = list(before)
            before.sort()
            after = j
            support = itemset[i]/numberofdata*100
            confidence = itemset[i]/itemset[tuple(before)]*100
            support = round(support,2)
            confidence = round(confidence,2)
            string = "{"+','.join(str(int(k)) for k in list(before))+"}\t"+"{"+','.join(str(int(l)) for l in list(after))+"}\t"+str(support)+"\t"+str(confidence)+"\n"
            OutputData.write(string)

OutputData.close()
InputData.close()
        
        

