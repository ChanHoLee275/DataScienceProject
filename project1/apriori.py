import os
import sys
import math
import time

def frequent(data,itemset):
  # itemset is dictionary and data is 2d-array, key of dictionary is tuple
  # count the number of element in data
  for i in data:
    for j in itemset.keys():
      if set(j).issubset(set(i)):
        itemset[j] += 1 
  return itemset
  
def generate_candidate(itemset):
    # using apriori theroem
    key = list(itemset.keys())
    new_key = dict()
    past_length = len(key[0])
    now_length = past_length + 1
    # self joining
    for i in range(len(key)):
        for j in range(len(key)):
            flag = 0
            if i == j:
                break
            candidate = set(key[i]).union(set(key[j]))
            candidate = list(candidate)
            candidate.sort()
            candidate_element = list()
            # pruncing
            for k in range(len(candidate)):
                temp = candidate.copy()
                del temp[k]
                temp = tuple(temp)
                candidate_element.append(temp)

            if not set(candidate_element).issubset(set(key)):

                flag = 1

            if len(candidate) == now_length and flag == 0:
                new_key[tuple(candidate)] = 0
    return new_key

def subset(itemset):
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

start = time.time()

path = os.getcwd()

command = sys.argv = ['','5','input.txt','output.txt']
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

itemset = dict()

NumberOfData = len(RawData)

for i in RawData:
    for j in i:
        if not itemset.get((j,),0):
            itemset[(j,)] = 0

# Check frequent 1-itemset
itemset = frequent(RawData,itemset)

low_key = list()

for i in itemset.keys():
    if itemset[i]/NumberOfData*100 < MinimumSupport:
        low_key.append(i)
for i in low_key:
    del itemset[i]

# Generate candidate itemsets of length (k+1) from frequent itemsets of length k ==> 기존의 데이터를 넣어주면, 이를 활용해서 새로운 데이터들을 hash table에 추가하는 형식으로
new_itemset = itemset

while True:
  # generate candidate
    new_itemset = generate_candidate(new_itemset)

  # check candidate
    new_itemset = frequent(RawData,new_itemset)

  # remove candidate  

    low_key = list()

    for i in new_itemset.keys():
        if new_itemset[i]/NumberOfData*100 < MinimumSupport:
            low_key.append(i)

    for i in low_key:
        del new_itemset[i]

  # if result is NULL, stop the generate candidate

    if new_itemset == dict():
        break
    else :
        itemset.update(new_itemset)

# calculate support and confidence

for i in itemset.keys():
    
    element = subset(i)

    for j in element:

        if len(i) == 1:
            break

        else : 
            before = list(set(i).difference(j))
            before.sort()
            after = j
            support = itemset[i]/NumberOfData*100
            confidence = itemset[i]/itemset[tuple(before)]*100
            support = round(support,2)
            confidence = round(confidence,2)
            string = "{"+','.join(str(int(k)) for k in list(before))+"}\t"+"{"+','.join(str(int(l)) for l in list(after))+"}\t"+'{0:.2f}'.format(support)+"\t"+'{0:.2f}'.format(confidence)+"\n"
            OutputData.write(string)

OutputData.close()
InputData.close()
print(time.time() - start,'s')