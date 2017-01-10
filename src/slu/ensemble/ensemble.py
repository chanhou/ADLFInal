import os
from collections import Counter
import operator

ans = {}

files = [f for f in os.listdir('src/slu/ensemble/')]
for f in files:
    if f=='ensemble.py': continue
    count = 0
    with open('src/slu/ensemble/'+ f,'r') as ff:
        for line in ff:
            line = line.strip()
            if count not in ans: ans[count] = []
            ans[count].append(line)
            count += 1

f = open('./src/slu/predict-sep/intent.txt','w')
for i in range(len(ans)):
    item = dict(Counter(ans[i]))
    item = sorted(item.items(), key=operator.itemgetter(1))
    item.reverse()
    best_item = item[0][0]
    '''    
    if len(item)==1:
        best_item = item[0][0]
    else:
        best_item = []
        for key in item:
            if key[1] > len(item)*1./2:
                print key[0]
                best_item.append(key[0])
        if len(best_item)==0:
             best_item = item[0][0]
        best_item = '|'.join(best_item)
    '''
    #print item, best_item
    f.write(best_item+'\n')
f.close()
