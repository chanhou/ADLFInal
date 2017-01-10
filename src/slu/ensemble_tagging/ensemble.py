import os
from collections import Counter
import operator

ans = {}

files = [f for f in os.listdir('./src/slu/ensemble_tagging')]
print files
for f in files:
    if f=='ensemble.py': continue
    count = 0
    with open('src/slu/ensemble_tagging/'+f,'r') as ff:
        for line in ff:
            line = line.strip()
            if count not in ans: ans[count] = {}
            for ind, jj in enumerate(line.split()):
                if ind not in ans[count]: ans[count][ind] = []            
                ans[count][ind].append(jj)
            count += 1

f = open('./src/slu/predict-sep/tagging.txt','w')
for i in range(len(ans)):
    ensem = []
    for j in range(len(ans[i])):
        item = dict(Counter(ans[i][j]))
        item = sorted(item.items(), key=operator.itemgetter(1))
        item.reverse()
        best_item = item[0][0]
        ensem.append(best_item)
        #print item, best_item
    f.write(' '.join(ensem)+'\n')
    #' '.join(ensem)+'\n')
f.close()
