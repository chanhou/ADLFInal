
import sys

count = 0
with open(sys.argv[1],'r')as f:
    for line in f:
        ll = len(line.split(' '))
        if ll > count: count=ll
print count
