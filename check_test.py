import os
import sys

ans = {}
files = [f for f in os.listdir(sys.argv[1]+'test_main') ]

with open( 'src/dstc5/scripts/config/dstc5_test_main.flist' ,'w') as w:
    for f in files:
        w.write(f+'\n')

