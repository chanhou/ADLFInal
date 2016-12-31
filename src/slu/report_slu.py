import sys

score = 0
with open(sys.argv[1],'r')as f:
    for line in f:
        line = line.split(',')
        if line[2]==' all' and line[3]==' f1':
            print 'f1\t',line[5][1:-1]
            if line[1]==' speech_act':
                score += 2.*float(line[5])
            else:
                score += float(line[5])
print 'final\t', score/3.
