#!/bin/bash

#==== download model ====
wget https://www.dropbox.com/s/cnhv815o1y1wl9n/best_model.ckpt?dl=0 -O ./src/slg/best_model.ckpt

#==== testing ===========
python2 ./src/slg/translate.py --decode 1\
                               --dataroot $1\
                               --output_file $2