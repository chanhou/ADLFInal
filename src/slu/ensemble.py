
import os

a = 17
#a = 2
for i in range(a):
    os.system('python ./src/slu/rnn-nlu-intent/run_multi-task_rnn_result.py --data_dir ./src/slu/rnn-nlu-intent/data/slu --train_dir ./src/slu/rnn-nlu-intent/model_tmp --max_sequence_length 72 --task intent --bidirectional_rnn True --output ./src/slu/ensemble --size 282 --word_embedding_size 182 --num_layers 1 --model '+str((i+4)*300))
