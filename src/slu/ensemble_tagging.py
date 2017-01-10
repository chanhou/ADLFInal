
import os

a = 29
for i in range(a):
    os.system('python ./src/slurnn-nlu-tagging/run_multi-task_rnn_result.py --data_dir ./src/slu/rnn-nlu-tagging/data/slu --train_dir ./src/slu/rnn-nlu-tagging/model_tmp --max_sequence_length 50 --task tagging --bidirectional_rnn True --output ./src/slu/ensemble_tagging --size 182 --word_embedding_size 182 --num_layers 1 --model '+str((i+19)*300))
