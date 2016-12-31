!/bin/bash

python rnn-nlu/run_multi-task_rnn_result.py --data_dir rnn-nlu/data/slu --train_dir rnn-nlu/model_tmp --max_sequence_length 75 --task joint --bidirectional_rnn True --output ./predict

