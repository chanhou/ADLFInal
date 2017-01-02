
python rnn-nlu-intent/run_multi-task_rnn_result.py --data_dir rnn-nlu-intent/data/slu --train_dir rnn-nlu-intent/model_tmp --max_sequence_length 71 --task intent --bidirectional_rnn True --output ./predict-sep

python rnn-nlu-tagging/run_multi-task_rnn_result.py --data_dir rnn-nlu-tagging/data/slu --train_dir rnn-nlu-tagging/model_tmp --max_sequence_length 71 --task tagging --bidirectional_rnn True --output ./predict-sep

