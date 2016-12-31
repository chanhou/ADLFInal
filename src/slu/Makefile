
dev: generate check score

predict:
	python rnn-nlu/run_multi-task_rnn_result.py --data_dir rnn-nlu/data/slu --train_dir rnn-nlu/model_tmp --max_sequence_length 75 --task joint --bidirectional_rnn True --output ./predict

generate:
	python generate_testResult_slu.py --trainset dstc5_train --testset dstc5_dev --dataroot ../../data_for_stu/dev --testIntent predict/intent.txt.9  --testSlot predict/tagging.txt.9 --modelfile ./aa --outfile slu_dev.json --roletype GUIDE

check:
	python ../dstc5/scripts/check_slu.py --dataset dstc5_dev --dataroot ../../data_for_stu/dev --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile slu_dev.json --roletype GUIDE

score:
	python ../dstc5/scripts/score_slu.py --dataset dstc5_dev --dataroot ../../data_for_stu/dev --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile slu_dev.json --roletype GUIDE --scorefile ./slu.score.csv
