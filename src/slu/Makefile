
num=
test=dev

dev: generate check score
test: generate check

predict:
	python rnn-nlu/run_multi-task_rnn_result.py --data_dir rnn-nlu/data/slu --train_dir rnn-nlu/model_tmp --max_sequence_length 75 --task joint --bidirectional_rnn True --output ./predict

generate:
	python generate_testResult_slu.py --trainset dstc5_train --testset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --testIntent predict/intent.txt$(num)  --testSlot predict/tagging.txt$(num) --modelfile ./aa --outfile result/slu_$(test)$(num).json --roletype GUIDE


check:
	python ../dstc5/scripts/check_slu.py --dataset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile result/slu_$(test)$(num).json --roletype GUIDE

score:
	python ../dstc5/scripts/score_slu.py --dataset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile result/slu_$(test)$(num).json --roletype GUIDE --scorefile result/slu.score$(num).csv
	cat ./result/slu.score$(num).csv 
