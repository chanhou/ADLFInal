
num=
test=dev
pred=
role=GUIDE

dev: generate check score
test: generate check

num1=
num2=

generate:
	python generate_testResult_slu.py --trainset dstc5_train --testset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --testIntent predict$(pred)/intent.txt$(num)  --testSlot predict$(pred)/tagging.txt$(num) --predictF predict$(pred) --outfile result$(pred)/slu_$(test)$(num).json --roletype $(role)

check:
	python ../dstc5/scripts/check_slu.py --dataset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile result$(pred)/slu_$(test)$(num).json --roletype $(role)

score:
	python ../dstc5/scripts/score_slu.py --dataset dstc5_$(test) --dataroot ../../data_for_stu/$(test) --ontology ../dstc5/scripts/config/ontology_dstc5.json --jsonfile result$(pred)/slu_$(test)$(num).json --roletype $(role) --scorefile result$(pred)/slu.score$(num).csv
	cat ./result$(pred)/slu.score$(num).csv 
	python report_slu.py result$(pred)/slu.score$(num).csv
