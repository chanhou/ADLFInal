python check_test.py $1

python src/dstc5/scripts/baseline.py --dataset dstc5_test_main --dataroot $1 --trackfile $2 --ontology src/dstc5/scripts/config/ontology_dstc5.json --method 2
