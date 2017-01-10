
# download model
wget https://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/US%20English/en-70k-0.2-pruned.lm.gz/download -O ./src/slu/en-70k-0.2-pruned.lm.gz 
wget https://mslab.csie.ntu.edu.tw/~chanhou/model_intent.zip -O ./src/slu/rnn-nlu-intent/model_intent.zip
unzip ./src/slu/rnn-nlu-intent/model_intent.zip

wget https://mslab.csie.ntu.edu.tw/~chanhou/model_tagging.zip -O ./src/slu/rnn-nlu-tagging/model_tagging.zip
unzip ./src/slu/rnn-nlu-tagging/model_tagging.zip

# generate dstc5_test_slu file list
python src/slu/check_test.py

# preprocess data
python ./src/slu/preprocess_slu_intent.py $1
python ./src/slu/preprocess_slu_tagging.py $1

# predict and ensemble
python ./src/slu/ensemble.py
python ./src/slu/ensemble_tagging.py
python ./src/slu/ensemble/ensemble.py
python ./src/slu/ensemble_tagging/ensemble.py

# generate
python ./src/slu/generate_testResult_slu.py --trainset dstc5_train --testset dstc5_test_slu --dataroot $1 --testIntent ./src/slu/predict-sep/intent.txt  --testSlot ./src/slu/predict-sep/tagging.txt --predictF predict-sep --outfile $2 --roletype GUIDE
