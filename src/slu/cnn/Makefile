
dest=

train:
	python train.py --embedding_dim 256 --filter_sizes '3,4,5' --num_filters 128 --l2_reg_lambda 0.001 --dropout_keep_prob 0.5 --batch_size 64 --num_epochs 2000 --evaluate_every 100 --checkpoint_every 50 --allow_soft_placement 

eval:
	python eval.py --eval_train --checkpoint_dir=$(dest)
