CUDA_VISIBLE_DEVICES=4 nohup python scripts/bert/advanced_train_test.py \
--model_type dual_enc_bert \
--pretrained_name bert-base-uncased \
--train_tsv data/alphanli/tsv/train.tsv \ 
--val_tsv data/alphanli/tsv/dev.tsv \
--batch_size 8 \
--early_stopping 1\
--num_epochs 5 \
--learning_rate 4e-5 \
--scheduler True \
--use_cuda True \
--evaluate True \
--scheduler True \
--seed 1234 \
>> dual_enc_non_pretrained.out