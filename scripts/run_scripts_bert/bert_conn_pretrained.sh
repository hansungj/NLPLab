CUDA_VISIBLE_DEVICES=4 nohup python scripts/bert/advanced_train_test.py \
--model_type pretrained-transformers-cls \
--pretrained_name bert_pretrained_BERTmlm\600000 \
--train_tsv data/alphanli/tsv/train.tsv \ 
--val_tsv data/alphanli/tsv/dev.tsv \
--batch_size 8 \
--early_stopping 1 \
--num_epochs 15 \
--learning_rate 4e-5 \
--scheduler True \
--use_cuda True \
--evaluate True \
--weight_decay 0.01 \
--scheduler True \
--seed 1234 \
>> bert_pretrained.out