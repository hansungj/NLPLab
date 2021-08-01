CUDA_VISIBLE_DEVICES=2 nohup python scripts/bert/advanced_train_test.py \
--model_type dual_enc_bert \
--pretrained_name /mount/arbeitsdaten/studenten1/team-lab-nlp/group5/NLPLab_BERT/NLPLab/specific_masking_pretrained_BERTmlm/0/400000 \
--train_tsv data/alphanli/tsv/train.tsv \
--test_tsv data/alphanli/tsv/dev.tsv \
--batch_size 16 \
--early_stopping 0 \
--output_dir checkpoint/bert_5 \
--num_epochs 5 \
--evaluate True \
--learning_rate 1e-5 \
--use_cuda True \
--scheduler True \
--weight_decay 0.01 \
--scheduler True \
--num_warming_steps 0.1 \
>> bert-dual-pretrained-2.out