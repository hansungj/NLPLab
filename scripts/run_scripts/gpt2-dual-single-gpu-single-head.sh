CUDA_VISIBLE_DEVICES=1 nohup python scripts/gpt2/gpt2train.py \
--batch_size 8 \
--learning_rate 8e-5 \
--scheduler True \
--num_epochs 5 \
--early_stopping 3 \
--num_warming_steps 0.05 \
--n_gpu 1 \
--use_cuda True \
--evaluate_training_during_training True \
--evaluate_during_training True \
--weight_decay 0.01 \
--scheduler cosine \
--output_dir checkpoint/gpt2_no_ctg_further_pretrained_single \
--model_type single-head \
--from_pretrained ./pretrain-checkpoint-new/checkpoint_gpt2/gpt2-step-1700000/ \
>> gpt2_dual.out
