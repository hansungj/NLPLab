CUDA_VISIBLE_DEVICES=8 nohup python scripts/gpt2/gpt2train.py \
--batch_size 16 \
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
--contiguous True \
--model_type double-head \
--output_dir checkpoint/gpt2_ctg_2 \
>> gpt2_dual_ctg.out