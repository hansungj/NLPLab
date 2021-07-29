CUDA_VISIBLE_DEVICES=8 nohup python scripts/gpt2/gpt2train.py \
--batch_size 8 \
--learning_rate 8e-5 \
--scheduler True \
--num_epochs 10 \
--early_stopping 0 \
--num_warming_steps 0.02 \
--n_gpu 1 \
--use_cuda True \
--evaluate_training_during_training True \
--evaluate_during_training True \
--weight_decay 0.01 \
--scheduler cosine \
--output_dir checkpoint/gpt2_pretrained_new_data_3 \
--model_type double-head \
--auxiliary_loss_lambda 0.5 \
--from_pretrained ./pretrain-checkpoint-aux-added/checkpoint_gpt2/gpt2-step-1200000/ \
--notes "1. test new data format. 2. lambda 0.5 3. further pretrained"  \
>> gpt2_dual_3.out
