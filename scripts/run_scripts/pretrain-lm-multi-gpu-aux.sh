CUDA_VISIBLE_DEVICES=2,3 nohup python -m torch.distributed.launch \
scripts/gpt2/pretrain-lm-aux.py \
--batch_size 8 \
--learning_rate 5e-5 \
--scheduler True \
--num_epochs 10 \
--num_warming_steps 16000 \
--n_gpu 2 \
--use_cuda True \
--max_context_length 92 \
--max_target_length  92 \
--weight_decay 0.01 \
--scheduler cosine \
--output_dir pretrain-checkpoint-aux-added \
--context_window 1 \
--grad_accumulation_steps 8 \
--random_negative_sample 0.5 \
--classifier_num_layers 1 \
--classifier_dropout 0.1 \
>> pretrain_lm_aux_.out