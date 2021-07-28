CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch \
scripts/gpt2/pretrain-lm.py \
--batch_size 8 \
--learning_rate 5e-5 \
--scheduler True \
--num_epochs 10 \
--num_warming_steps 16000 \
--n_gpu 2 \
--use_cuda True \
--max_context_length 128 \
--max_target_length  92 \
--weight_decay 0.01 \
--scheduler cosine \
--output_dir pretrain-checkpoint-new-lr-new-format-fixed \
--context_window 1 \
--grad_accumulation_steps 8 \
>> pretrain_lm_seg_diff.out 