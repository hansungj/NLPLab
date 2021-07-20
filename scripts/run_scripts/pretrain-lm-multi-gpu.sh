CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch \
scripts/gpt2/pretrain-lm.py \
--batch_size 8 \
--learning_rate 1e-5 \
--scheduler True \
--num_epochs 1 \
--num_warming_steps 0.05 \
--n_gpu 2 \
--grad_norm_clip 1 \
--use_cuda True \
--max_context_length 128 \
--max_target_length  92 \
--weight_decay 0.01 \
--scheduler cosine \
--output_dir pretrain-checkpoint-new-seg-diff \
>> pretrain_lm_seg_diff.out