CUDA_VISIBLE_DEVICES=0 python \
scripts/gpt2/pretrain-lm.py \
--batch_size 8 \
--learning_rate 1e-5 \
--scheduler True \
--num_epochs 5 \
--num_warming_steps 0.05 \
--n_gpu 1 \
--grad_norm_clip 0.5 \
--use_cuda True \
--max_context_length 128 \
--max_target_length 128 \