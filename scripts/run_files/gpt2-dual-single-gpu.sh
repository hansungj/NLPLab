CUDA_VISIBLE_DEVICES=0 python scripts/gpt2/gpt2train.py \
--batch_size 16 \
--evaluate True \
--learning_rate 2e-5 \
--scheduler True \
--num_epochs 8 \
--early_stopping 3 \
--num_warming_steps 0.05 \
--n_gpu 1 \
--grad_norm_clip 0.5 \
--use_cuda True \
