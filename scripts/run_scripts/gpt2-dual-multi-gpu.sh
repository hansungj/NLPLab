CUDA_VISIBLE_DEVICES=4,7 python -m torch.distributed.launch \
scripts/gpt2/gpt2train.py \
--batch_size 16 \
--learning_rate 8e-5 \
--scheduler True \
--num_epochs 5 \
--early_stopping 3 \
--num_warming_steps 0.05 \
--n_gpu 2 \
--use_cuda True \
--evaluate_training_during_training True \
--evaluate_during_training True \

