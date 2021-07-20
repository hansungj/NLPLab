CUDA_VISIBLE_DEVICES=1,3 \
nohup \
python -m torch.distributed.launch \
scripts/train_test.py \
--model_type pretrained-transformers-pooling \
--pretrained_name gpt2  \
--batch_size 32 \
--evaluate True \
--learning_rate 5e-5 \
--use_cuda True \
--scheduler True \
--num_epochs 5 \
--early_stopping 3 \
--num_warming_steps 0.05 \
--n_gpu 2