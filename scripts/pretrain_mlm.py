'''
Author: Anastasiia 
Description: for pretraining BERT
'''
# tutorial by James Briggs used:
# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c

from transformers import BertTokenizer
from transformers import AdamW
from datasets import load_dataset

import numpy as np
import argparse
from tqdm import tqdm
import logging
import time
import os
import torch



from nli.pretrain_mlm.BERT.bert_mlm import BertMLM
from nli.pretrain_mlm.BERT.data import MLM_Dataloader

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

parser.add_argument('--corpus_name', default='bookcorpus', type=str)

parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str, help='can be used to initialize a pretrained model from huggingface')
parser.add_argument('--max_samples_per_epoch', default=1000, type=int, help='Number of samples per epoch')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')

parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')

#training 
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--masking_prob', default=0.15, type=float)

parser.add_argument('--save_model_to', default='nli\pretrain-mlm\BERT\pretrained_model', type=str)

def main(args):
	
	tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
	
	dataloader_kwargs = {'data':args.corpus_name,
						'tokenizer':tokenizer,
						'batch_size':args.batch_size,
						'shuffle':args.shuffle,
						'num_workers':args.num_workers,
						'masking_prob':args.masking_prob
						}
	
	train_loader = MLM_Dataloader(**dataloader_kwargs)
	
	#val_loader = 
	
	model = BertMLM(args.pretrained_name)
	
	optim = AdamW(model.parameters(), lr=5e-5)

	if args.use_cuda:
		if not torch.cuda.is_available():
			print('use_cuda=True but cuda is not available')
		device = torch.device("cuda")
	else:
		device = torch.device('cpu')
	
	model.to(device)
	model.train()
	
	for epoch in range(args.epochs):

		loop = tqdm(train_loader, leave=True)
		
		for batch in loop:

			optim.zero_grad()

			input_ids = batch['input_ids'].to(device)
			masks = batch['masks'].to(device)
			labels = batch['output_ids'].to(device)
			segment_ids = batch['segment_ids'].to(device)

			logits, loss = model(input_ids, segment_ids = segment_ids, attention_mask=masks, labels=labels)
			
			loss.backward()

			optim.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(loss=loss.item())
	
	model.save_pretrained(args.save_model_to)
	

if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)	