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


import nli.utils as utils 
from nli.pretrain_mlm.bert_mlm import BertMLM
from nli.pretrain_mlm.dataloader import MLM_Dataloader

from transformers import get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

parser.add_argument('--corpus_name', default='bookcorpus', type=str)

parser.add_argument('--number_of_samples', default=None, type=int, help='Number of samples used for training')
parser.add_argument('--max_context_length', default=128, type=int, help='Max length of a context sentence')
parser.add_argument('--max_target_length', default=92, type=int, help='Max length of a target sentence')
parser.add_argument('--context_left', default=True, type=bool, help='Whether the context includes the sentences to the left of the target')
parser.add_argument('--context_right', default=True, type=bool, help='Whether the context includes the sentences to the right of the target')

parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str, help='can be used to initialize a pretrained model from huggingface')


#training
parser.add_argument('--max_samples_per_epoch', default=1000, type=int, help='Number of samples per epoch')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')

parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')

parser.add_argument('--grad_norm_clip', default=1, type=float, help='clip the norm')
parser.add_argument('--scheduler', default=None, type =bool, help='')
parser.add_argument('--num_warmup_steps', default=10000, type =int, help='')


parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--masking_prob', default=0.15, type=float)

parser.add_argument('--save_model_to', default='pretrained_BERTmlm', type=str)

def main(args):

	utils.set_seed(args.seed)

	
	tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
	#tokenizer = BertTokenizer.from_pretrained(args.pretrained_name, cache_dir = '../hugginface')
	
	if tokenizer.cls_token == None:
		tokenizer.add_special_tokens({'cls_token': '<CLS>'})
	if tokenizer.sep_token == None:
		tokenizer.add_special_tokens({'sep_token': '<SEP>'})
	if tokenizer.eos_token == None:
		tokenizer.add_special_tokens({'eos_token': '<EOS>'})

	logger.info('Further pretraining BERT model: {}'.format(args.pretrained_name))

	dataloader_kwargs = {'data':args.corpus_name,
						'tokenizer':tokenizer,
						'batch_size':args.batch_size,
						'shuffle':args.shuffle,
						'num_workers':args.num_workers,
						'masking_prob':args.masking_prob,
						'max_context_length':args.max_context_length,
						'max_target_length':args.max_target_length,
						'context_left':args.context_left,
						'context_right':args.context_right,
						'number_of_samples':args.number_of_samples
						}

	
	logger.info('Creating Dataloader:')
	logger.info(dataloader_kwargs)

	train_loader = MLM_Dataloader(**dataloader_kwargs)

	#val_loader = 
	
	logger.info('Initializing a BERT model: {}'.format(args.pretrained_name))
	model = BertMLM(args.pretrained_name, tokenizer)
	
	optim = AdamW(model.parameters(), lr=5e-5)

	if args.use_cuda:
		if not torch.cuda.is_available():
			print('use_cuda=True but cuda is not available')
		device = torch.device("cuda")
	else:
		device = torch.device('cpu')
		
	model_kwargs = {'epochs':args.epochs,
						'max_samples_per_epoch':args.max_samples_per_epoch,
						'grad_norm_clip':args.grad_norm_clip,
						'scheduler':args.scheduler,
						'num_warmup_steps':args.num_warmup_steps,
						'use_cuda':args.use_cuda
						}
	logger.info('Model settings:')
	logger.info(dataloader_kwargs)
	
	model.to(device)
	model.train()
	starttime = None
	
	scheduler = None 
	if args.scheduler:
		num_training_steps = int((len(train_loader)//args.batch_size)*args.num_epochs)
		scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps,num_training_steps)
	
	for epoch in range(args.epochs):
		
		#if not starttime:
			#starttime = time.time()
			#logger.info('Starting epoch {}'.format(epoch))
		#else:
			#logger.info('The epoch {} lasted for '.format(epoch-1, starttime - time.time()))
			#logger.info('Starting epoch {}'.format(epoch))
			#starttime = time.time()
		
		starttime_loop = time.time()
		
		loop = tqdm(train_loader, leave=True)
		
		#logger.info('Loading data lasted for'.format(starttime_loop - time.time()))
		
		step = 0
		
		for batch in loop:
			#logger.info('Working with batch {}'.format(t))
			optim.zero_grad()
			
			input_ids = batch['input_ids'].to(device)
			masks = batch['masks'].to(device)
			labels = batch['target_ids'].to(device)
			segment_ids = batch['segment_ids'].to(device)

			logits, loss = model(input_ids, segment_ids = segment_ids, attention_mask=masks, labels=labels)
			
			loss.backward()
			
			if args.grad_norm_clip:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

			optim.step()

			if scheduler:
				scheduler.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(loss=loss.item())

			step += 1
			logger.info('Epoch: {}, step {}, loss value {}'.format(epoch, step, loss.item()))

			if step % 100000 == 0:
				save_path = args.save_model_to + '/' + str(step)
				model.model.save_pretrained(save_paths)

	
	#wrapped_model = model.model.base_model
	#wrapped_model.save_pretrained(args.save_model_to)
	
	model.model.save_pretrained(args.save_model_to)
	

if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s',
		handlers=[
			logging.FileHandler("BERT_pretraining.log"),
			logging.StreamHandler()
		]
		)
	#logging.FileHandler('BERT_pretraining.log')
	main(args)