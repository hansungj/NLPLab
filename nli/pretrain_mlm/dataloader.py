"""
	Author: Anastasiia Sirotina
	Data loader for batching/collating for BERT 
"""

import json
import numpy as np
import re
from datasets import load_dataset
from transformers import BertTokenizer
import random


import torch
from torch.utils.data import DataLoader, Dataset

class MLM_Dataloader(DataLoader):

	def __init__(self,**kwargs):

		data = kwargs.pop('data')
		tokenizer = kwargs.pop('tokenizer')
		context_left = kwargs.pop('context_left')
		context_right = kwargs.pop('context_right')
		max_context_length = kwargs.pop('max_context_length')
		max_target_length = kwargs.pop('max_target_length')
		masking_prob = kwargs.pop('masking_prob')
		number_of_samples = kwargs.pop('number_of_samples')
		self.tokenizer = tokenizer

		dataset = MLM_Dataset(data,
					tokenizer,
					max_context_length,
					max_target_length,
					context_left,
					context_right,
					number_of_samples,
					masking_prob = masking_prob,
					ignore_index = -100,
					max_samples = None)
		
		kwargs['collate_fn'] = self.mlm_collate_fn
		kwargs['dataset'] = dataset
		super().__init__(**kwargs)
	

	def mlm_collate_fn(self, batch):
		item={}
		for key in batch[0].keys():
			item[key] = [d[key] for d in batch]

		input_ids = self.padding(item['input_ids'], 0)
		masks = self.padding(item['masks'], 0)
		segment_ids = self.padding(item['segment_ids'], 0)
		target_ids = self.padding(item['target_ids'], -100)

		d = {}
		d['input_ids'] = input_ids
		d['masks'] = masks
		d['reference'] = item['reference']
		d['target_ids'] = target_ids
		d['segment_ids'] = segment_ids
		return d

	def padding(self, datalist, pad_token_id):
		#pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
		max_len = max([len(item) for item in datalist])
		padded_datalist = torch.zeros((len(datalist), max_len)).long()
		for i in range(len(datalist)):
			padded_datalist[i, :len(datalist[i])] = datalist[i]
			if len(datalist[i]) < max_len:
				padded_datalist[i, len(datalist[i]):] = torch.Tensor([pad_token_id]*(max_len - len(datalist[i]))).long()
		return padded_datalist
		
	

class MLM_Dataset(Dataset):
	
	def __init__(self,
				data,
				tokenizer,
				max_context_length,
				max_target_length,
				context_left,
				context_right,
				number_of_samples,
				masking_prob = 0.15,
				ignore_index = -100,
				max_samples = None):
				
		#loading dataset using huggingface's load_dataset
		#dataset_preload = load_dataset(data, cache_dir = '../hugginface') #split='train'
		dataset_preload = load_dataset(data)
		
		#self.data will be of the following format {'text':[sent_1, sent_2, ....]}
		#self.data[i] will return the following {'text':[sent_i]}
		self.data = dataset_preload['train'][:number_of_samples]
		#self.data = dataset_preload['train']
		
		self.max_samples = max_samples
		self.tokenizer = tokenizer
		self.masking_prob = masking_prob
		self.ignore_index = ignore_index
		self.max_context_length = max_context_length
		self.max_target_length = max_target_length
		self.context_left = context_left
		self.context_right = context_right
		
		super().__init__()
		
	def __len__(self):
		return len(self.data['text'])-2

	def __getitem__(self, idx):
		
		
		if self.context_left:
		
		#sentence 0 doesn't have a previous sentence
			if idx == 0:
				idx += 1
			output_sent = self.data['text'][idx]
			left_input_sent = self.data['text'][idx-1]

		if self.context_right:
			if idx == len(self.data):
				idx -= 1
			right_input_sent = self.data['text'][idx+1]
			output_sent = self.data['text'][idx]

		if self.context_left and self.context_right:
			input_sent = (left_input_sent, right_input_sent)
		elif self.context_left:
			input_sent = (left_input_sent,)
		elif self.context_right:
			input_sent = (right_input_sent,)
		else:
			raise ValueError('Both --context_left and --context_right cannot be set to False')

		input, target, masking, segment_ids = self.tokenize_and_mask(input_sent, output_sent, self.masking_prob)

		item = {}
		item['input_ids'] = torch.tensor(input)
		item['masks'] = torch.tensor(masking)
		context = ' [SEP] '.join(list(input_sent))
		item['reference'] = '[CLS] ' + context + ' [SEP] ' + output_sent + ' [EOS]'
		item['target_ids'] = torch.tensor(target)
		item['segment_ids'] = torch.tensor(segment_ids)
		return item
	

	def tokenize_and_mask(self, input_sent, output_sent, masking_prob = 0.15):
		"""
		For a pair of sentences "He has an elder sister. Her name is Sarah." , we create the following input and output lists:
			input = 	CLS S1_1 S1_2 S1_3 S1_4 S1_5 S1_6 SEP S2_1 S2_2 S2_3 S2_4 S2_5 EOS
			target = 	IGN IGN  IGN  IGN  IGN  IGN  IGN  IGN S2_1 S2_2 S2_3 S2_4 S2_5 EOS
		Input consist of a CLS token then indexes of the input's tokens a separation token and indexes of the output's tokens
		Output consist of N times ignore_index token, where N-1 is the length of input, indexes of the output's tokens a separation token
		and an end of sentence token,
		Ignore_index is a parameter of torch.nn.CrossEntropyLoss that specifies a target value that is ignored and does not contribute to the input gradient. 
		For BERT model Ignore_index = -100.
		
		For performing masking, tutorial by James Briggs was used:
		https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
		"""
		input = []
		target = []

		input.append(self.tokenizer.cls_token)
		
		for input_sent_ in input_sent:
			input_tokenized = self.tokenizer.tokenize(input_sent_)
			if len(input_tokenized) < self.max_context_length:
				input += input_tokenized
			else:
				input += input_tokenized[:self.max_context_length]

			input.append(self.tokenizer.sep_token)

		input = self.tokenizer.convert_tokens_to_ids(input)

		target_tokenized = self.tokenizer.tokenize(output_sent)
		
		if len(target_tokenized) < self.max_target_length:
			target = target_tokenized
		else:
			target = target_tokenized[:self.max_target_length]
		
		target.append(self.tokenizer.eos_token)
		
		#deprecated: mask a token with prob of 15%,
		#random_nums = torch.rand(len(target))
		##mask a token with prob of 15%, excluding CLS tokens, SEP tokens
		##generating a masking filter 
		#masking = (random_nums < masking_prob) * (target != self.tokenizer.cls_token) \
			#* (target != self.tokenizer.sep_token)  * (target != self.tokenizer.eos_token)
		## which tokens have to be masked according to the masking filter
		#tokens_to_mask = []
		#for i in range(len(masking)):
			#if masking[i] == True:
					#tokens_to_mask.append(i)

		target_unmasked = [self.ignore_index]*len(target)
		target_masked = self.tokenizer.convert_tokens_to_ids(target)
		
		indeces = [t for t in range(len(target)-1)] # -1 so that we don't mask the EOS token
		#print('indeces = {}'.format(indeces))
		
		num_to_mask = round(len(target)*masking_prob)
		#print('num_to_mask = {}'.format(num_to_mask))
		
		tokens_to_mask = random.sample(indeces, k=num_to_mask)
		#print('tokens_to_mask = {}'.format(tokens_to_mask))
		
		randoms = list(torch.rand(len(tokens_to_mask)))
		#print('randoms = {}'.format(randoms))
				
		mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
		
		for idx in tokens_to_mask:
		
			#print('idx = {}'.format(idx))
			target_unmasked[idx] = target_masked[idx]
			prob = randoms.pop()
			#print('prob = {}'.format(prob))
			
			if prob <= 0.8:
				target_masked[idx] = mask_token
				
			elif (prob > 0.8) and (prob <= 0.9):
				random_token = random.randint(0, len(self.tokenizer)-1)
				#print('random_token = {}'.format(random_token))
				#print('target_masked[idx]  = {}'.format(target_masked[idx]))
				target_masked[idx] = random_token
				
			#elif prob >0.9:
				#target_unmasked[idx] is unchanged
		
		segment_ids = [0] * len(input) + [1] * len(target)

		len_input = len(input)
		input += target_masked
		
		pre_masking = torch.tensor([False]*len_input)

		target = [self.ignore_index] * len_input + target_unmasked
		
		masking_fin = [1]*len(input)
		return input, target, masking_fin, segment_ids

if __name__ == '__main__':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	#tokenizer = BertTokenizer.from_pretrained(args.pretrained_name, cache_dir = '../huggingface') #for running on the ims server

	#ensuring that all functional tokens are added to the default tokenizer
	if tokenizer.cls_token == None:
		tokenizer.add_special_tokens({'cls_token': '<CLS>'})
	if tokenizer.sep_token == None:
		tokenizer.add_special_tokens({'sep_token': '<SEP>'})
	if tokenizer.eos_token == None:
		tokenizer.add_special_tokens({'eos_token': '<EOS>'})
	
	print(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
	print(tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
	print(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
	print(tokenizer.sep_token)
	print(tokenizer.mask_token)
	
	dataset_preload = load_dataset('bookcorpus')
	
	dataloader_kwargs = {'data':'bookcorpus',
						'tokenizer':tokenizer,
						'batch_size':1,
						'shuffle':True,
						'num_workers':0,
						'masking_prob':0.30,
						'max_context_length':128,
						'max_target_length':92,
						'context_left':True,
						'context_right':True,
						'number_of_samples':4
						}

	train_loader = MLM_Dataloader(**dataloader_kwargs)
	
	for batch in enumerate(train_loader):
		print(batch)
		print(batch[1]['input_ids'].size())
		print(batch[1]['masks'].size())
		print(batch[1]['target_ids'].size())
		print(batch[1]['segment_ids'].size())