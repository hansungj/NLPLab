"""
	Author: Anastasiia Sirotina
	Data loader for batching/collating for BERT 
"""

import json
import numpy as np
import re
from datasets import load_dataset


import torch
from torch.utils.data import DataLoader, Dataset

class MLM_Dataloader(DataLoader):

	def __init__(self,**kwargs):

		data = kwargs.pop('data')
		tokenizer = kwargs.pop('tokenizer')
		context_direction = kwargs.pop('context_direction')
		max_context_length = kwargs.pop('max_context_length')
		max_target_length = kwargs.pop('max_target_length')
		masking_prob = kwargs.pop('masking_prob')
		self.tokenizer = tokenizer

		dataset = MLM_Dataset(data,
					tokenizer,
					max_context_length,
					max_target_length,
					context_direction,
					masking_prob = 0.15,
					ignore_index = -100,
					max_samples = None)
		
		kwargs['collate_fn'] = self.mlm_collate_fn
		kwargs['dataset'] = dataset
		super().__init__(**kwargs)
	

	def mlm_collate_fn(self, batch):
		item={}
		for key in batch[0].keys():
			item[key] = [d[key] for d in batch]

		input_ids = self.padding(item['input_ids'])
		masks = self.padding(item['masks'])
		segment_ids = self.padding(item['segment_ids'])
		target_ids = self.padding(item['target_ids'])

		d = {}
		d['input_ids'] = input_ids
		d['masks'] = masks
		d['reference'] = item['reference']
		d['target_ids'] = target_ids
		d['segment_ids'] = segment_ids
		return d

	def padding(self, datalist):
		pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
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
				context_direction,
				masking_prob = 0.15,
				ignore_index = -100,
				max_samples = None):
				
		#loading dataset using huggingface's load_dataset
		dataset_preload = load_dataset(data) #split='train'
		
		#self.data will be of the following format {'text':[sent_1, sent_2, ....]}
		#self.data[i] will return the following {'text':[sent_i]}
		self.data = dataset_preload['train'][:101]['text']
		#self.data = dataset_preload['train']
		
		self.max_samples = max_samples
		self.tokenizer = tokenizer
		self.masking_prob = masking_prob
		self.ignore_index = ignore_index
		self.max_context_length = max_context_length
		self.max_target_length = max_target_length
		self.context_direction = context_direction
		
		super().__init__()
		
	def __len__(self):
		return len(self.data)-2

	def __getitem__(self, idx):
		
		
		if self.context_direction == 'preceed':
		
		#sentence 0 doesn't have a previous sentence
			if idx == 0:
				idx += 1
			output_sent = self.data[idx]
			input_sent = self.data[idx-1]

		elif self.context_direction == 'succeed':
			if idx == len(self.data):
				idx -= 1
			input_sent = self.data[idx+1]
			output_sent = self.data[idx]
		
		else:
			raise ValueError('For --context_direction argument choose one of the values: "preceed"/"succeed"')

		
		input, target, masking, segment_ids = self.tokenize_and_mask(input_sent, output_sent, self.masking_prob)

		item = {}
		item['input_ids'] = torch.tensor(input)
		item['masks'] = torch.tensor(masking)
		item['reference'] = '[CLS] ' + input_sent + ' [SEP] ' + output_sent + ' [EOS]'
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
		For BERT model Ignore_index = -1.
		
		For performing masking tutorial by James Briggs was used:
		https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
		"""
		input = []
		target = []

		input.append(self.tokenizer.cls_token)
		input_tokenized = self.tokenizer.tokenize(input_sent)
		if len(input_tokenized) < self.max_context_length:
				input += input_tokenized
		else:
			input += input_tokenized[:self.max_context_length]

		input.append(self.tokenizer.sep_token)

		random_nums = torch.rand(len(input))

		#mask a token with prob of 15%, excluding CLS tokens, SEP tokens
		#generating a masking filter 
		masking = (random_nums < masking_prob) * (input != self.tokenizer.cls_token) \
			* (input != self.tokenizer.sep_token) # * (input!= self.tokenizer.pad_token)

		# which tokens have to be masked according to the masking filter
		tokens_to_mask = []
		for i in range(len(masking)):
			if masking[i] == True:
					tokens_to_mask.append(i)

		for token in tokens_to_mask:
			input[token] = self.tokenizer.mask_token
		
		input = self.tokenizer.convert_tokens_to_ids(input)

		target_tokenized = self.tokenizer.tokenize(output_sent)
		
		if len(target_tokenized) < self.max_target_length:
				target = target_tokenized
		else:
			target = target_tokenized[:self.max_target_length]
		
		###.add_speical_token
		#if self.tokenizer.eos_token != None:
			#target.append(self.tokenizer.eos_token)
		#else:
			#self.tokenizer.add_special_tokens({'eos_token': '<EOS>'})
			#target.append(self.tokenizer.eos_token)
		target = self.tokenizer.convert_tokens_to_ids(target)
		
		segment_ids = [0] * len(input) + [1] * len(target)

		len_input = len(input)
		input += target
		
		post_masking = torch.tensor([False]*len(target))

		target = [self.ignore_index] * len_input + target
		
		masking_fin = torch.cat((masking, post_masking), 0)
		return input, target, masking_fin, segment_ids
