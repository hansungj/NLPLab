'''
will contain all data loader

'''

import json
import numpy as np
import re



import torch
from torch.utils.data import DataLoader, Dataset

from nli.utils import open_tsv_file
from nli.preprocess import tokenize

class AlphaDatasetBaseline(Dataset):
	def __init__(self,
				 data_path, 
				 max_samples=None):

		self.dataset = open_tsv_file(data_path, dic=True)
		self.max_samples = max_samples 

	def __len__(self):
		if self.max_samples is None:
			return len(self.dataset['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		obs1 = tokenize(self.dataset['obs1'][idx])
		obs2 = tokenize(self.dataset['obs2'][idx])
		hyp1 = tokenize(self.dataset['hyp1'][idx])
		hyp2 = tokenize(self.dataset['hyp2'][idx])
		obs = obs1 + obs2
		label = self.dataset['label'][idx]
		return obs, hyp1, hyp2, label

class AlphaDataset(Dataset):
	def __init__(self,
				data_path,
				tokenizer, 
				max_samples=None):
		self.data = open_tsv_file(data_path, dic=True)
		self.tokenizer = tokenizer
		self.max_samples = max_samples

	def __len__(self):
		if self.max_samples is None:
			return len(self.data['obs1'])
		return self.max_samples

	def __getitem__(self, idx):

		items = {}
		items['hyp1'], items['hyp1_mask'], items['hyp1_reference'] = self.preprocess_hypothesis(self.data['hyp1'][idx])
		items['hyp2'], items['hyp2_mask'], items['hyp2_reference'] = self.preprocess_hypothesis(self.data['hyp2'][idx])

		observation = [self.data['obs1'][idx], self.data['obs2'][idx]]
		items['obs'], items['obs_mask'], items['obs_reference'] = self.preprocess_premise(observation)
		items['label'] = torch.tensor(self.data['label'][idx])
		items['pad_id'] = self.tokenizer.vocab['token2idx'][self.tokenizer.pad_token]

	def preprocess_hypothesis(self, hyp):
		hyp_tokens = self.tokenizer.tokenize(hyp)
		hyp_tokens.insert(0, self.tokenizer.start_token)
		hyp_tokens.append(self.tokenizer.end_token)
		hyp_ids = self.tokenizer.convert_tokens_to_id(hyp_tokens)
		masks = [1]*len(hyp_ids)
		return torch.tensor(hyp_ids), torch.tensor(masks), hyp

	def preprocess_premise(self, obs):
		obs = (' ' + self.tokenizer.split_token + ' ').join(obs) # sentence </s> sentence 
		tokens = self.tokenizer.tokenize(obs)
		tokens.insert(0, self.tokenizer.start_token)
		tokens.append(self.tokenizer.end_token)
		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
		masks = [1]*len(tokens_id)
		return torch.tensor(tokens_id), torch.tensor(masks), obs

def merge(sequences, pad_id):
		lengths = [len(l) for l in sequences]
		max_length = max(lengths)

		padded_batch = torch.full((len(sequences), max_length), pad_id).long()
		#pad to max_length
		for i, seq in enumerate(sequences):
			padded_batch[i, :len(seq)] = seq

		return padded_batch, torch.LongTensor(lengths)

def alpha_collate_fn_base(batch):
	item={}
	for key in batch[0].keys():
		item[key] = [d[key] for d in batch] # [item_dic, item_idc ]

	pad_id = item['pad_id'][0]
	hyp1, hyp1_length = merge(item['hyp1'], pad_id)
	hyp1_mask, _ = merge(item['hyp1_mask'], pad_id)
	hyp2, hyp2_length = merge(item['hyp2'], pad_id)
	hyp2_mask, _ = merge(item['hyp2_mask'], pad_id)
	obs, obs_length = merge(item['obs'], pad_id)
	obs_mask, _ = merge(item['obs_mask'], pad_id)
	label = torch.stack(item['label']).float()

	d = {}
	d['hyp1'] = hyp1
	d['hyp1_length'] = hyp1_length
	d['hyp1_mask'] = hyp1_mask
	d['hyp1_reference'] = item['hyp1_reference']

	d['hyp2'] = hyp2
	d['hyp2_length'] = hyp2_length
	d['hyp2_mask'] = hyp2_mask
	d['hyp2_reference'] = item['hyp2_reference']

	d['obs'] = obs
	d['obs_length'] = obs_length
	d['obs_mask'] = obs_mask
	d['obs_reference'] = item['obs_reference']
	d['label'] = label

	return d

class AlphaDatasetTransformer(Dataset):

	'''
	prepares by just catenating everything
	'''
	def __init__(self,
				data_path,
				tokenizer, 
				max_samples=None):
		self.data = open_tsv_file(data_path, dic=True)
		self.tokenizer = tokenizer
		self.max_samples = max_samples

	def __len__(self):
		if self.max_samples is None:
			return len(self.data['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		observation = (' ' + self.tokenizer.sep_token + ' ').join([self.data['obs1'][idx], self.data['obs2'][idx]])
		hypotheses = (' ' + self.tokenizer.sep_token + ' ').join([self.data['hyp1'][idx],self.data['hyp2'][idx]])

		tokens = self.tokenizer.tokenize(observation)
		tokens.insert(0, self.tokenizer.cls_token)
		tokens.append(self.tokenizer.sep_token)
		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

		segment_ids = [0]*len(tokens_id)

		tokens = self.tokenizer.tokenize(hypotheses)
		hyp_id = self.tokenizer.convert_tokens_to_ids(tokens)

		segment_ids.extend([1]*len(hyp_id))
		tokens_id.extend(hyp_id)
		masks = [1]*len(tokens_id)

		item = {}
		item['input_ids'] = torch.tensor(tokens_id)
		item['segment_ids'] = torch.tensor(segment_ids)
		item['masks'] = torch.tensor(masks)
		item['reference'] = observation + ' <SEP> ' + hypotheses
		item['label'] = torch.tensor(self.data['label'][idx])
		item['pad_id'] = self.tokenizer.pad_token_id

		return item

def alpha_collate_fn_transformer(batch):
	item={}
	for key in batch[0].keys():
		item[key] = [d[key] for d in batch] # [item_dic, item_idc ]

	pad_id = item['pad_id'][0]
	input_ids, input_length = merge(item['input_ids'], pad_id)
	segment_ids, _ = merge(item['segment_ids'], pad_id)
	masks, _ = merge(item['masks'], pad_id)
	label = torch.stack(item['label']).float()

	d = {}

	d['input_ids'] = input_ids
	d['segment_ids'] = segment_ids
	d['input_length'] = input_length
	d['masks'] = masks
	d['reference'] = item['reference']
	d['label'] = label
	return d



def load_dataloader_base(dataset, test_dataset, val_dataset, batch_size, shuffle=True, drop_last = True, num_workers = 0):
	dataloader = DataLoader(dataset, 
		batch_size, 
		collate_fn = alpha_collate_fn_base, 
		shuffle=shuffle, 
		drop_last=drop_last,
		num_workers=num_workers )

	test_dataloader = DataLoader(test_dataset, 
		batch_size, 
		collate_fn = alpha_collate_fn_base, 
		shuffle=False, 
		drop_last=False,
		num_workers=num_workers )

	val_dataloader = None
	if val_dataset is not None:
		val_dataloader = DataLoader(val_dataset, 
			batch_size, 
			collate_fn = alpha_collate_fn_base, 
			shuffle=shuffle, 
			drop_last=False,
			num_workers=num_workers )
	return dataloader, test_dataloader, val_dataloader

def load_dataloader_transformer(dataset, test_dataset, val_dataset, batch_size ,shuffle=True, drop_last = True, num_workers=0):
	dataloader = DataLoader(dataset, 
		batch_size, 
		collate_fn = alpha_collate_fn_transformer, 
		shuffle=shuffle, 
		drop_last=drop_last,
		num_workers=num_workers )

	test_dataloader = DataLoader(test_dataset, 
		batch_size, 
		collate_fn = alpha_collate_fn_transformer, 
		shuffle=False, 
		drop_last=False,
		num_workers=num_workers )

	val_dataloader = None
	if val_dataset is not None:
		val_dataloader = DataLoader(val_dataset, 
			batch_size, 
			collate_fn = alpha_collate_fn_transformer, 
			shuffle=shuffle, 
			drop_last=False,
			num_workers=num_workers)

	return dataloader, test_dataloader, val_dataloader
