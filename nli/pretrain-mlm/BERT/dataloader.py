"""Data-loader for batching/collating for BERT """

import json
import numpy as np
import re
from datasets import load_dataset


import torch
from torch.utils.data import DataLoader, Dataset

from nli.pretrain-mlm.BERT.BERT_preprocess import prepare_dataset, tokenize_and_mask

class MLMDataset(Dataset):
	
	def __init__(self,
				dataset = 'bookcorpus'
				max_samples=None):
		dataset_preload = load_dataset(dataset) #split='train'
		self.data = dataset_preload['train']
		self.max_samples = max_samples

	def __len__(self):
		if self.max_samples is None:
			return len(self.data)
		return self.max_samples

	def __getitem__(self, idx):
		input = self.data[0][idx]
		output = self.data[1][idx]
		
		datapair = (input, output)
		inputs, masking = tokenize_and_mask(datapair, tokenizer = 'bert-base-uncased', max_length = 60, padding = 'max_length', masking_prob = 0.15)

		item = {}
		item['input_ids'] = torch.tensor(inputs['input_ids'])
		item['masks'] = torch.tensor(masking)
		#item['reference'] = input + ' <SEP> ' + output
		item['reference'] = output
		item['label'] = torch.tensor(inputs['label'])
		return item


def collate_fn(batch):
	item={}
	for key in batch[0].keys():
		item[key] = [d[key] for d in batch]

	input_ids = torch.stack(item['input_ids'])
	masks = torch.stack(item['masks'])
	label = torch.stack(item['label'])

	d = {}
	d['input_ids'] = input_ids
	d['masks'] = masks
	d['reference'] = item['reference']
	d['label'] = label
	return d


def load_dataloader_mlm(dataset, test_dataset, val_dataset, batch_size ,shuffle=True, drop_last = True, num_workers=0):
	dataloader = DataLoader(dataset, 
		batch_size, 
		collate_fn = collate_fn, 
		shuffle=shuffle, 
		drop_last=drop_last,
		num_workers=num_workers )

	test_dataloader = DataLoader(test_dataset, 
		batch_size, 
		collate_fn = collate_fn, 
		shuffle=False, 
		drop_last=False,
		num_workers=num_workers )

	val_dataloader = None
	if val_dataset is not None:
		val_dataloader = DataLoader(val_dataset, 
			batch_size, 
			collate_fn = collate_fn, 
			shuffle=shuffle, 
			drop_last=False,
			num_workers=num_workers)

	return dataloader, test_dataloader, val_dataloader