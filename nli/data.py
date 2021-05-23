'''
will contain all data loader

'''

import json
import numpy as np
import torch
import re
from torch.utils.data import DataLoader, Dataset

from nli.utils import open_tsv_file

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
		obs, hyp1, hyp2, label = self.process(idx)
		return obs, hyp1, hyp2, label

	def process(self, idx):
		obs1 = self.tokenize(self.dataset['obs1'][idx])
		obs2 = self.tokenize(self.dataset['obs2'][idx])
		hyp1 = self.tokenize(self.dataset['hyp1'][idx])
		hyp2 = self.tokenize(self.dataset['hyp2'][idx])
		obs = obs1 + obs2
		label = self.dataset['label'][idx]
		return obs, hyp1, hyp2, label

	def tokenize(self, x):
		tokenized = re.split(r'\s+', x.strip())
		return tokenized


class AlphaDataset(Dataset):

	def __init__(self,
				data_path,
				tokenizer, 
				max_samples=None):

		self.dataset = open_tsv_file(data_path, dic=True)
		self.tokenizer = tokenizer
		self.max_samples = max_samples 

	def __len__(self):
		if self.max_samples is None:
			return len(self.dataset['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		pass

