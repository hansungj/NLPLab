'''
will contain all data loader

'''

import argparse
import json
import numpy as np
import pickle

from nli.utils import open_tsv_file
import nli.metrics 
import torch
from torch.utils.data import DataLoader, Dataset


def convert_to_tense(x):
	NotImplementedError

class AlphaLoader(DataLoader):

	def __init__(self, **kwargs):

		data_type = kwargs.pop('data_type', 'vector')
		data_path = kwargs.pop('data_path')
		vocab = kwargs.pop('vocab')
		max_samples = kwargs.pop('max_samples', None)
		self.dataset = AlphaDataset(
								data_type,
								data_path,
								vocab,
								max_samples)

		#if kwargs['datatype'] == 'string':
		if data_type == 'string':
			kwargs['collate_fn'] = baseline_collate_fn
		else:
			NotImplementedError

		super().__init__(self.dataset, **kwargs)

def baseline_collate_fn(x):
	# leave it as is without changing ti to tensor
	obs = x[0][0]
	hyp1 = x[0][1]
	hyp2 = x[0][2]
	label = x[0][3]
	return obs, hyp1, hyp2, label

class AlphaDataset(Dataset):

	def __init__(self,
				 data_type,
				 data_path,
				 vocab,
				 max_samples):

		self.data_type = data_type
		self.data_path = data_path
		self.vocab = vocab
		self.max_samples = max_samples

		#load vocab
		with open(vocab, 'r') as f:
			self.vocab = json.loads(f.read())

		#for annotation - load witout tokenizing 
		if self.data_type == 'tsv': # load directly from unprocessed data
			data = open_tsv_file(self.data_path)

			self.story_ids = data['story_id']
			self.obs1 = data['obs1']
			self.obs2 = data['obs2']
			self.hyp1 = data['hyp1']
			self.hyp2 = data['hyp2']
			self.label = np.array(data['label'], dtype=np.int32)

		# for baseline load without mapping to idx
		elif self.data_type == 'pickle':
			with open(self.data_path, 'rb') as f:
				dataset = pickle.load(f)

			self.obs = dataset['obs']
			self.hyp1 = dataset['hyp1']
			self.hyp2 = dataset['hyp2']
			self.label = dataset['label'] # convert to 0 or 1

		# for vector based models
		elif data_type == 'h5': # load from processed data
			NotImplementedError


	def __len__(self):
		if self.max_samples is not None:
			return self.max_samples

		return len(self.obs)

	def __getitem__(self, idx):

		if self.data_type == 'tsv':
			return self.obs1[idx], self.obs2[idx], self.hyp1[idx], self.hyp2[idx], self.label[idx]

		if self.data_type == 'pickle':
			return self.obs[idx], self.hyp1[idx], self.hyp2[idx], self.label[idx]

