'''
will contain all data loader

'''

import argparse
import numpy as np

from nli.utils import open_tsv_file
import nli.metrics 
import torch
from torch.utils.data import DataLoader, Dataset


class AlphaLoader(DataLoader):

	def __init__(self, **kwargs):
		raise NotImplementedError

class AlphaDataset(Dataset):

	def __init__(self,
				 tsv,
				 vocab,
				 max_samples,
				 annotate=False,
				 eval_measure='accuracy'):


		self.tsv_path = tsv
		self.vocab = vocab
		self.max_samples = max_samples
		self.annotate = annotate #annotation mode
		self.eval_measure = eval_measure

		#evaluation 
		if eval_measure == 'accuracy':
			self.eval_function = nli.metrics.accuracy
		elif eval_measure == 'precision':
			self.eval_function = nli.metrics.precision
		elif eval_measure == 'recall':
			self.eval_function = nli.metrics.recall
		elif eval_measure == 'f1score':
			self.eval_function = nli.metrics.f1score
		else:
			raise ValueError('unknown evaluation measure')


		if self.tsv_path is not None:
			data = open_tsv_file(self.tsv_path)

			self.story_ids = data['story_id']
			self.obs1 = data['obs1']
			self.obs2 = data['obs2']
			self.hyp1 = data['hyp1']
			self.hyp2 = data['hyp2']
			self.label = np.array(data['label'], dtype=np.int32)

		if not self.annotate:
			# convert natural language to tensor through encoding
			pass


		

	def __len__(self):
		if self.max_samples is not None:
			return self.max_samples

		return len(self.story_ids)

	def __getitem__(self, idx):

		if self.annotate:
			return self.obs1[idx], self.obs2[idx], self.hyp1[idx], self.hyp2[idx], self.label[idx]

		raise NotImplementedError


	def eval(self):

		if self.label is None or self.pred is None:
			return None
		self.eval_result = self.eval_function(self.label, self.pred)

