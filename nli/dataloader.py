'''

Author:  Sungjun Han
Description: contains data loader / dataset objects for the AlphaNLI task 

1. dataset object for BoW baseline 
2. dataset object for simple neural network models (sem-encoder-pooling models)
3. dataset object for transformer models 
4. define dataloader function for (2)
5. define dataloader function for (3)

'''

import json
import numpy as np
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nli.utils import open_tsv_file
from nli.preprocess import tokenize

class AlphaDatasetBaseline(Dataset):
	'''
	Author:  Sungjun Han
	Description : custom dataloader for BoW baseline models 
				- inherits from torch.utils.data.Dataset (mapping style dataset)
	data_path : str
	max_samples : int 
	'''
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
	'''
	Author:  Sungjun Han
	Description : custom dataloader for DL baseline models 
				- inherits from torch.utils.data.Dataset (mapping style dataset)
	data_path : str
	tokenizer : python object 
	max_samples : int 
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

		items = {}
		items['hyp1'], items['hyp1_mask'], items['hyp1_reference'] = self.preprocess_hypothesis(self.data['hyp1'][idx])
		items['hyp2'], items['hyp2_mask'], items['hyp2_reference'] = self.preprocess_hypothesis(self.data['hyp2'][idx])

		observation = [self.data['obs1'][idx], self.data['obs2'][idx]]
		items['obs'], items['obs_mask'], items['obs_reference'] = self.preprocess_premise(observation)
		items['label'] = torch.tensor(self.data['label'][idx])
		items['pad_id'] = self.tokenizer.vocab['token2idx'][self.tokenizer.pad_token]

		return items

	def preprocess_hypothesis(self, hyp):
		'''
		Description: prepares hypothesis - adding start token / end token and encode
		hyp : str
		'''
		hyp_tokens = self.tokenizer.tokenize(hyp)
		hyp_tokens.insert(0, self.tokenizer.start_token)
		hyp_tokens.append(self.tokenizer.end_token)
		hyp_ids = self.tokenizer.convert_tokens_to_ids(hyp_tokens)
		masks = [1]*len(hyp_ids)
		return torch.tensor(hyp_ids), torch.tensor(masks), hyp

	def preprocess_premise(self, obs):
		'''
		Description: prepares premise - adding start token / end token, and encode
		obs : str
		'''
		obs = (' ' + self.tokenizer.split_token + ' ').join(obs) # sentence </s> sentence 
		tokens = self.tokenizer.tokenize(obs)
		tokens.insert(0, self.tokenizer.start_token)
		tokens.append(self.tokenizer.end_token)
		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)
		masks = [1]*len(tokens_id)
		return torch.tensor(tokens_id), torch.tensor(masks), obs

def merge(sequences, pad_id):
	'''
	Author:  Sungjun Han
	Description: Merges the batch by padding to the max length
	sequences : list 
	pad_id : integer 
	'''
	lengths = [len(l) for l in sequences]
	max_length = max(lengths)

	padded_batch = torch.full((len(sequences), max_length), pad_id).long()
	#pad to max_length
	for i, seq in enumerate(sequences):
		padded_batch[i, :len(seq)] = seq
	return padded_batch, torch.LongTensor(lengths)

def alpha_collate_fn_base(batch):
	'''
	Author:  Sungjun Han
	Description:  function for the sem-encoder-pooling models
	batch : list of dictionaries
	'''
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
	Author:  Sungjun Han
	Description: custom dataloader for BERT (transformer models from huggingface) models 
			- inherits from torch.utils.data.Dataset (mapping style dataset)
			Prepares input as [obs1, [SEP], obs2, [SEP], hyp1, [SEP], hyp2]. 
			Depending on the task [CLS] token will be inserted at position 0 or -1
	data_path : str
	tokenizer : python object (huggingface)
	max_samples : int 
	sep_token : str
	ped_token_id : int
	cls_at_start : bool
	'''
	def __init__(self,
				data_path,
				tokenizer, 
				max_samples=None,
				sep_token=None,
				pad_token_id=None,
				cls_at_start=True):
		self.data = open_tsv_file(data_path, dic=True)
		self.tokenizer = tokenizer
		self.max_samples = max_samples
		self.pad_token_id = tokenizer.pad_token_id if pad_token_id is  None else pad_token_id
		self.sep_token = tokenizer.sep_token if sep_token is None else sep_token
		self.cls_at_start = cls_at_start

	def __len__(self):
		if self.max_samples is None:
			return len(self.data['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		observation = (' ' + self.sep_token + ' ').join(['observation 1 :'+ self.data['obs1'][idx], 'observation 2 :'+self.data['obs2'][idx]])
		hypotheses = (' ' + self.sep_token + ' ').join(['hypothesis 1 :'+ self.data['hyp1'][idx],'hypothesis 2 :'+  self.data['hyp2'][idx]])

		tokens = self.tokenizer.tokenize(observation)
		#if we are working with a transformer encoder 
		if self.cls_at_start: 
			#tokens.insert(0, self.tokenizer.bos_token)
			tokens.insert(0, self.tokenizer.cls_token)

		tokens.append(self.sep_token)
		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

		segment_ids = [0]*len(tokens_id)

		# if we are working with a transformer decoder 
		if not self.cls_at_start:
			tokens.append(self.tokenizer.eos_token)
			tokens.append(self.tokenizer.cls_token)

		tokens = self.tokenizer.tokenize(hypotheses)
		hyp_id = self.tokenizer.convert_tokens_to_ids(tokens)

		segment_ids.extend([1]*len(hyp_id))
		tokens_id.extend(hyp_id)
		masks = [1]*len(tokens_id)

		item = {}
		item['input_ids'] = torch.tensor(tokens_id)
		item['segment_ids'] = torch.tensor(segment_ids)
		item['masks'] = torch.tensor(masks)
		item['reference'] = observation + self.sep_token + hypotheses
		item['label'] = torch.tensor(self.data['label'][idx])
		item['pad_id'] = self.pad_token_id

		return item

def alpha_collate_fn_transformer(batch):
	'''
	Author:  Sungjun Han
	Description: Colate_fn function for the BERT (transformer) models	
	batch : list of dictionaries 
	'''

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
	d['input_lengths'] = input_length
	d['masks'] = masks
	d['reference'] = item['reference']
	d['label'] = label
	return d

def load_dataloader_base(
	dataset, 
	test_dataset, 
	val_dataset, 
	batch_size, 
	shuffle=True, 
	drop_last = True, 
	num_workers = 0):
	'''
	Author:  Sungjun Han
	Description: prepares dataloader for train/val/test for DL baseline models 
			distributed not supported
	
	dataset : torch.utils.data.Dataset object 
	test_dataset : torch.utils.data.Dataset object 
	val_dataset : torch.utils.data.Dataset object 
	batch_size : int
	shuffle : bool
	drop_last : bool
	num_workers : bool
	'''
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

def prepare_dataloader(dataset, 
	test_dataset, 
	val_dataset, 
	batch_size ,
	collate_fn,
	shuffle=True, 
	drop_last = True, 
	num_workers=0,
	distributed = False ):
	'''
	Author:  Sungjun Han
	Description: prepares dataloader for train/val/test for transformer models 
		note that we do not test using with multiple gpus

	dataset : torch.utils.data.Dataset object 
	test_dataset : torch.utils.data.Dataset object 
	val_dataset : torch.utils.data.Dataset object 
	batch_size : int
	shuffle : bool
	drop_last : bool
	num_workers : bool
	'''
	dataloader = None
	test_dataloader = None
	val_dataloader = None

	if dataset:
		train_shuffle = shuffle # so that we keep this value for valloader
		sampler = None
		if distributed:
			sampler = DistributedSampler(dataset, shuffle=train_shuffle)
			train_shuffle = False 
		dataloader = DataLoader(dataset, 
			batch_size, 
			collate_fn = collate_fn, 
			shuffle=train_shuffle, 
			drop_last=drop_last,
			sampler = sampler,
			num_workers=num_workers )

	if test_dataset:
		test_dataloader = DataLoader(test_dataset, 
			batch_size, 
			collate_fn = collate_fn, 
			shuffle=False, 
			drop_last=False,
			sampler = None,
			num_workers=num_workers )

	sampler = None
	if val_dataset:
		if distributed:
			sampler = DistributedSampler(val_dataset, shuffle=shuffle)
			shuffle = False 
			
		val_dataloader = DataLoader(val_dataset, 
			batch_size, 
			collate_fn = collate_fn, 
			shuffle=shuffle, 
			drop_last=False,
			sampler = sampler, 
			num_workers=num_workers)

	return dataloader, test_dataloader, val_dataloader


class AlphaDatasetDualEncoder(Dataset):
	'''
	Author: Anastasiia
	Description: Prepares  2 instances: [[CLS], obs1, [SEP], obs2 [SEP], hyp1, [EOS]] and [[CLS], obs1, [SEP], obs2 [SEP], hyp2, [EOS]]. 	
	'''
	def __init__(self,
				data_path,
				tokenizer, 
				max_samples=None,
				sep_token=None,
				pad_token_id=None
				):

		self.data = open_tsv_file(data_path, dic=True)
		self.tokenizer = tokenizer
		self.max_samples = max_samples
		self.pad_token_id = tokenizer.pad_token_id if pad_token_id is None else pad_token_id
		self.sep_token = tokenizer.sep_token if sep_token is None else sep_token

	def __len__(self):
		if self.max_samples is None:
			return len(self.data['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		observations = (' ' + self.sep_token + ' ').join([self.data['obs1'][idx], self.data['obs2'][idx]])
		hypotheses = (' ' + self.sep_token + ' ').join(['hypothesis 1 :'+ self.data['hyp1'][idx],'hypothesis 2 :'+  self.data['hyp2'][idx]])
		tokenized_observations = self.tokenizer.tokenize(observations)

		input1 = []
		input1.append(self.tokenizer.cls_token)
		input1 += tokenized_observations
		input1.append(self.sep_token)

		segment_ids1 = [0]*len(input1)

		hyp1 = self.data['hyp1'][idx]
		hyp1_tokenized = self.tokenizer.tokenize(hyp1)
		input1 += hyp1_tokenized
		input1.append(self.tokenizer.eos_token)
		input1 = self.tokenizer.convert_tokens_to_ids(input1)

		segment_ids1.extend([1]*(len(hyp1_tokenized)+1))
		masks1 = [1]*len(input1)

		input2 = []
		input2.append(self.tokenizer.cls_token)
		input2 += tokenized_observations
		input2.append(self.sep_token)

		segment_ids2 = [0]*len(input2)
		
		hyp2 = self.data['hyp2'][idx]
		hyp2_tokenized = self.tokenizer.tokenize(hyp2)
		input2 += hyp2_tokenized
		input2.append(self.tokenizer.eos_token)
		input2 = self.tokenizer.convert_tokens_to_ids(input2)

		segment_ids2.extend([1]*(len(hyp2_tokenized)+1))
		masks2 = [1]*len(input2)

		item = {}
		item['input1'] = torch.tensor(input1)
		item['segment_ids1'] = torch.tensor(segment_ids1)
		item['masks1'] = torch.tensor(masks1)

		item['input2'] = torch.tensor(input2)
		item['segment_ids2'] = torch.tensor(segment_ids2)
		item['masks2'] = torch.tensor(masks2)

		item['reference'] = observations + self.sep_token + hypotheses
		item['label'] = torch.tensor(self.data['label'][idx])
		return item


class DualEncoder_Dataloader(DataLoader):
	'''
	Author: Anastasiia
	'''
	def __init__(self,**kwargs):

		tokenizer = kwargs.pop('tokenizer')
		self.tokenizer = tokenizer
		
		kwargs['collate_fn'] = self.collate_fn_BBDualEnc
		super().__init__(**kwargs)
	

	def collate_fn_BBDualEnc(self, batch):
		'''
		Author: Anastasiia
		'''
		item={}
		for key in batch[0].keys():
			item[key] = [d[key] for d in batch]

		input1 = self.padding(item['input1'])
		masks1 = self.padding(item['masks1'])
		segment_ids1 = self.padding(item['segment_ids1'])

		input2 = self.padding(item['input2'])
		masks2 = self.padding(item['masks2'])
		segment_ids2 = self.padding(item['segment_ids2'])
		
		label = torch.stack(item['label']).float()

		d = {}
		d['input1'] = input1
		d['masks1'] = masks1
		d['segment_ids1'] = segment_ids1
		d['input2'] = input2
		d['masks2'] = masks2
		d['segment_ids2'] = segment_ids2
		d['reference'] = item['reference']
		d['label'] = label

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

def load_dataloader_BBDualEnc(tokenizer, dataset, test_dataset, val_dataset, batch_size ,shuffle=True, drop_last = True, num_workers=0):
	'''
	Author:  Anastasiia
	Description:
	prepares dataloader for train/val/test for bert based dual encoder models 
	'''
	train_kwargs = {'dataset':dataset,
					'tokenizer':tokenizer,
					'batch_size':batch_size,
					'shuffle':shuffle,
					'num_workers':num_workers,
					'drop_last':drop_last}

	dataloader = DualEncoder_Dataloader(**train_kwargs)

	test_kwargs = {'dataset':test_dataset,
					'tokenizer':tokenizer,
					'batch_size':batch_size,
					'shuffle':shuffle,
					'num_workers':num_workers,
					'drop_last':drop_last}

	test_dataloader = DualEncoder_Dataloader(**test_kwargs)

	val_dataloader = None
	if val_dataset is not None:
		val_kwargs = {'dataset':val_dataset,
					'tokenizer':tokenizer,
					'batch_size':batch_size,
					'shuffle':shuffle,
					'num_workers':num_workers,
					'drop_last':drop_last}
					
		val_dataloader = DualEncoder_Dataloader(**val_kwargs)

	return dataloader, test_dataloader, val_dataloader
