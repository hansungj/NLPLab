'''
Author:  Sungjun Han
contains data loader / dataset objects for the pre-training and fine-tuning for GPT-2

1. dataloader for pretraining 
2. dataset object for BookCorpus - for pretraining  
3. dataset object for fine-tuning 
4. dataloader for fine-tuning and zero-shot classification 
'''

import json
import numpy as np
import re
import random

import torch
import torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset # use datasets 

class BookCorpusLmLoader(DataLoader):
    '''
    Author:  Sungjun Han
    '''
    
    def __init__(self,**kwargs):
        
        data = kwargs.pop('data')
        tokenizer = kwargs.pop('tokenizer')
        max_context_length = kwargs.pop('max_context_length')
        max_target_lenfgth = kwargs.pop('max_target_length')
        left_context = kwargs.pop('left_context', True)
        right_context = kwargs.pop('right_context', False )
        context_window = kwargs.pop('context_window', 1)

        # define dataset here 
        dataset = BookCorpusLmDataset(
            data, 
            tokenizer,
            left_context,
            right_context,
            max_context_length,
            max_target_length,
            context_window)

        # if distributed use Sampler and set shuffle to False 
        distributed = kwargs.pop('distributed', False)
        if distributed:
            shuffle=kwargs['shuffle']# we need to define shuffle in terms of our sampler
            kwargs['shuffle'] = False  # set shuffle to false for the dataloader as handled by sampler
            sampler=DistributedSampler(dataset=dataset, shuffle=shuffle)
            kwargs['sampler'] = sampler 
        
        kwargs['collate_fn'] = collate_fn_bookcorpus_lm
        super(BookCorpusLmLoader, self).__init__(self, dataset, **kwargs)


def merge(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs =  torch.zeros((len(sequences), max(lengths))).long() # gpt tokenizer has pad token id of zero
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

def collate_fn_bookcorpus_lm(data):

    batch = {}
    for key in data[0].keys():
        batch[key] = [d[key] for d in data]
    
    input_ids, input_ids_lengths = merge(batch['input_ids'])
    target_ids, _ = merge(batch['input_ids'])
    segment_ids, _ = merge(batch['input_ids'])

    item['input_ids'] = input_ids
    item['target_ids'] = target_ids
    item['segment_ids'] = segment_ids 

    return item 

class BookCorpusLmDataset(Dataset): 
    '''
    Author:  Sungjun Han

    dataset objective for lm loader 
    data['text'] = list of two 
    '''

    def __init__(self, 
        data,
        tokenizer,
        left_context=True,
        right_context=False,
        max_context_length=128,
        max_target_length=128,
        context_window=1):
        super(BookCorpusLmLoader, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.left_context = left_context
        self.right_context = right_context
        self.max_context_length = max_context_length
        self.max_target_length = max_target_length
        self.context_window = context_window

        #left context and right context cannot be both False 
        assert(left_context or right_context)

    def __len__(self):
        # -1 because we cannot use the first sample
        return len(self.data['text'])-2 
    
    def __getitem__(self, idx):
        '''
        using the huggingface dataset object 
        we will dynamically prepare the samples

        we will create the control from the previous/next context window through random sampling 
        '''
        
        if idx == 0 and self.left_context:
            idx += 1  # we cannot use the first sample because its previous context does not exist 
        
        if idx == len(self.data['text'])-1 and self.right_context:
            idx -= 1 # we cannot use the last sample because its subsequent context does not exist 

        target = self.data['text'][idx] 

        control = []
        #random sample 
        if self.left_context:
            l_idx=random.randint(max(0,idx-self.context_window, idx-1))
            l_context = self.data['text'][l_idx]
            control.append(l_context)
        
        if self.right_context:
            r_idx=random.randint(idx+1, min(len(self.data['text'])-1,idx+self.context_window))
            r_context = self.data['text'][r_idx]
            control.append(r_context)

        input_ids = []
        target_ids = []
        tokens = self.tokenizer.tokenize(control)
		tokens.insert(0, self.tokenizer.cls_token)
		tokens.append(self.tokenizer.sep_token)

		input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
        target_ids.append([-100] * len(input_ids)) # mask out the output tokens 
		segment_ids = [0]*len(input_ids)

        tokens = self.tokenizer.tokenize(target)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids.append(self.tokenizer.eos_token_id)

        input_ids.append(ids)
        target_ids.append(ids)
        segment_ids.append([1]*len(input_ids))

        '''
        implement any truncation if necessary - but pass for now 
        '''

        d = {}
        d['input_ids'] = torch.tensor(input_ids)
        d['target_ids'] = torch.tensor(target_ids)
        d['segment_ids'] = torhc.tensor(segment_ids)
        return d

class LMClassificationDataset(Dataset):
    '''
    Author:  Sungjun Han
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

	def __len__(self):
		if self.max_samples is None:
			return len(self.data['obs1'])
		return self.max_samples

	def __getitem__(self, idx):
		story1 = [self.data['obs1'][idx], self.data['hyp1'][idx], self.data['obs2'][idx]]
		story2 = [self.data['obs1'][idx], self.data['hyp2'][idx], self.data['obs2'][idx]]

        story1_tokens_id, story1_masks, story1_reference = self.process_story(story1)
        story2_tokens_id, story2_masks, story2_reference = self.process_story(story2)

		item = {}
		item['story1_input_ids'] = torch.tensor(story1_tokens_id)
        item['story2_input_ids'] = torch.tensor(story2_tokens_id)
		#item['segment_ids'] = torch.tensor(segment_ids)
		item['story1_masks'] = torch.tensor(story1_masks)
        item['story2_masks'] = torch.tensor(story2_masks)
		item['story1_reference'] = story1_reference
        item['story2_reference'] = story2_reference
		item['label'] = torch.tensor(self.data['label'][idx])
		item['pad_id'] = self.pad_token_id

		return item
    
    def process_story(self, input):

        story = ' '.join(input)
		tokens = self.tokenizer.tokenize(story)
        tokens.insert(0, self.tokenizer.bos_token)
        tokens.append(self.tokenizer.eos_token)
		#if we are working with a transformer encoder 
		tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

		# for now we do not distinguish segments 
        '''
        but one possible way to assign the segments would be to assign segment id =1 for hyp1 and segment id=0 for obs
        '''
        #segment_ids = [0]*len(tokens_id)
		#segment_ids.extend([1]*len(hyp_id))

		masks = [1]*len(tokens_id)
        reference = '[SEP]'.join(input)
        return tokens_id, masks, reference 

def lm_transformer_collate_fn(batch):
    '''
    Author:  Sungjun Han
    '''
	item={}
	for key in batch[0].keys():
		item[key] = [d[key] for d in batch] # [item_dic, item_idc ]

	pad_id = item['pad_id'][0]
	story1_input_ids, story1_input_length = merge(item['story1_input_ids'], pad_id)
    story2_input_ids, story2_input_length = merge(item['story2_input_ids'], pad_id)

	#segment_ids, _ = merge(item['segment_ids'], pad_id)
	story1_masks, _ = merge(item['story1_masks'], pad_id)
    story2_masks, _ = merge(item['story2_masks'], pad_id)
	label = torch.stack(item['label']).float()

	d = {}
	d['input_ids'] = (story1_input_ids, story2_input_ids)
	#d['segment_ids'] = segment_ids
	d['input_lengths'] = (story1_input_length,story2_input_length)
	d['masks'] = (story1_masks, story2_masks)
	d['reference'] = (item['story1_reference'], item['story2_reference'])
	d['label'] = label
	return d

if __name__ == '__main__':
    # dataset = load_dataset("bookcorpus")