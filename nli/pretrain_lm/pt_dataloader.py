'''
Author:  Sungjun Han
contains data loader / dataset objects for the pre-training and fine-tuning for GPT-2

1. dataloader for pretraining 
2. dataset object for BookCorpus - for pretraining  
'''

import json
import numpy as np
import re
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset # use datasets 
from nli.dataloader import merge
from nli.utils import open_tsv_file

class BookCorpusLmLoader(DataLoader):
    '''
    Author:  Sungjun Han

    Description : Pretraining dataloader for GPT2 using BookCorpus
    data : huggingface dataset python object 
    kwargs : dictionary 
    '''

    def __init__(self, data, **kwargs):
        
        tokenizer = kwargs.pop('tokenizer')
        max_context_length = kwargs.pop('max_context_length')
        max_target_length = kwargs.pop('max_target_length')
        left_context = kwargs.pop('left_context', True)
        right_context = kwargs.pop('right_context', False )
        context_window = kwargs.pop('context_window', 1)
        random_negative_sample = kwargs.pop('random_negative_sample', 0.)

        # define dataset here 
        dataset = BookCorpusLmDataset(
            data, 
            tokenizer,
            left_context,
            right_context,
            max_context_length,
            max_target_length,
            context_window,
            random_negative_sample)

        # if distributed use Sampler and set shuffle to False 
        distributed = kwargs.pop('distributed', False)
        if distributed:
            shuffle=kwargs['shuffle']# we need to define shuffle in terms of our sampler
            kwargs['shuffle'] = False  # set shuffle to false for the dataloader as handled by sampler
            sampler=DistributedSampler(dataset=dataset, shuffle=shuffle)
            kwargs['sampler'] = sampler 
        
        kwargs['collate_fn'] = collate_fn_bookcorpus_lm
        super().__init__(dataset, **kwargs)

def collate_fn_bookcorpus_lm(data):
    '''
    Description : collates from the map-style pytorch dataset object into batches 
    data : list of dictionaries 
    '''
    batch = {}
    for key in data[0].keys():
        batch[key] = [d[key] for d in data]

    pad_id = torch.tensor(0)
    target_pad_id = torch.tensor(-100)
    input_ids, lengths = merge(batch['input_ids'], pad_id)
    target_ids, _ = merge(batch['target_ids'], target_pad_id)
    segment_ids, _ = merge(batch['segment_ids'], pad_id)
    masks, _  = merge(batch['attention_masks'], pad_id)
    labels = torch.stack(batch['labels'])

    item = {}
    item['input_ids'] = input_ids
    item['target_ids'] = target_ids
    item['segment_ids'] = segment_ids 
    item['attention_masks']= masks
    item['reference'] = batch['reference']
    item['labels'] = labels
    item['lengths'] = lengths
    return item 

class BookCorpusLmDataset(Dataset): 
    '''
    Author:  Sungjun Han

    Desription : Pretraining dataset object for GPT2 using BookCorpus
    data : huggingface dataset python object
    tokenizer : hugginface tokenizer python object 
    left_context : bool 
    right_context : bool
    max_context_length : bool 
    max_target_length : int 
    context_window : int 
    random_negative_sample  : float  < 1 
    '''

    def __init__(self, 
        data,
        tokenizer,
        left_context=True,
        right_context=False,
        max_context_length=128,
        max_target_length=128,
        context_window=1,
        random_negative_sample = 0.):
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.left_context = left_context
        self.right_context = right_context
        self.max_context_length = max_context_length
        self.max_target_length = max_target_length
        self.context_window = context_window
        self.random_negative_sample = random_negative_sample

        #left context and right context cannot be both False 
        assert(left_context or right_context)

    def __len__(self):
        # -1 because we cannot use the first sample
        return len(self.data)-2 

    def __getitem__(self, idx):
        '''
        Descrtiption:
            using the huggingface dataset object 
            we will dynamically prepare the samples

            we will create the control from the previous/next context window through random sampling 

            here the control codes are
            "before : " - for the left context  
            "after : " - for the right context 
            "hypothesis : " - for the hypothesis 
        
        idx : int 
        '''
        
        if idx == 0 and self.left_context:
            idx += 1  # we cannot use the first sample because its previous context does not exist 
        
        if idx == len(self.data)-1 and self.right_context:
            idx -= 1 # we cannot use the last sample because its subsequent context does not exist 

        #random negative sample 
        u = random.random()
        if  u < self.random_negative_sample:
            # randomly sample to create a negative sample
            rand_idx = random.randint(0, len(self.data)-1)
            target = self.data[rand_idx] 
            label = 1 # negative 
        else:
            target = self.data[idx] 
            label = 0 # positive 

        control = []
        #random sample 
        if self.left_context:
            l_idx=random.randint(max(0,idx-self.context_window),idx-1)
            l_context = 'before : ' + self.data[l_idx]
            #l_context= self.data[l_idx]
            control.append(l_context)
        
        if self.right_context:
            r_idx=random.randint(idx+1, min(len(self.data)-1,idx+self.context_window))
            r_context = 'after : ' + self.data[r_idx]
            #r_context= self.data[r_idx]
            control.append(r_context)

        segment_ids = []
        tokens_all = []
        for i, text in enumerate(control):

            if i != 0 : # add separation 
                text = ' | ' + text

            tokens = self.tokenizer.tokenize(text)
            if i == 0:
                tokens.insert(0, self.tokenizer.bos_token)

            # here limit context length 
            if len(tokens) > self.max_context_length:
                tokens=tokens[:self.max_context_length]

            segment_ids.extend([0]*len(tokens))

            tokens_all.extend(tokens)
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens_all)
        target_ids = [-100] * len(input_ids) # mask out the output tokens 
             

        #add control code - because this should be masked as well 
        target_control_code = ' | hypothesis : '
        target_control_code = self.tokenizer.tokenize(target_control_code)
        target_ids.extend([-100]*len(target_control_code))
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(target_control_code))
    
        tokens = self.tokenizer.tokenize(target)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(ids) > self.max_target_length-1:
            ids=ids[:self.max_target_length-1]

        ids.append(self.tokenizer.eos_token_id)
        ids.append(self.tokenizer.cls_token_id)
        input_ids.extend(ids)

        #if negative sampling then do not calcaulte the loss 
        if  u < self.random_negative_sample:
            target_ids.extend([-100]*len(ids))
        else:
            ids[-1] = -100 # dont predict the cls token
            target_ids.extend(ids)

        segment_ids.extend([1]*(len(ids) + len(target_control_code)))

        masks = [1]*len(input_ids)
        d = {}
        d['input_ids'] = torch.tensor(input_ids)
        d['target_ids'] = torch.tensor(target_ids)
        d['segment_ids'] = torch.tensor(segment_ids)
        d['attention_masks'] = torch.tensor(masks)
        d['reference'] = ' | '.join(control) + ' $ ' + target 
        d['labels'] = torch.tensor(label).float()

        return d

if __name__ == '__main__':
    pass