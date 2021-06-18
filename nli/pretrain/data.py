'''
will host data loading objects for 

1.mlm objective 
2.lm objective 

'''

import json
import numpy as np
import re

import torch
import torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets import load_dataset # use datasets 

class BookCorpusLmLoader(DataLoader):
    
    def __init__(self,**kwargs):
        
        data = kwargs.pop('data')
        tokenizer = kwargs.pop('tokenizer')
        max_context_length = kwargs.pop('max_context_length')
        max_target_lenfgth = kwargs.pop('max_target_length')
        distributed = kwargs.pop('distributed', False)

        # define dataset here 
        dataset = BookCorpusLmDataset(
            data, 
            tokenizer,
            max_context_length,
            max_target_length)

        # if distributed use Sampler and set shuffle to False 
        if distributed:
            shuffle=kwargs['shuffle']# we need to define shuffle in terms of our sampler
            kwargs['shuffle'] = False  # set shuffle to false for the dataloader as handled by sampler
            sampler=DistributedSampler(dataset=dataset, shuffle=shuffle)
            kwargs['sampler'] = sampler 
        
        kwargs['collate_fn'] = collate_fn_BookCorpusLM
        super(BookCorpusLmLoader, self).__init__(self, dataset, **kwargs)


def merge(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs =  torch.zeros((len(sequences), max(lengths))).long() # gpt tokenizer has pad token id of zero
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

def collate_fn_BookCorpusLM(data):

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
    dataset objective for lm loader 

    data['text'] = list of two 
    '''

    def __init__(self, 
        data,
        tokenizer,
        max_context_length,
        max_target_length):
        super(BookCorpusLmLoader, self).__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data['text'])
    
    def __getitem__(self, idx):
        text = self.data['text'][idx]# assume it is in this form   
        control = text[:-1] # this can be a list or a single element 
        target =text[-1]  

        input_ids = []
        target_ids = []

        #make the control code explictit -- this way we can clearly distingjish between the antecedent/subsequent if we choose to do so 
        if len(text) > 2: 
            control = 'antecedent : ' + control[0] + ' subsequent : ' + control[1:]
        else:
            control = 'antecedent : ' + control

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

if __name__ == '__main__':
    # dataset = load_dataset("bookcorpus")