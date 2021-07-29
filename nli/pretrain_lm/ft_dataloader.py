'''
1. dataset object for fine-tuning 
2. dataloader for fine-tuning and zero-shot classification 

'''
import json
import numpy as np
import re
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2Tokenizer
from nli.dataloader import merge
from nli.utils import open_tsv_file

class LMClassificationDataset(Dataset):
    '''
    Author:  Sungjun Han

    Desription: Dataset object for ART dataset for GPT-2
        prepares for dual-encoder architecture 

    data_patbh : str
    tokenizer : hugginface tokenizer python object 
    contiguous : bool
    max_samples : bool
    max_target_length : int 
    max_context_length : int 

    '''
    def __init__(self,
                data_path,
                tokenizer,
                contiguous =True, 
                max_samples=None,
                max_context_length=128,
                max_target_length=92):
        self.data = open_tsv_file(data_path, dic=True)
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.max_context_length  = max_context_length
        self.max_target_length = max_target_length
        self.contiguous = contiguous 
        self.target_index = 2 if not contiguous else 1 

    def __len__(self):
        if self.max_samples is None:
            return len(self.data['obs1'])
        return self.max_samples

    def __getitem__(self, idx):
        if self.contiguous:
            story1 = (self.data['obs1'][idx], self.data['hyp1'][idx], self.data['obs2'][idx])
            story2 = (self.data['obs1'][idx], self.data['hyp2'][idx] ,self.data['obs2'][idx])
        else:
            story1 = (self.data['obs1'][idx], self.data['obs2'][idx], self.data['hyp1'][idx])
            story2 = (self.data['obs1'][idx], self.data['obs2'][idx],self.data['hyp2'][idx])
        

        label = self.data['label'][idx]
        # depending on the label we are going to decide to mask the target or not 
        story1_tokens_id, story1_segment_ids, story1_masks, story1_reference, story1_target = self.process_story(story1, (label == 0))
        story2_tokens_id, story2_segment_ids, story2_masks, story2_reference, story2_target= self.process_story(story2, (label == 1))

        item = {}
        item['story1_input_ids'] = torch.tensor(story1_tokens_id)
        item['story2_input_ids'] = torch.tensor(story2_tokens_id)
        item['story1_segment_ids'] = torch.tensor(story1_segment_ids)
        item['story2_segment_ids'] = torch.tensor(story2_segment_ids)
        item['story1_masks'] = torch.tensor(story1_masks)
        item['story2_masks'] = torch.tensor(story2_masks)
        item['story1_reference'] = story1_reference
        item['story2_reference'] = story2_reference
        item['story1_target'] = torch.tensor(story1_target)
        item['story2_target'] = torch.tensor(story2_target)
        item['label'] = torch.tensor(self.data['label'][idx])

        return item

    def process_story(self, input, correct=True):

        segment_ids = []
        tokens = []
        reference = ''
        for i, story in enumerate(input):
            if i == 0:
                story = 'before :  ' + story
            elif i == 1:
                story = ' | after :  ' + story
            else:
                story = ' | hypothesis :  ' + story
            
            toks = self.tokenizer.tokenize(story)

            
            if i == 0:
                toks.insert(0, self.tokenizer.bos_token)

            if i != self.target_index:
                if len(toks) > self.max_context_length: 
                    toks = toks[:self.max_context_length]
            else:
                if len(toks) > self.max_target_length-2:
                    toks = toks[:self.max_target_length-2]
                    
            if i == len(input)-1:
                toks.append(self.tokenizer.eos_token)
                toks.append(self.tokenizer.cls_token)

            tokens.extend(toks)

            if i != self.target_index: # apply the segment id for the hypothesis differently than the observation
                segment_ids.extend(len(toks)*[0])
            else: 
                segment_ids.extend(len(toks)*[1])
            
            reference += story
        
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

        masks = [1]*len(tokens_id)

        if correct:
            target = tokens_id.copy()
            target[-1] = -100 # for the last token is CLS 
        else:
            target = [-100]*len(tokens_id)

        assert(len(tokens_id) == len(segment_ids))
        return tokens_id, segment_ids, masks, reference, target

def lm_transformer_collate_fn(batch):
    '''
    Author:  Sungjun Han

    Description : collating for batches 
    batch : list of dictionaries 
    '''
    item={}
    for key in batch[0].keys():
        item[key] = [d[key] for d in batch] # [item_dic, item_idc ]

    pad_id = torch.tensor(0)
    story1_input_ids, story1_input_length = merge(item['story1_input_ids'], pad_id)
    story2_input_ids, story2_input_length = merge(item['story2_input_ids'], pad_id)

    #segment_ids, _ = merge(item['segment_ids'], pad_id)
    story1_segment_ids, _ = merge(item['story1_segment_ids'], pad_id)
    story2_segment_ids, _ = merge(item['story2_segment_ids'], pad_id)

    story1_masks, _ = merge(item['story1_masks'],pad_id)
    story2_masks, _ = merge(item['story2_masks'],pad_id)

    target_pad_id = torch.tensor(-100)
    story1_target, _ = merge(item['story1_target'],target_pad_id)
    story2_target, _ = merge(item['story2_target'],target_pad_id)

    label = torch.stack(item['label']).long()

    d = {}
    d['input_ids'] = (story1_input_ids, story2_input_ids)
    d['segment_ids'] = (story1_segment_ids, story2_segment_ids)
    d['input_lengths'] = (story1_input_length,story2_input_length)
    d['targets'] = (story1_target, story2_target)
    d['masks'] = (story1_masks, story2_masks)
    d['reference'] = (item['story1_reference'], item['story2_reference'])
    d['label'] = label
    return d

if __name__ == '__main__':
    dd = LMClassificationDataset('data/alphanli/tsv/train.tsv',
    GPT2Tokenizer.from_pretrained('gpt2', cache_dir='../huggingface'))
    import random
    for i in range(10):
        idx = random.randint(0,len(dd))
        print(dd.__getitem__(idx))