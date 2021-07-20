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
        
        # here randomly shuffle the two stories 
        # stories = [story1, story2]
        # random.shuffle(stories)
        # story1, story2 = stories 

        story1_tokens_id, story1_segment_ids, story1_masks, story1_reference = self.process_story(story1)
        story2_tokens_id, story2_segment_ids, story2_masks, story2_reference = self.process_story(story2)

        item = {}
        item['story1_input_ids'] = torch.tensor(story1_tokens_id)
        item['story2_input_ids'] = torch.tensor(story2_tokens_id)
        item['story1_segment_ids'] = torch.tensor(story1_segment_ids)
        item['story2_segment_ids'] = torch.tensor(story2_segment_ids)
        item['story1_masks'] = torch.tensor(story1_masks)
        item['story2_masks'] = torch.tensor(story2_masks)
        item['story1_reference'] = story1_reference
        item['story2_reference'] = story2_reference
        item['label'] = torch.tensor(self.data['label'][idx])

        return item

    def process_story(self, input):

        segment_ids = []
        tokens = []
        reference = ''
        for i, story in enumerate(input):
            # if i != len(input)-1:
            #     control_code = "observation {} : ".format(i+1)
            # else:
            #     control_code = "hypothesis : "

            if i !=0:
                story = ' $ ' + story
            toks = self.tokenizer.tokenize(story)

            # if i != 0:
            #     toks.insert(0, self.tokenizer.sep_token)
            
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

            if i % 2 == 0:
                segment_ids.extend(len(toks)*[0])
            else: 
                segment_ids.extend(len(toks)*[1])
            
            reference += story
        
        tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

        masks = [1]*len(tokens_id)

        assert(len(tokens_id) == len(segment_ids))
        return tokens_id, segment_ids, masks, reference 

def lm_transformer_collate_fn(batch):
    '''
    Author:  Sungjun Han
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
    label = torch.stack(item['label']).long()

    d = {}
    d['input_ids'] = (story1_input_ids, story2_input_ids)
    d['segment_ids'] = (story1_segment_ids, story2_segment_ids)
    d['input_lengths'] = (story1_input_length,story2_input_length)
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