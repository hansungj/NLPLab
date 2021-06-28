# utils.py
'''
Author: Sungjun Han
Description: Various utilty functions 
'''
import argparse 
import json 
from collections import defaultdict 

import random
import numpy as np
import torch

def open_label_file(path):
	if path:
		with open(path, 'r') as f:
			labels = [int(l) for l in f.read().splitlines()]
		return labels 
	raise ValueError('Must give label path')

def open_json_file(path):
	if path:
		with open(path, 'r') as f:
		 	data = json.load(path)
		return data 
	raise ValueError('Must give json data path')

def open_tsv_file(path, dic=False):
	if path:
		with open(path, 'r') as f:
			data = [l.split('\t') for l in f.read().splitlines()]
		if dic:
			res = defaultdict(list)
			for line in data:
				res['story_id'].append(line[0])
				res['obs1'].append(line[1])
				res['obs2'].append(line[2])
				res['hyp1'].append(line[3])
				res['hyp2'].append(line[4])
				res['label'].append(0 if int(line[5])==1 else 1)
			return res
		return data
	raise ValueError('Must give tsv data path')

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)