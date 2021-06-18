'''
for pretraining GPT-2 or other language models (transformer decoder archiecture )
'''

import logging
import os
import numpy as np
import argparse
import pickle
import time
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import transformers 

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)


argparse.add_argument('--corpus_dir', default=None, type=str)

#training 
argparse.add_argument('--shuffle', default=none, type=str)

#optimizer 
argparse.add_argument('--optimizer', default='adam', type =str)
argparse.add_argument('--learning_rate', default='1e-5', type=float)
argparse.add_argument('--beta1', default=0.9, type=float)
argparse.add_argument('--beta2', default=0.99, type=float)
argparse.add_argument('--weight_decay', default=0.01, type=float)
argparse.add_argument('--learning_rate_scheduler', default='linear_warm_up', type=str)

def main(args):
    '''
    load and train using masked modelling next sentence objective 
    '''

    '''
    TO DO
    1. define dataloader - make it usable with distributed training 
    2. define optimizer and model and tokenizer
    2. write training loop and implement validation and early stopping 

    '''


if __name__ == '__main__':
    args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)