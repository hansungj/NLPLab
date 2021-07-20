'''
Author:  Sungjun Han
Description:

Fine tuning various gp2 finetuning models here 

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

import nli.utils as utils 
import nli.preprocess as preprocess
import nli.metrics as metrics
from nli.models import *

from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup, AdamW, get_cosine_schedule_with_warmup
from nli.dataloader import prepare_dataloader
from nli.pretrain_lm.ft_dataloader import LMClassificationDataset, lm_transformer_collate_fn
from gpt2train import run_epoch

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

parser.add_argument('--val_data_path', default='data/alphanli/tsv/val_split.tsv', type=str)
parser.add_argument('--test_data_path', default='data/alphanli/tsv/test_split.tsv', type=str)
parser.add_argument('--output_dir', default='checkpoint/gpt2_no_ctg_further_pretrained', type=str)

#general training settings 
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--use_cuda', default=False, type =bool)

#model data settings 
parser.add_argument('--max_context_length', default=128, type=int)
parser.add_argument('--max_target_length', default=92, type=int)
parser.add_argument('--classifier_head_num_layers', default=3, type=int)
parser.add_argument('--classifier_dropout', default=0.1, type=float)
parser.add_argument('--contiguous', default =False, type = bool, help='whether to do obs1, obs2, hyp or obs1, hyp, obs2')

def main(args):

    utils.set_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda:
        print('use_cuda=True')
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    logger.info('CONFIGURATION:')
    logger.info(args)

    output_dir = args.output_dir
    logger.info('OUTPUT DATA PATH:')
    logger.info(output_dir)

    #initialize metric keeper 
    test_stats = metrics.MetricKeeper(args.eval_measure.split(','))
    val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir = '../huggingface') 
    tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})

    #initialize dataloader -- test
    val_dataset = LMClassificationDataset(
                data_path = args.val_data_path,
                tokenizer=tokenizer, 
                max_context_length = args.max_context_length,
                max_target_length = args.max_target_length,
                contiguous = args.contiguous)    
    #initialize dataloader -- val
    test_dataset = LMClassificationDataset(
                data_path = args.test_data_path,
                tokenizer=tokenizer, 
                max_context_length = args.max_context_length,
                max_target_length = args.max_target_length,
                contiguous = args.contiguous)


    _, test_loader, val_loader =prepare_dataloader(
                                        None, 
                                        test_dataset,
                                        val_dataset,
                                        args.batch_size, 
                                        collate_fn = lm_transformer_collate_fn,
                                        shuffle=False, 
                                        drop_last = True, 
                                        num_workers = args.num_workers,
                                        distributed = False)

    model = PretrainedDecoderTransformerDual('gpt2',
                    dropout = args.classifier_dropout, 
                    num_layers = args.classifier_head_num_layers)
    model.model.resize_token_embeddings(len(tokenizer))

    if args.use_cuda:
        model.cuda()

    '''
    load model 
    '''
    load_path =  os.path.join(output_dir, 'checkpoint_gpt2_dual' + '.pt')
    model.load_state_dict(torch.load(load_path))
    model.eval()

    ## dontest rightnow 
    logger.info('Testing...')
    model.eval()

    model, _, __, val_stats, __, val_pred = run_epoch(
            args=args,
            model=model,
            device=device,
            data_loader=val_loader,
            optimizer=None,
            scheduler=None,
            stats=val_stats, 
            desc='val_step', 
            train=False,
            use_cuda=args.use_cuda)

    model, _, __, test_stats, __, test_pred = run_epoch(
            args=args,
            model=model,
            device=device,
            data_loader=test_loader,
            optimizer=None,
            scheduler=None,
            stats=test_stats, 
            desc='test_step', 
            train=False,
            use_cuda=args.use_cuda)

    '''
    SAVE STATS and PREDICTIONS
    '''
	#save prediction 
    # with open(os.path.join(output_dir, 'predictions.txt'),'w') as f:
    #     for p in test_pred:
    #         f.write(str(p) + '\n')

    checkpoint = {
    'val_stats': val_stats.keeper,
    'test_stats': test_stats.keeper,
    'args': args.__dict__
    }
    with open(os.path.join(output_dir, 'test-stats.json'), 'w') as f:
        json.dump(checkpoint, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    #If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
    # make output directory if it does not exist 
    output_dir = os.path.join(args.output_dir, '{}_{}'.format('checkpoint', 'gpt2'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.output_dir = output_dir

    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s: %(asctime)s: %(message)s',
        handlers=[
                logging.FileHandler(os.path.join(args.output_dir,'log-gpu:{}.txt'.format(args.use_cuda))),
                logging.StreamHandler()
        ]
        )
    main(args)
