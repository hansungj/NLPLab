'''
Author: Sungjun Han
Description: for pretraining GPT-2 or other language models (transformer decoder archiecture )
'''
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

from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, GPT2LMHeadModel, AdamW
from nli.pretrain_lm.pt_dataloader import BookCorpusLmLoader
from datasets import load_dataset

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory for data/train/val 

parser.add_argument('--output_dir', default='pretrain-checkpoint-new', type=str)
parser.add_argument('--local_rank', default=0, type=int)
#general training settings 
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default =100, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate_during_training', default=False, type=bool, help='Decide to evaluate on validation set')
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--n_gpu', default=1, type=int)

#deep learning models 
parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--tokenizer', default='regular', help='choose tokenizer: regular/bpe - for baseline model')
parser.add_argument('--optimizer', default='adam', help='adam/adamW/sgd/..')
parser.add_argument('--beta_1', default=0.9, type=float, help='beta1 for first moment')
parser.add_argument('--beta_2', default=0.99, type=float, help='beta2 for second moment')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--scheduler', default='cosine', type =str, help='')
parser.add_argument('--num_warming_steps', default=0.1, type=float, help='number of warming steps for the scheduler - between 0 and 1')
parser.add_argument('--dropout', default=0.5, type=float, help='')
parser.add_argument('--grad_norm_clip', default=1, type=float, help='clip the norm')
parser.add_argument('--grad_accumulation_steps', default=1, type=int, help='number of steps to accumulate gradient')

# pretraining hyperparmeters 
parser.add_argument('--left_context', default=True, type=bool)
parser.add_argument('--right_context', default=True, type=bool)
parser.add_argument('--max_context_length', default=128, type=int)
parser.add_argument('--max_target_length', default=92, type=int)
parser.add_argument('--context_window', default=1, type=int)

def main(args):

    utils.set_seed(args.seed)

    if args.n_gpu > 1:
        #initializae to synchronize gpus
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device('cuda', args.local_rank)
    else:
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
    stats = metrics.MetricKeeper()
    if args.evaluate_during_training:
        val_stats = metrics.MetricKeeper()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir = '../huggingface') 
    '''
    DEFINE DATA 
    '''
    #initialize dataloader -- train
    dataset_kwargs = {
        'tokenizer':tokenizer,
        'left_context':args.left_context,
        'right_context':args.right_context,
        'max_context_length':args.max_context_length,
        'max_target_length':args.max_target_length,
        'context_window':args.context_window,
        'shuffle':args.shuffle,
        'distributed': (args.n_gpu > 1),
        'num_workers': args.num_workers,
        'batch_size': args.batch_size
    }
    logger.info('loading train bookcorpus :TRAIN')
    train_data = load_dataset("bookcorpus", cache_dir = '../huggingface')['train']['text']
    #train_data = load_dataset("bookcorpus", split = 'train[:95%]', cache_dir = '../huggingface/bookcorpus') 
    train_loader = BookCorpusLmLoader(train_data, **dataset_kwargs)    

    if args.evaluate_during_training:
        logger.info('loading train bookcorpus : taking last 5% as VAL, we will keep track of perplexity')
        val_data = load_dataset("bookcorpus", split = 'train[95%:]', cache_dir = '../huggingface')
        val_loader = BookCorpusLmLoader(val_data, **kwargs)

    model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir = '../huggingface')

    if args.n_gpu > 1:
        model.cuda(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters=True)

    elif args.use_cuda:
        model.cuda()

    '''
    DEFINE OPTIMIZER, SCHEDULER
    '''
    #group parmaeters if we are weight decaying
    if args.weight_decay:
        parameters = utils.prepare_model_parameters_weight_decay(model.named_parameters(), args.weight_decay)
    else:
        parameters = model.parameters()

    #optimizer 
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(parameters, args.learning_rate, (args.beta_1,args.beta_2), args.eps)

    #scheduler
    scheduler = None 
    if args.scheduler:
        num_training_steps = int((len(train_loader)//args.batch_size)*args.num_epochs)
        num_warmup_steps = int(num_training_steps*args.num_warming_steps)
        if args.scheduler ==  'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        elif args.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
            
    '''
    TRAIN
    '''
    for epoch in tqdm(range(args.num_epochs), desc='epoch'):
        run_epoch(
            args=args,
            model=model,
            device =device,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            stats=stats, 
            desc='train_step', 
            train=True,
            use_cuda=True)

        #print for status update
        logger.info('\nTrain stats:')
        stats.print()

        if args.evaluate_during_training:
            run_epoch(
            args=args,
            model=model,
            device =device,
            data_loader=val_loader,
            optimizer=None,
            scheduler=None,
            stats=val_stats, 
            desc='val_step', 
            train=False,
            use_cuda=True)
            #print for status update
            logger.info('\nVal stats:')
            val_stats.print()

        if args.local_rank == 0:
            model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(os.path.join(output_dir, 'gpt2-epoch-{}'.format(epoch)))
    '''
    SAVE STATS 
    '''
    if args.local_rank == 0:
        checkpoint = {
        'stats': stats.keeper,
        'args': args.__dict__
        }
        with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
            json.dump(checkpoint, f, indent=4)

def run_epoch(
            args,
            model,
            device,
            data_loader,
            optimizer,
            scheduler,
            stats, 
            desc='train_step', 
            train=True,
            use_cuda=True,
            ):
            
    labels = []
    pred = []
    total_loss = 0
    if train:
        model.train()
    else:
        model.eval()

    model.zero_grad()
    grad_acc_steps = 0 
    for step, batch in enumerate(tqdm(data_loader, desc=desc)):
        input_ids = batch['input_ids']
        attention_masks = batch['attention_masks']
        segment_ids = batch['segment_ids']
        target_ids = batch['target_ids']


        if use_cuda:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            segment_ids = segment_ids.to(device)
            attention_masks = attention_masks.to(device)

        if train:
            output = model(
                    input_ids = input_ids, 
                    attention_mask = attention_masks, 
                    token_type_ids = segment_ids,
                    labels = target_ids )
            loss = output.loss 
        else:
            with torch.no_grad():
                output = model(
                    input_ids = input_ids, 
                    attention_mask = attention_masks, 
                    token_type_ids = segment_ids,
                    labels = target_ids )
                loss = output.loss

        if train:
            loss.backward()
            
            if args.grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

            grad_acc_steps += 1
            if args.grad_accumulation_steps == grad_acc_steps:
                optimizer.step()
                grad_acc_steps = 0

            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.mean().item()
        logger.info('{}: At step {}, loss = {}'.format(desc, step, loss.mean().item()))

        if step % 100000 == 0 and  step != 0:
            model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(os.path.join(output_dir, 'gpt2-step-{}'.format(step)))

    #update keepr for log liklihood
    stats.update('loglikelihood',total_loss / len(data_loader))
    stats.update('perplexity', np.exp(total_loss / len(data_loader)))

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