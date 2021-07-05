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

from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from nli.dataloader import prepare_dataloader
from nli.pretrainlm.ft_dataloader import LMClassificationDataset, lm_transformer_collate_fn

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory for data/train/val 
parser.add_argument('--train_tsv', default='data/alphanli/tsv/train.tsv', type=str)
parser.add_argument('--val_tsv', default='data/alphanli/tsv/val_split.tsv', type=str)
parser.add_argument('--test_tsv', default='data/alphanli/tsv/test_split.tsv', type=str)

parser.add_argument('--output_dir', default='checkpoint', type=str)
parser.add_argument('--local_rank', default=0, type=int)
#general training settings 
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default =100, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate_during_training', default=False, type=bool, help='Decide to evaluate on validation set')
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--n_gpu', default=1, type=int)


#deep learning models 
parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--tokenizer', default='regular', help='choose tokenizer: regular/bpe - for baseline model')
parser.add_argument('--optimizer', default='adam', help='adam/adamW/sgd/..')
parser.add_argument('--beta_1', default=0.99, type=float, help='beta1 for first moment')
parser.add_argument('--beta_2', default=0.999, type=float, help='beta2 for second moment')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--scheduler', default=None, type =bool, help='')
parser.add_argument('--num_warming_steps', default=0.1, type=float, help='number of warming steps for the scheduler - between 0 and 1')
parser.add_argument('--dropout', default=0.5, type=float, help='')
parser.add_argument('--grad_norm_clip', default=1, type=float, help='clip the norm')
parser.add_argument('--grad_accumulation_steps', default=None, type=int, help='number of steps to accumulate gradient')
parser.add_argument('--early_stopping', default=10,  type=int, help='patience for early stopping - if 0 no early stopping used')

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
    logger.info('TRAIN DATA PATH:')
    logger.info(args.train_tsv)
    logger.info('DEV DATA PATH:')
    logger.info(args.val_tsv)
    logger.info('TEST DATA PATH:')
    logger.info(args.test_tsv)

    output_dir = args.output_dir
    logger.info('OUTPUT DATA PATH:')
    logger.info(output_dir)

    #initialize metric keeper 
    stats = metrics.MetricKeeper(args.eval_measure.split(','))
    test_stats = metrics.MetricKeeper(args.eval_measure.split(','))
    if args.evaluate_during_training:
        val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir = '../huggingface') 
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})

    #initialize dataloader -- train
    train_dataset = LMClassificationDataset(
                data_path = args.train_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch)
    test_dataset = LMClassificationDataset(
                data_path = args.test_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch)

    #initialize val-dataloader
    val_dataset = None
    if args.evaluate_during_training:
        val_dataset = LMClassificationDataset(
                data_path = args.val_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch)

    train_loader, test_loader, val_loader =prepare_dataloader(
                                        train_dataset, 
                                        test_dataset,
                                        val_dataset,
                                        args.batch_size, 
                                        collate_fn = lm_transformer_collate_fn,
                                        shuffle=args.shuffle, 
                                        drop_last = True, 
                                        num_workers = args.num_workers,
                                        distributed = (args.n_gpu >1))

    model = PretrainedDecoderTransformerDual('gpt2')

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
        parameters = utils.prepare_model_parameters_weight_decay(model.named_parameters())
    else:
        parameters = model.parameters()

    #optimizer 
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.learning_rate, (args.beta_1,args.beta_2), args.eps)

    #scheduler
    scheduler = None 
    if args.scheduler:
        num_training_steps = int((len(train_loader)//args.batch_size)*args.num_epochs)
        num_warmup_steps = int(num_training_steps*args.num_warming_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,num_training_steps)
            

    '''
    TRAIN and Test
    '''

    val_accuracy = 0.0  
    earlyStop = 0
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
                device=device,
                data_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                stats=val_stats, 
                desc='val_step', 
                train=False,
                use_cuda=True)
            
            #print for status update
            logger.info('\nVal stats:')
            val_stats.print()
        
            #early stopping
            if args.early_stopping:
                current_accuracy = val_stats.keeper['accuracy'][-1]
                if current_accuracy > val_accuracy:
                    earlyStop = 0
                    if args.local_rank == 0: # save  only with the first 
                        torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type + '.pt'))
                    val_accuracy = current_accuracy
                    continue

                earlyStop += 1
                if args.early_stopping == earlyStop:
                    logger.info('Early stopping criterion met - terminating')
                    if args.local_rank == 0: # save  only with the first 
                        torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type + '.pt') )
                    break 

                logger.info('Early stopping patience {}'.format(earlyStop))
        
    logger.info('Testing...')
    _, test_pred = run_epoch(
            args=args,
            model=model,
            devic=device,
            data_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            stats=test_stats, 
            desc='test_step', 
            train=False,
            use_cuda=True)

    #print for status update
    logger.info('\nTest stats:')
    test_stats.print()

    '''
    SAVE STATS and PREDICTIONS
    '''
	#save prediction 
    with open(os.path.join(output_dir, 'predictions.txt'),'w') as f:
        for p in test_pred:
            f.write(str(p) + '\n')

    checkpoint = {
    'stats': stats.keeper,
    'val_stats': val_stats.keeper if args.evaluate else None,
    'test_stats': test_stats.keeper,
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
    for step, d in enumerate(tqdm(data_loader, desc=desc)):
        input1, input2 = d['input_ids']
        seg1, seg2 = d['segment_ids']
        mask1, mask2 = d['masks']
        length1, length2 = d['input_lengths']
        label = d['label']

        if use_cuda:
            input1 = input1.to(device)
            input2 = input2.to(device)
            seg1 = seg1.to(device)
            seg2 = seg2.to(device)
            mask1 = mask1.to(device)
            mask2 = mask2.to(device)
            length1 = length1.to(device)
            length2 = length2.to(device)
            label = label.to(device)

        inputs = {
        'input1': input1,
        'input2': input2,
        'length1': length1,
        'length2': length2, 
        'masks1': mask1,
        'masks2': mask2 ,
        'segment_ids1': seg1,
        'segment_ids2': seg2,
        'labels':label
        }
        if train:
            logits, loss = model(**inputs)
        else:
            with torch.no_grad():
                logits, loss = model(**inputs)
            

        if train:
            loss.backward()
            '''
            implement gradient norm clip 
            '''
            if args.grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        #keep things 
        total_loss += loss.mean().item()
        labels.extend(label.tolist())
        pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())
    
        logger.info('{}: At step {}, loss = {}'.format(desc, step, loss.mean().item()))

    #update keepr for log liklihood
    stats.update('loglikelihood',total_loss / len(data_loader))
    stats.eval(labels,pred)
    return labels, pred


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
