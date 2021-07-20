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

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory for data/train/val 
parser.add_argument('--train_tsv', default='data/alphanli/tsv/train.tsv', type=str)
parser.add_argument('--val_tsv', default='data/alphanli/tsv/val_split.tsv', type=str)
parser.add_argument('--test_tsv', default='data/alphanli/tsv/test_split.tsv', type=str)

parser.add_argument('--output_dir', default='checkpoint', type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--from_pretrained', default=None, type=str)

#general training settings 
parser.add_argument('--model_type', default='double-head', type=str)
parser.add_argument('--label_mark', default=False, type=bool, help='only apply lm auxilliary loss on the correct')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default =100, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate_during_training', default=False, type=bool, help='Decide to evaluate on validation set')
parser.add_argument('--evaluate_training_during_training', default=False, type=bool, help='track training by evaluating periodically')
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--n_gpu', default=1, type=int)

#model data settings 
parser.add_argument('--max_context_length', default=128, type=int)
parser.add_argument('--max_target_length', default=92, type=int)
parser.add_argument('--classifier_head_num_layers', default=3, type=int)
parser.add_argument('--classifier_dropout', default=0.1, type=float)
parser.add_argument('--contiguous', default =False, type = bool, help='whether to do obs1, obs2, hyp or obs1, hyp, obs2')

#deep learning models 
parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--tokenizer', default='regular', help='choose tokenizer: regular/bpe - for baseline model')
parser.add_argument('--optimizer', default='adam', help='adam/adamW/sgd/..')
parser.add_argument('--beta_1', default=0.9, type=float, help='beta1 for first moment')
parser.add_argument('--beta_2', default=0.99, type=float, help='beta2 for second moment')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--eps', default=1e-6, type=float)
parser.add_argument('--scheduler', default='linear', type =str, help='linear/cosine')
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
    tokenizer.add_special_tokens({'cls_token': '[CLS]', 'sep_token': '[SEP]'})

    #initialize dataloader -- train
    train_dataset = LMClassificationDataset(
                data_path = args.train_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch,
                max_context_length = args.max_context_length,
                max_target_length = args.max_target_length,
                contiguous = args.contiguous)
    test_dataset = LMClassificationDataset(
                data_path = args.test_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch,
                max_context_length = args.max_context_length,
                max_target_length = args.max_target_length,
                contiguous = args.contiguous)

    #initialize val-dataloader
    val_dataset = None
    if args.evaluate_during_training:
        val_dataset = LMClassificationDataset(
                data_path = args.val_tsv,
                tokenizer=tokenizer, 
                max_samples=args.max_samples_per_epoch,
                max_context_length = args.max_context_length,
                max_target_length = args.max_target_length,
                contiguous = args.contiguous)

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

    model_name = args.from_pretrained if args.from_pretrained else 'gpt2'
    logger.info('Loading from {}'.format(model_name))

    if args.model_type == 'double-head':
        model = PretrainedDecoderTransformerDual(model_name,
                        dropout = args.classifier_dropout, 
                        num_layers = args.classifier_head_num_layers)
    
    elif args.model_type == 'single-head':
        model = PretrainedDecoderTransformerDualSingleClassifier(model_name,
                dropout = args.classifier_dropout, 
                num_layers = args.classifier_head_num_layers)

    model.model.resize_token_embeddings(len(tokenizer))

    # load pretrained model here 


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
    TRAIN and Test
    '''
    save_path =  os.path.join(output_dir, 'checkpoint_gpt2_dual' + '.pt')
    val_accuracy = 0.0  
    earlyStop = 0
    for epoch in tqdm(range(args.num_epochs), desc='epoch'):
        model.train()
        model, optimizer, scheduler, stats, _, __ = run_epoch(
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
            model.eval()
            model, _, __, val_stats, __, ___ = run_epoch(
                args=args,
                model=model,
                device=device,
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
        
            #early stopping
            if args.early_stopping:
                current_accuracy = val_stats.keeper['accuracy'][-1]
                if current_accuracy > val_accuracy:
                    logger.info('Early stopping patience - resetting')
                    earlyStop = 0
                    if args.local_rank == 0: # save  only with the first 
                        save_model(model, save_path)
                    val_accuracy = current_accuracy
                    continue

                earlyStop += 1
                if args.early_stopping == earlyStop:
                    logger.info('Early stopping criterion met - terminating')
                    if args.local_rank == 0: # save  only with the first 
                        save_model(model, save_path)
                    break 

                logger.info('Early stopping patience {}'.format(earlyStop))
            else: # then just save the latest model 
                if args.local_rank == 0: # save  only with the first
                    save_model(model, save_path)
                    
    # load and test 
    logger.info('Testing...')
    load_path =  os.path.join(output_dir, 'checkpoint_gpt2_dual' + '.pt')
    model.load_state_dict(torch.load(load_path))
    model.eval()
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
            use_cuda=True)

    #print for status update
    logger.info('\nTest stats:')
    test_stats.print()

    '''
    SAVE STATS and PREDICTIONS
    '''
	#save prediction 
    # with open(os.path.join(output_dir, 'predictions.txt'),'w') as f:
    #     for p in test_pred:
    #         f.write(str(p) + '\n')

    if args.local_rank == 0:
        checkpoint = {
        'train_stats': stats.keeper,
        'val_stats': val_stats.keeper if args.evaluate_during_training else None,
        'test_stats': test_stats.keeper,
        'args': args.__dict__
        }
        with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
            json.dump(checkpoint, f, indent=4)

def save_model(model,path):
    model_to_save = model.module if hasattr(model,'module') else model 
    torch.save(model_to_save.state_dict(), path )

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
    total_loss = 0.
    total_lm_loss = 0.

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
            logits, loss, lm_loss = model(**inputs)
            
        else:
            with torch.no_grad():
                logits, loss, lm_loss = model(**inputs)
            
        if train:
            joint_loss = loss + 0.25*lm_loss
            joint_loss.backward()
            '''
            implement gradient norm clip 
            '''
            if args.grad_norm_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

            if args.evaluate_training_during_training and step % int(len(data_loader)/30) == 0 and step != 0:
                print('\nAt Step={}, we evaluate on train for the labels, predictions collected so far'.format(step))
                temp_stats = metrics.MetricKeeper(['accuracy'])
                temp_stats.eval(labels,pred)
                logger.info(temp_stats.keeper)

        #keep things 
        total_loss += loss.mean().item()
        total_lm_loss += lm_loss.mean().item()
        labels.extend(label.tolist())
        pred.extend(torch.argmax(logits, dim=-1).tolist())
        logger.info('{}: At step {}, class_loss = {}, lm_loss = {} '.format(desc, step, loss.mean().item(), lm_loss.mean().item()))

    #update keepr for log liklihood
    stats.update('classification_loglikelihood',total_loss / len(data_loader))
    stats.update('lm_loglikelihood',total_lm_loss / len(data_loader))
    stats.eval(labels,pred)
    return model, optimizer, scheduler, stats, labels, pred


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
