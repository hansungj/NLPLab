'''
Author:  Sungjun Han, Anastasiia 
Description:
Main train method of advanced models for aNLI task (also evaluates during training) 
Pretrained transformers:
1 Pretrained Transformer CLS: BERT + Head: CLS classifier (see more in nli/models/Transformers)
2 Pretrained Transformer Pooling: BERT/GPT2 + Mean Pooling + Head: Classfier (see more in nli/models/Transformers)
3 Transformer Based Dual Encoders (Siamese models):
	3.1 with a single Head: CLS classifier (for GPT2 based see nli/models/GPT2; for BERT based see nli/models/bert_dual_encoder)
	3.2 with two Heads: CLS classifier (for GPT2 based see nli/models/GPT2)
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
from nli.dataloader import  AlphaDatasetTransformer, AlphaDatasetDualEncoder, prepare_dataloader, alpha_collate_fn_transformer, load_dataloader_BBDualEnc
import nli.preprocess as preprocess
import nli.metrics as metrics
from nli.tokenization import WhiteSpaceTokenizer
from nli.embedding import build_embedding_glove
from nli.models import *
from nli.models.bert_dual_encoder import BB_DualEncoder

from transformers import BertTokenizer, GPT2Tokenizer, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory 
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--vocab', default='data/vocab.json', type=str)

#directory for data/train/val 
parser.add_argument('--train_tsv', default='data/alphanli/tsv/train.tsv', type=str)
parser.add_argument('--val_tsv', default='data/alphanli/tsv/val_split.tsv', type=str)
parser.add_argument('--test_tsv', default='data/alphanli/tsv/test_split.tsv', type=str)

#directory for output
parser.add_argument('--annot_pred', default='annot_pred.lst', type=str)
parser.add_argument('--annot_label', default='annot_label.lst', type=str)
parser.add_argument('--output_dir', default='checkpoint', type=str)
parser.add_argument('--output_name', default='', type=str)

#general training settings 
parser.add_argument('--model_type', default='pretrained-transformers-cls', type=str, help='choose one of the transformer based models')
parser.add_argument('--batch_size', default=128, type=int, help='set the batch size')
parser.add_argument('--num_workers', default=0, type=int, help='set the number of workers')
parser.add_argument('--shuffle', default=True, type=bool, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--num_epochs', default =100, type=int, help = 'set the number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='set the number of samples per epoch')
parser.add_argument('--evaluate', default=False, type=bool, help='set True to evaluate on validation set')
parser.add_argument('--eval_measure', default = 'accuracy', help='set theevaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--n_gpu', default=1, type=int, help="set the number of gpu's")

#deep learning models 
parser.add_argument('--use_cuda', default=False, type=bool, help = 'set True activate to use cuda')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--tokenizer', default='regular', help='choose tokenizer: regular/bpe - for baseline model')
parser.add_argument('--optimizer', default='adam', help='adam/adamW/sgd/..')
parser.add_argument('--beta_1', default=0.99, type=float, help='beta1 for first moment')
parser.add_argument('--beta_2', default=0.999, type=float, help='beta2 for second moment')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--eps', default=1e-8, type=float)
parser.add_argument('--scheduler', default=None, type = bool, help='')
parser.add_argument('--num_warming_steps', default=0.1, type=float, help='number of warming steps for the scheduler - between 0 and 1')
parser.add_argument('--dropout', default=0.5, type=float, help='')
parser.add_argument('--grad_norm_clip', default=1, type=float, help='clip the norm')
parser.add_argument('--grad_accumulation_steps', default=None, type=int, help='number of steps to accumulate gradient')
parser.add_argument('--early_stopping', default=10,  type=int, help='patience for early stopping - if 0 no early stopping used')

#model -static embedding  
parser.add_argument('--glove_model', default='glove-wiki-gigaword-50', type=str, help='choose from fasttext-wiki-news-subwords-300, conceptnet-numberbatch-17-06-300, word2vec-ruscorpora-300, word2vec-google-news-300, glove-wiki-gigaword-50, glove-wiki-gigaword-100, glove-wiki-gigaword-200, glove-wiki-gigaword-300, glove-twitter-25, glove-twitter-50, glove-twitter-100, glove-twitter-200') 
parser.add_argument('--freeze_embedding', default=False, type=bool, help='freezes the glvoe pretrained embedding')
parser.add_argument('--se_hidden_encoder_size', default=200, type=int, help='hidden size for the encoder')
parser.add_argument('--se_hidden_decoder_size', default=200, type=int, help='hidden size for the decoder')
parser.add_argument('--se_num_encoder_layers', default=2, type=int, help='number of encoder layers ')
parser.add_argument('--se_num_decoder_layers', default=2, type=int, help='number of decoder layers ')

#model - static embedding Mixture specific
parser.add_argument('--sem_pooling', default='sum', help='choose from sum/product/max -- used to pool the vectors from premise/hyp1/hyp2')

#model -static embedding RNN specific
parser.add_argument('--sernn_bidirectional', default=False, type=bool, help='bidirectional for encoder or not') 

#Pretrained transformer models 
parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str, help='can be used to initialize a pretrained model from huggingface')
#directory for data/train/val


def transformer_initialize_model(args, vocab_size = None):
	'''
	Author: Sungjun Han
	Description: initializes transformer models
	'''
		#load models
	if args.model_type == 'pretrained-transformers-cls':
		if 'gpt' in args.pretrained_name:
			raise ValueError('we cannot use CLS classifier with GPT2')

		model = PretrainedTransformerCLS(args.pretrained_name)
		
	elif args.model_type == 'pretrained-transformers-pooling':

		model = PretrainedTransformerPooling(args.pretrained_name) # mean pooling 

	elif args.model_type == 'pretrained-transformers-decoder':
		if 'gpt' not in args.pretrained_name:
			raise ValueError('for now we only support gpt model')
			
		model = PretrainedDecoderTransformerCLS(args.pretrained_name)

	elif args.model_type == 'pretrained-transformers-decoder-dual':
		if 'gpt' not in args.pretrained_name:
			raise ValueError('for now we only support gpt model')

		model = PretrainedDecoderTransformerDual(args.pretrained_name, vocab_size = vocab_size, pooling_type = None)	 # none means mean pooling 
	
	if args.n_gpu > 1:
		model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

	return model 

def test(
	model_type,
	model,
	test_stats,
	test_loader,
	output_dir,
	device,
	use_cuda=False):
	'''
	Author: Sungjun Han, Anastasiia
	Description: tests and generates predictions on a teset set
	'''

	for step, batch in enumerate(test_loader):
		model.eval()

		#load the best model
		model_checkpoint_path = os.path.join(output_dir, 'checkpoint_'+ args.model_type + '.pt')
		model.load_state_dict(torch.load(model_checkpoint_path))
		
		if use_cuda:
			model.cuda()

		model.eval()
		test_pred = []
		test_labels = []
		test_loss = 0.
		with torch.no_grad():
			#prepare data, run the model for different model types
			if model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids,  masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)
		
				logits, loss = model(input_ids, segment_ids, masks, label)
			
			elif model_type in ['pretrained-transformers-decoder']:
				if use_cuda:
					batch['input_ids'] = batch['input_ids'].to(device)
					batch['segment_ids'] = batch['segment_ids'].to(device)
					batch['masks'] = batch['masks'].to(device)
					batch['label'] = batch['label'].to(device)
					batch['input_lengths'] = batch['input_lengths'].to(device)
					batch['lm_label'] = batch['lm_label'].to(device)

				inputs = {
				'input_ids': batch['input_ids'],
				'attention_mask':batch['segment_ids'],
				'token_type_ids':batch['masks'],
				'mc_token_ids':batch['input_lengths'] -1,
				'mc_labels':batch['label'],
				'labels': batch['lm_label']
				}
			
				logits, loss_mc, loss_lm = model(**inputs)
				loss = 2*loss_mc + loss_lm

				label = batch['label']

			elif model_type in ['dual_enc_bert']:
				if use_cuda:
					batch['input1'] = batch['input1'].to(device)
					batch['segment_ids1'] = batch['segment_ids1'].to(device)
					batch['masks1'] = batch['masks1'].to(device)
					batch['input2'] = batch['input2'].to(device)
					batch['segment_ids2'] = batch['segment_ids2'].to(device)
					batch['masks2'] = batch['masks2'].to(device)
					batch['label'] = batch['label'].to(device)


				inputs = {
				'input1': batch['input1'],
				'input2': batch['input2'],
				'segment_ids1':batch['segment_ids1'],
				'segment_ids2':batch['segment_ids2'],
				'attention_mask1':batch['masks1'],
				'attention_mask2':batch['masks2'],
				'y': batch['label']
				}
			
				logits, loss = model(**inputs)

				label = batch['label']

		#update keeper for log likelihood
		test_loss += loss.mean().item()
		
		#collect predicted labels by using sigmoid for the output layer
		test_pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())
		test_labels.extend(label.tolist())

	test_stats.eval(test_labels,test_pred)
	test_stats.update('loglikelihood',test_loss)
	test_stats.print()

	return test_pred, test_stats
	
def train(
	model_type,
	model,
	num_epochs,
	optimizer,
	train_loader, 
	scheduler,
	stats,
	device,
	grad_norm_clip,
	evaluate_during_training=False,
	val_loader=None,
	val_stats=None,
	early_stopping = 0,
	use_cuda=False,
	output_dir=None):
	'''
	Author: Sungjun Han, Anastasiia
	Description: trains on a train set for a given number of epochs, implements EarlyStopping if specified - saves the best model
	'''
	val_accuracy = 0.0
	earlyStop = 0
	for epoch in tqdm(range(num_epochs), desc='epoch'):
		labels = []
		pred = []
		train_loss = 0
		model.train()
		model.zero_grad()
		for step, batch in enumerate(tqdm(train_loader, desc ='train-step')):

			#prepare data, run the model for different model types
			if model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids,  masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)

				logits, loss = model(input_ids, segment_ids, masks, y=label)
			
			elif model_type in ['pretrained-transformers-decoder']:
				if use_cuda:
					batch['input_ids'] = batch['input_ids'].to(device)
					batch['segment_ids'] = batch['segment_ids'].to(device)
					batch['masks'] = batch['masks'].to(device)
					batch['label'] = batch['label'].to(device)
					batch['input_lengths'] = batch['input_lengths'].to(device)

				inputs = {
				'input_ids': batch['input_ids'],
				'attention_mask':batch['segment_ids'],
				'token_type_ids':batch['masks'],
				'mc_token_ids':batch['input_lengths'] -1,
				'labels':batch['label']
				}
				
				logits, loss = model(**inputs)
				label = batch['label']
			
			elif model_type in ['dual_enc_bert']:
				if use_cuda:
					batch['input1'] = batch['input1'].to(device)
					batch['segment_ids1'] = batch['segment_ids1'].to(device)
					batch['masks1'] = batch['masks1'].to(device)
					batch['input2'] = batch['input2'].to(device)
					batch['segment_ids2'] = batch['segment_ids2'].to(device)
					batch['masks2'] = batch['masks2'].to(device)
					batch['label'] = batch['label'].to(device)


				inputs = {
				'input1': batch['input1'],
				'input2': batch['input2'],
				'segment_ids1':batch['segment_ids1'],
				'segment_ids2':batch['segment_ids2'],
				'attention_mask1':batch['masks1'],
				'attention_mask2':batch['masks2'],
				'y': batch['label']
				}
			
				logits, loss = model(**inputs)

				label = batch['label']

			loss.backward()

			#gradient check for debugging 
			# logger.info('aasdfasdfasdf')
			# for n, p in model.named_parameters():
			# 	if p.grad is not None:
			# 		logger.info(n)
			# 		logger.info(p.grad.mean())
			# 	else:
			# 		logger.info('{} grad is None'.format(n))
			'''
			implement gradient norm clip 
			'''
			if grad_norm_clip:
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

			optimizer.step()
			if scheduler:
				scheduler.step()
			optimizer.zero_grad()

			#keep track of labels and losses during traning 
			train_loss += loss.mean().item()
			labels.extend(label.tolist())
			pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())
		
			logger.info('At step {}, train loss = {}'.format(step, loss.mean().item()))

		#update keeper for log likrlihood
		stats.update('loglikelihood',train_loss / len(train_loader))
		stats.eval(labels,pred)
		#print for status update
		logger.info('\nTrain stats:')
		stats.print()

		if evaluate_during_training:
			model, val_stats = evaluate(
				model_type = model_type,
				model =  model,
				val_loader =val_loader,
				val_stats = val_stats,
				device = device,
				use_cuda = use_cuda)
		
			#early stopping
			if early_stopping:
				current_accuracy = val_stats.keeper['accuracy'][-1]
				if current_accuracy > val_accuracy:
					earlyStop = 0
					torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type + '.pt'))
					val_accuracy = current_accuracy
					continue

				earlyStop += 1
				if early_stopping == earlyStop:
					logger.info('Early stopping criterion met - terminating')
					return model, (stats, val_stats)

				logger.info('Early stopping patience {}'.format(earlyStop))

	torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type + '.pt') )
	if evaluate_during_training:
		return model, (stats, val_stats)
		
	return model, stats

def evaluate(
	model_type,
	val_stats,
	model, 
	val_loader,
	device,
	use_cuda):
	'''
	Author: Sungjun Han, Anastasiia
	Description: evaluates on a validation set 
	'''
	
	terminate = False 
	model.eval()
	with torch.no_grad():
		labels = []
		pred = []
		total_loss = 0
		for step, batch in enumerate(val_loader):

			#prepare data, run the model for different model types
			if model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids, masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)

				logits, loss = model(input_ids, segment_ids, masks, label)
			
			elif model_type in ['pretrained-transformers-decoder']:
				if use_cuda:
					batch['input_ids'] = batch['input_ids'].to(device)
					batch['segment_ids'] = batch['segment_ids'].to(device)
					batch['masks'] = batch['masks'].to(device)
					batch['label'] = batch['label'].to(device)
					batch['input_lengths'] = batch['input_lengths'].to(device)
					batch['lm_label'] = batch['lm_label'].to(device)

				inputs = {
				'input_ids': batch['input_ids'],
				'attention_mask':batch['segment_ids'],
				'token_type_ids':batch['masks'],
				'mc_token_ids':batch['input_lengths'] -1,
				'mc_labels':batch['label'],
				'labels': batch['lm_label']
				}
				
				logits, loss_mc, loss_lm = model(**inputs)
				loss = 2*loss_mc + loss_lm

				label = batch['label']
			
			elif model_type in ['gpt2-zeroshot']:
				if use_cuda:
					batch['input_ids_1'] = batch['input_ids_1'].to(device)
					batch['input_ids_2'] = batch['input_ids_2'].to(device)
					batch['masks'] = batch['masks'].to(device)

			elif model_type in ['dual_enc_bert']:
				if use_cuda:
					batch['input1'] = batch['input1'].to(device)
					batch['segment_ids1'] = batch['segment_ids1'].to(device)
					batch['masks1'] = batch['masks1'].to(device)
					batch['input2'] = batch['input2'].to(device)
					batch['segment_ids2'] = batch['segment_ids2'].to(device)
					batch['masks2'] = batch['masks2'].to(device)
					batch['label'] = batch['label'].to(device)


				inputs = {
				'input1': batch['input1'],
				'input2': batch['input2'],
				'segment_ids1':batch['segment_ids1'],
				'segment_ids2':batch['segment_ids2'],
				'attention_mask1':batch['masks1'],
				'attention_mask2':batch['masks2'],
				'y': batch['label']
				}
			
				logits, loss = model(**inputs)

				label = batch['label']

			#update keepr for log liklihood
			total_loss += loss.mean().item()

			labels.extend(label.tolist())
			pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())

	val_stats.update('loglikelihood',total_loss / len(val_loader))
	val_stats.eval(labels,pred)

	logger.info('\nVal stats:')
	val_stats.print()

	return model, val_stats

def main(args):

	utils.set_seed(args.seed)

	if args.n_gpu > 1:
		#initializae to synchronize gpus
		torch.distributed.init_process_group(backend="nccl")

		#make distributed True to use DistributedSampler
		train_dataset_kwargs['distributed'] = True

	logger.info('Saving the output to %s' % args.output_dir)
	logger.info('CONFIGURATION:')
	logger.info(args)
	logger.info('MODEL:')
	logger.info(args.model_type)
	logger.info('TRAIN DATA PATH:')
	logger.info(args.train_tsv)
	logger.info('DEV DATA PATH:')
	logger.info(args.val_tsv)
	logger.info('TEST DATA PATH:')
	logger.info(args.test_tsv)

	# make output directory if it does not exist 
	output_dir = os.path.join(args.output_dir, '{}_{}'.format('checkpoint', args.model_type))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	logger.info('OUTPUT DATA PATH:')
	logger.info(output_dir)

	#initialize metric keeper 
	stats = metrics.MetricKeeper(args.eval_measure.split(','))
	test_stats = metrics.MetricKeeper(args.eval_measure.split(','))
	if evaluate:
		val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

	#prepare kwargs for a model of a specific model types
	if args.model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling', 'pretrained-transformers-decoder']:

		dataset_kwargs = {'data_path': args.train_tsv,
				'max_samples': args.max_samples_per_epoch,
				'sep_token': None,
				'pad_token_id': None,
				'cls_at_start': True
				}

		if 'bert' in args.pretrained_name:
			tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
		
		elif 'gpt' in args.pretrained_name:
			tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_name) 
			tokenizer.add_special_tokens({'cls_token': '[CLS]'})
			dataset_kwargs['sep_token'] = '|'
			dataset_kwargs['pad_token_id'] = 0
			dataset_kwargs['cls_at_start'] = False
		
		dataset_kwargs['tokenizer'] = tokenizer

		#initialize dataloader -- train
		train_dataset = AlphaDatasetTransformer(**dataset_kwargs)

		#initialize dataloader -- test
		dataset_kwargs['data_path'] = args.test_tsv
		test_dataset = AlphaDatasetTransformer(**dataset_kwargs)

		#initialize val-dataloader
		val_dataset = None
		if args.evaluate:
			dataset_kwargs['data_path'] = args.val_tsv
			val_dataset = AlphaDatasetTransformer(**dataset_kwargs)

		train_loader, test_loader, val_loader =prepare_dataloader(
											train_dataset, 
											test_dataset,
											val_dataset,
											args.batch_size, 
											collate_fn = alpha_collate_fn_transformer ,
											shuffle=args.shuffle, 
											drop_last = True, 
											num_workers = args.num_workers)
		
		model = transformer_initialize_model(args, vocab_size = len(tokenizer))
		# model.model.resize_token_embeddings(len(tokenizer))

	elif args.model_type =='dual_enc_bert':

		# initialize tokenizer
		tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
		#ensure that the tokenizer has all the functional tokens
		if tokenizer.cls_token == None:
			tokenizer.add_special_tokens({'cls_token': '[CLS]'})
		if tokenizer.sep_token == None:
			tokenizer.add_special_tokens({'sep_token': '[SEP]'})
		if tokenizer.eos_token == None:
			tokenizer.add_special_tokens({'eos_token': '[EOS]'})

		model = BB_DualEncoder(args.pretrained_name, tokenizer)

		dataset_kwargs = {'data_path': args.train_tsv,
				'max_samples': args.max_samples_per_epoch,
				'pad_token_id': None,
				'tokenizer':tokenizer}

		#initialize dataloader -- train
		train_dataset = AlphaDatasetDualEncoder(**dataset_kwargs)

		#initialize dataloader -- test
		dataset_kwargs['data_path'] = args.test_tsv
		test_dataset = AlphaDatasetDualEncoder(**dataset_kwargs)

		#initialize val-dataloader
		val_dataset = None
		if args.evaluate:
			dataset_kwargs['data_path'] = args.val_tsv
			val_dataset = AlphaDatasetDualEncoder(**dataset_kwargs)
		
		train_loader, test_loader, val_loader =load_dataloader_BBDualEnc(tokenizer,
											train_dataset, 
											test_dataset,
											val_dataset,
											args.batch_size, 
											shuffle=args.shuffle, 
											drop_last = True, 
											num_workers = args.num_workers)
		
	else:
		raise ValueError('model type not recognized')

	'''
	DEFINE OPTIMIZER, SCHEDULER, DEVICE 
	'''
	if args.use_cuda:
		if not torch.cuda.is_available():
			print('use_cuda=True but cuda is not available')
		device = torch.device("cuda")
		model.cuda()
	else:
		device = torch.device('cpu')

	# device = args.device

	#group parmaeters if we are weight decaying
	if args.weight_decay:
		parameters = prepare_model_parameters_weight_decay(model.named_parameters(), args.weight_decay)
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
	TRAIN and TESt
	'''
	model, stats  = train(model_type = args.model_type,
						model = model, 
						num_epochs = args.num_epochs, 
						optimizer = optimizer,
						train_loader = train_loader, 
						scheduler = scheduler, 
						stats  = stats, 
						device = device,
						grad_norm_clip = args.grad_norm_clip,
						evaluate_during_training = args.evaluate,
						val_loader=val_loader if val_loader else None,
						val_stats=val_stats if val_loader else None,
						early_stopping = args.early_stopping,
						use_cuda=args.use_cuda,
						output_dir=output_dir)
	if args.evaluate:
		stats, val_stats = stats

	logger.info('Testing...')

	# here load the best model 

	test_pred, test_stats = test(
						model_type = args.model_type,
						model = model,
						test_stats= test_stats,
						test_loader = test_loader,
						output_dir = output_dir,
						device = device, 
						use_cuda=args.use_cuda)
 
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

if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s',
		handlers=[
				logging.FileHandler(os.path.join(args.output_dir,'log-gpu:{}.txt'.format(args.use_cuda))),
				logging.StreamHandler()
		]
		)
	main(args)
