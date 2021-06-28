'''
Author:  Sungjun Han, Anastasiia 
Description:
Main train method of all models for aNLI task - also evaluates during training. 
If specified, it will use the model for prediction on a specified test set. 


1. baseline models 
2. sem-encoder pooling models 
3. transformer models 
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
from nli.data import AlphaDatasetBaseline, AlphaDataset, AlphaDatasetTransformer, load_dataloader_base, load_dataloader_transformer
import nli.preprocess as preprocess
import nli.metrics as metrics
from nli.tokenization import WhiteSpaceTokenizer
from nli.embedding import build_embedding_glove
from nli.models import *
from nli.models import StaticEmbeddingCNN

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

#directory for data/train/val - but questions only tokenized
# parser.add_argument('--train_pickle', default='train.pickle', type=str)
# parser.add_argument('--val_pickle', default='dev.pickle', type=str)

# #directory for data/train/val - but questions preprocessed
# parser.add_argument('--train_h5', default='alphanli/tsv/train.h5', type=str)
# parser.add_argument('--val_h5', default='alphanli/tsv/dev.h5', type=str)

#directory for output
parser.add_argument('--annot_pred', default='annot_pred.lst', type=str)
parser.add_argument('--annot_label', default='annot_label.lst', type=str)
parser.add_argument('--output_dir', default='data', type=str)
parser.add_argument('--output_name', default='', type=str)

#general training settings 
parser.add_argument('--model_type', default='BoW', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default =100, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate', default=False, type=bool, help='Decide to evaluate on validation set')
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')

#for testing 
# parser.add_argment('--fit_one_batch', default=False, type=bool, help='fits one batch for santy check for model')

#model - BOW options
parser.add_argument('--bow_classifier', default='maxent', type=str, help='Maximum entropy classifier / Logistic Regression / Perceptron')
parser.add_argument('--bow_sim_function', default='levenshtein', type=str, help='similarity function to compare tokens')
parser.add_argument('--bow_weight_function', default='idf', type=str, help='lexical weighting function for alignment cost')
parser.add_argument('--bow_max_cost', default=100, type=int, help='maximum cost constraint')
parser.add_argument('--bow_bidirectional', default=False, type=bool, help='If True, two features are considered, p(h|p) and p(p|h)')
parser.add_argument('--bow_lemmatize', default=False, type=bool, help='lemmatize the tokens or not')

#BoW - BOW-perceptron classifier options
parser.add_argument('--bow_prc_bias', default=True, type=bool)
parser.add_argument('--bow_prc_lr', default=0.1, type=float)

#BoW  - BOW-logistic regression classifier options
parser.add_argument('--bow_lgr_bias', default=True, type=bool, help='Use bias or not')
parser.add_argument('--bow_lgr_lr', default=0.1, type=float, help='learning rate for the gradient update')
parser.add_argument('--bow_lgr_regularization', default=True, help='L2 regularization for logistic regression')
parser.add_argument('--bow_lgr_regularization_coef', default=0.1, help='L2 regularization weighting')

#BoW  - BOW-Maximum entropy classifier options
parser.add_argument('--bow_me_step_size', default=0.1, type=float, help='size of the bucket - convert continous to a set of discrete features through bucketing')
parser.add_argument('--bow_me_num_buckets', default=30, type=int, help='number of buckets to use with step_size sized')
parser.add_argument('--bow_me_lr', default=0.1, type=float, help='learning rate for the gradient update')
parser.add_argument('--bow_me_regularization', default=True, type=bool, help= 'L2 regularization for maximum entropy')
parser.add_argument('--bow_me_regularization_coef', default=0.1, help='L2 regularization weighting')

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


def baseline_initialize_model(args):
	'''
	Author: Sungjun Han,  Anastassia 
	Description: initializes BoW baseline models 
	'''
	logger.info('BASELINE CLASSIFIER: {}'.format(args.bow_classifier))
	if args.bow_classifier == 'maxent':
		maxent_kwargs = {
						'num_features': 2 if args.bow_bidirectional else 1,
						'num_classes' : 2,
						'step_size':args.bow_me_step_size,
						'num_buckets': args.bow_me_num_buckets,
						'lr' : args.bow_me_lr,
						'reg' : args.bow_me_regularization,
						'reg_lambda' : args.bow_me_regularization_coef}

		logger.info(maxent_kwargs)
		classifier = MaxEnt(**maxent_kwargs)

	elif args.bow_classifier == 'lgr':

		lgr_kwargs = {
				'num_features': 2 if args.bow_bidirectional else 1,
				'lr' : args.bow_lgr_lr,
				'bias' : args.bow_lgr_bias,
				'regularization' : args.bow_lgr_regularization,
				'lmda' : args.bow_lgr_regularization_coef
		}

		logger.info(lgr_kwargs)
		classifier = LogisticRegression(**lgr_kwargs)

	elif args.bow_classifier == 'prc':

		prc_kwargs = {
				'num_features': 2 if args.bow_bidirectional else 1,
				'lr' : args.bow_prc_lr,
				'bias' : args.bow_prc_bias}

		logger.info(prc_kwargs)
		classifier = Perceptron(**prc_kwargs)
	
			# for baseline 
	model_kwargs = {
			'classifier': classifier,
			'sim_function': args.bow_sim_function,
			'weight_function': args.bow_weight_function,
			'max_cost' : args.bow_max_cost,
			'bidirectional' : args.bow_bidirectional,
			'lemmatize':args.bow_lemmatize,}
	model = BagOfWords(**model_kwargs)
	
	return model 

def baseline_train(
	model, 
	train_dataset,
	stats,
	verbose=True):
	'''
	Author: Sungjun Han
	Description: trains baseline - BoW 
	'''
	pred, L = model.fit_transform(train_dataset, num_epochs=args.num_epochs, ll=True)

	logger.info('Fitting and Transforming: BoW took {:.2f}s - {:.5f}s per data point'.format(starttime - time.time(), 
		(starttime - time.time()) / len(train_dataset)))

	#convert to predictions
	y_pred = np.argmax(pred,axis=-1)
	y = model.labels 

	#update keepr for log liklihood
	for epoch, l in enumerate(L):
		stats.update('loglikelihood',l)
	stats.eval(y,y_pred)

	#print to log if specified 
	if verbose:
		for eval_name, eval_history in stats.keeper.items():
			logger.info('Train {} - {}'.format( eval_name, eval_history))

	return model, stats

def baseline_test(
	model,
	test_dataset,
	test_stats):
	#evaluate on test set 
	pred = model.transform(test_dataset)

	#convert to predictions
	y_pred = np.argmax(pred,axis=-1)

	y = np.array(test_dataset.dataset['label'])
	test_stats.eval(y, y_pred)

	return y_pred, test_stats 

def sem_initialize_model(args):
	'''
	Author: Sungjun Han, Anastasiia
	Description: initializes sem (static embedding) neural network baseline models
	'''
	#initialize model 
	if  args.model_type == 'StaticEmb-mixture':
		padding_idx = tokenizer.vocab['token2idx'][tokenizer.pad_token]
		embedding_matrix = build_embedding_glove(vocab, args.glove_model,padding_idx, args.freeze_embedding)
		model = StaticEmbeddingMixture(embedding_matrix,
					args.se_hidden_encoder_size,
					args.se_hidden_decoder_size,
					args.se_num_encoder_layers,
					args.se_num_decoder_layers,
					args.dropout,
					args.sem_pooling)

	elif args.model_type == 'StaticEmb-rnn':
		padding_idx = tokenizer.vocab['token2idx'][tokenizer.pad_token]
		embedding_matrix = build_embedding_glove(vocab, args.glove_model, padding_idx, args.freeze_embedding)
		model = StaticEmbeddingRNN (embedding_matrix,
					args.se_hidden_encoder_size,
					args.se_hidden_decoder_size,
					args.se_num_encoder_layers,
					args.se_num_decoder_layers,
					args.dropout,
					args.sernn_bidirectional)
	
	elif args.model_type == 'StaticEmb-cnn':
		padding_idx = tokenizer.vocab['token2idx'][tokenizer.pad_token]
		embedding_matrix = build_embedding_glove(vocab, args.glove_model, padding_idx, args.freeze_embedding)
		model = StaticEmbeddingCNN(embedding_matrix,
					args.se_hidden_decoder_size,
					args.dropout)
	
	return model 

def transformer_initialize_model(args):
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
		if 'gpt' in args.pretrained_name:
			raise ValueError('we cannot use pooling classifier with GPT2')

		# fix this so that it pools 
		model = PretrainedTransformerPooling(args.pretrained_name)

	elif args.model_type == 'pretrained-transformers-decoder':
		if 'gpt' not in args.pretrained_name:
			raise ValueError('for now we only support gpt model')
			
		model = PretrainedDecoderTransformer(args.pretrained_name)
		model.model.resize_token_embeddings(len(tokenizer))
	return model 

def prepare_model_parameters_weight_decay(named_parameters):
	'''
	Author: Sungjun Han
	Description: groups parameters for weight decay as biases should not be weight decayed 
	'''

	no_decay = ['bias', 'LayerNorm.weight']
	grouped_params = [
	{'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
	'weight_decay': args.weight_decay},
	{'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
	'weight_decay': 0.0}
	]
	return grouped_params

def test(
	model_type,
	model,
	test_loader,
	output_dir):
	'''
	Author: Sungjun Han
	Description: tests and generates predictions on a teset set
	'''

	for step, batch in enumerate(test_loader):
		model.eval()

		#load best model
		model_checkpoint_path = os.path.join(output_dir, 'checkpoint_'+ args.model_type)
		model.load_state_dict(torch.load(model_checkpoint_path))
		
		if args.use_cuda:
			model.cuda()

		model.eval()
		test_pred = []
		test_labels = []
		test_loss = 0.
		with torch.no_grad():
			if model_type in ['StaticEmb-mixture', 'StaticEmb-rnn', 'StaticEmb-cnn']:
				hyp1, hyp2, premise, label = batch['hyp1'], batch['hyp2'], batch['obs'], batch['label']		
				if args.use_cuda:
					hyp1 = hyp1.to(device)
					hyp2 = hyp2.to(device)
					premise = premise.to(device)
					label = label.to(device)

				logits, loss = model(premise, hyp1, hyp2, label)

			elif model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids,  masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if args.use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)
		
				logits, loss = model(input_ids, segment_ids, masks, label)
			
			elif model_type in ['pretrained-transformers-decoder']:
				if args.use_cuda:
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
				loss = loss_mc + loss_lm

				label = batch['label']

		#update keepr for log liklihood
		test_loss += loss.mean().item()
		
		test_pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())
		test_labels.extend(label.tolist())

	test_stats.eval(test_labels,test_pred)
	test_stats.update('loglikelihood',test_loss)
	test_stats.print()

	return test_pred, test_stats
	
def train(
	model_type,
	model,
	num_epochs
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
	Author: Sungjun Han
	Description: trains on a train set for a given number of epochs
	'''

	val_accuracy = 0. # for early stop
	earlyStop = 0
	for epoch in tqdm(range(num_epochs), desc='epoch'):
		labels = []
		pred = []
		train_loss = 0
		model.train()
		model.zero_grad()
		for step, batch in enumerate(train_loader):

			if model_type in ['StaticEmb-mixture', 'StaticEmb-rnn', 'StaticEmb-cnn']:
				hyp1, hyp2, premise, label = batch['hyp1'], batch['hyp2'], batch['obs'], batch['label']		
				if use_cuda:
					hyp1 = hyp1.to(device)
					hyp2 = hyp2.to(device)
					premise = premise.to(device)
					label = label.to(device)

				logits, loss = model(premise, hyp1, hyp2, label)

			elif model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids,  masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)

				logits, loss = model(input_ids, segment_ids, masks, label)
			
			elif model_type in ['pretrained-transformers-decoder']:
				if args.use_cuda:
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
				loss = loss_mc + loss_lm
				label = batch['label']

			loss.backward()

			'''
			implement gradient norm clip 
			'''
			if grad_norm_clip:
				torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

			optimizer.step()
			if args.scheduler:
				scheduler.step()
			optimizer.zero_grad()

			#keep things 
			train_loss += loss.mean().item()
			labels.extend(label.tolist())
			pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())

		#update keepr for log liklihood
		stats.update('loglikelihood',train_loss)
		stats.eval(labels,pred)
		#print for status update
		logging.info('\nTrain stats:')
		stats.print()

		if evaluate_during_training:
			model, val_stats = evaluate(
				model_type,
				model,
				val_loader,
				early_stopping,
				device)

	if evaluate:
		return model, (stats, val_stats)
	return model, stats

def evaluate(
	model_type
	model, 
	val_loader,
	early_stopping,
	device):
	'''
	Author: Sungjun Han
	Description: evaluates on a validation set, implements EarlyStopping if specified - saves the best model 
	'''
	model.eval()
	with torch.no_grad():
		labels = []
		pred = []
		total_loss = 0
		for step, batch in enumerate(val_loader):
			if model_type in ['StaticEmb-mixture', 'StaticEmb-rnn', 'StaticEmb-cnn']:
				hyp1, hyp2, premise, label = batch['hyp1'], batch['hyp2'], batch['obs'], batch['label']		
				if use_cuda:
					hyp1 = hyp1.to(device)
					hyp2 = hyp2.to(device)
					premise = premise.to(device)
					label = label.to(device)

				logits, loss = model(premise, hyp1, hyp2, label)

			elif model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling']:
				input_ids, segment_ids, masks, label = batch['input_ids'], batch['segment_ids'], batch['masks'], batch['label']		
				if use_cuda:
					input_ids = input_ids.to(device)
					segment_ids = segment_ids.to(device)
					masks = masks.to(device)
					label = label.to(device)

				logits, loss = model(input_ids, segment_ids, masks, label)
			
			elif args.model_type in ['pretrained-transformers-decoder']:
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
				loss = loss_mc + loss_lm

				label = batch['label']

			#update keepr for log liklihood
			total_loss += loss.mean().item()

			labels.extend(label.tolist())
			pred.extend((torch.sigmoid(logits.view(-1))>0.5).long().tolist())

		val_stats.update('loglikelihood',total_loss)
		val_stats.eval(labels,pred)

		logging.info('\nVal stats:')
		val_stats.print()
	
	return model, val_stats

	#early stopping
	if early_stopping:
		current_accuracy = val_stats.keeper['accuracy'][-1]
		if current_accuracy > val_accuracy:
			earlyStop = 0
			torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type))

			val_accuracy = current_accuracy
			continue 

		earlyStop += 1
		if args.early_stopping == earlyStop:
			logging.info('Early stopping criterion met - terminating')
			break

		logging.info('Early stopping patience {}'.format(earlyStop))

	# if we dont early stop just save the last model 
	if not args.early_stopping:
		torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoint_'+ model_type))


def main(args):

	utils.set_seed(args.seed)

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
	if evaluate:
		val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

	if  args.model_type == 'BoW':
		model = baseline_initialize_model(args)

		# define dataset
		train_dataset = AlphaDatasetBaseline(args.train_tsv, args.max_samples_per_epoch)
		test_dataset = AlphaDatasetBaseline(args.test_tsv)

		# train 
		model, stats = baseline_train(model, train_dataset, stats, verbose=True)

		#test 
		test_pred, test_stats  = baseline_test(model, test_dataset, test_stats)
	
	elif args.model_type in ['StaticEmb-mixture', 'StaticEmb-rnn', 'StaticEmb-cnn']:\
		#initialize tokenizer 
		vocab = json.load(open(args.vocab, 'r'))
		if args.tokenizer == 'regular':
			#sanity check
			assert(vocab['unk_token'] is not None)
			assert(vocab['start_token'] is not None)
			assert(vocab['end_token'] is not None)
			assert(vocab['pad_token'] is not None)
			tokenizer = WhiteSpaceTokenizer(vocab)

		#initialize dataloader
		train_dataset = AlphaDataset(args.train_tsv, tokenizer, args.max_samples_per_epoch)
		test_dataset = AlphaDataset(args.test_tsv, tokenizer)

		#initialize val-dataloader
		val_dataset = None
		if args.evaluate:
			val_dataset = AlphaDataset(args.val_tsv, tokenizer, args.max_samples_per_epoch)


		train_loader, test_loader, val_loader =load_dataloader_base(
								train_dataset, 
								test_dataset,
								val_dataset,
								args.batch_size, 
								shuffle=args.shuffle,
								drop_last = True, 
								num_workers = args.num_workers)

		model = sem_initialize_model(args)
	
	elif args.model_type in ['pretrained-transformers-cls', 'pretrained-transformers-pooling', 'pretrained-transformers-decoder']:

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

		train_loader, test_loader, val_loader =load_dataloader_transformer(
											train_dataset, 
											test_dataset,
											val_dataset,
											args.batch_size, 
											shuffle=args.shuffle, 
											drop_last = True, 
											num_workers = args.num_workers)
		
		model = transformer_initialize_model(args)

	else:
		raise ValueError('model type not recognized')

	if args.model_type != 'BoW':
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

		#group parmaeters if we are weight decaying
		if args.weight_decay:
			parameters = prepare_model_parameters_weight_decay(model.named_parameters())
		else:
			parameters = model.parameters()

		#optimizer 
		if args.optimizer == 'adam':
			optimizer = torch.optim.Adam(parameters, args.learning_rate, (args.beta_1,args.beta_2), args.eps)

		#scheduler 
		if args.scheduler:
			num_training_steps = int((len(train_loader)//args.batch_size)*args.num_epochs)
			num_warmup_steps = int(num_training_steps*args.num_warming_steps)
			scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,num_training_steps)
			pass

		'''
		TRAIN and TESt
		'''
		model, stats  = train(args.model_type,
							model, 
							args.num_epochs, 
							optimizer,
							train_loader, 
							scheduler, 
							stats, 
							device,
							args.grad_norm_clip,
							evaluate_during_training = args.evaluate,
							val_loader=val_loader if val_loader else None,
							val_stats=val_stats if val_loader else None,
							early_stopping = args.early_stopping,
							use_cuda=args.use_cuda,
							output_dir=output_dir)
		if args.evaluate
			stats, val_stats = stats
								
		logging.info('Testing...')
		test_pred, test_stats = test(
							model_type,
							model,
							test_loader,
							output_dir)
 
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
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)
