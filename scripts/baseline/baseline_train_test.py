'''
Author:  Sungjun Han, Anastasiia 
Description:
Main train method of baseline (non DL) models for aNLI task 
1. MaxEnt Classifier
2. Logistic regression
3. Perceptron 
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
from nli.dataloader import AlphaDatasetBaseline, AlphaDataset, load_dataloader_base
import nli.preprocess as preprocess
import nli.metrics as metrics
from nli.tokenization import WhiteSpaceTokenizer
from nli.embedding import build_embedding_glove
from nli.models import *

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
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--num_epochs', default =100, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate', default=False, type=bool, help='Decide to evaluate on validation set')
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','
parser.add_argument('--seed', default=1234, type=int, help='set seed for random, numpy, torch, torch.cuda')
parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')

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


def baseline_initialize_model(args):
	'''
	Author: Sungjun Han,  Anastassia 
	Description: initializes BoW baseline models 
	'''
	logger.info('BASELINE CLASSIFIER: {}'.format(args.bow_classifier))
	
	#prepare kwargs to run initialize classifiers
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
	starttime = time.time()

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


def main(args):

	utils.set_seed(args.seed)

	logger.info('Saving the output to %s' % args.output_dir)
	logger.info('CONFIGURATION:')
	logger.info(args)
	logger.info('MODEL:')
	logger.info('BoW')
	logger.info('TRAIN DATA PATH:')
	logger.info(args.train_tsv)
	logger.info('DEV DATA PATH:')
	logger.info(args.val_tsv)
	logger.info('TEST DATA PATH:')
	logger.info(args.test_tsv)

	# make output directory if it does not exist 
	output_dir = os.path.join(args.output_dir, '{}_{}'.format('checkpoint','BoW'))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	logger.info('OUTPUT DATA PATH:')
	logger.info(output_dir)

	#initialize metric keeper 
	stats = metrics.MetricKeeper(args.eval_measure.split(','))
	test_stats = metrics.MetricKeeper(args.eval_measure.split(','))
	if args.evaluate:
		val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

	model = baseline_initialize_model(args)

	# define dataset
	train_dataset = AlphaDatasetBaseline(args.train_tsv, args.max_samples_per_epoch)
	test_dataset = AlphaDatasetBaseline(args.test_tsv)

	# train 
	model, stats = baseline_train(model, train_dataset, stats, verbose=True)

	#test 
	test_pred, test_stats  = baseline_test(model, test_dataset, test_stats)

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