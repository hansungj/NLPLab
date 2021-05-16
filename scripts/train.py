#generate.py
'''

prepeare data into h5 file which can be used to run the models 

'''
import logging
import os
import numpy as np
import argparse
import pickle
import time
import json

import nli.utils as utils 
from nli.data import AlphaLoader
import nli.preprocess as preprocess
import nli.metrics as metrics

from nli.models.BoW import *

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory 
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--vocab', default='vocab.json', type=str)

#directory for data/train/val - but questions only tokenized
parser.add_argument('--train_pickle', default='train.pickle', type=str)
parser.add_argument('--val_pickle', default='dev.tsv', type=str)

#directory for data/train/val - but questions preprocessed
parser.add_argument('--train_h5', default='alphanli/tsv/train.h5', type=str)
parser.add_argument('--val_h5', default='alphanli/tsv/dev.h5', type=str)

#directory for output
parser.add_argument('--annot_pred', default='annot_pred.lst', type=str)
parser.add_argument('--annot_label', default='annot_label.lst', type=str)
parser.add_argument('--output_dir', default='data', type=str)
parser.add_argument('--output_name', default='', type=str)

#general training settings 
parser.add_argument('--model_type', default='BoW', type=str)
parser.add_argument('--num_epochs', default = 20, type=int)
parser.add_argument('--max_samples_per_epoch', type=int)
parser.add_argument('--eval_measure', default = 'accuracy') # put multiple eval measures separated by ','

#model - BOW options
parser.add_argument('--bow_classifier', default='maxent', type=str)
parser.add_argument('--bow_sim_function', default='levenshtein', type=str)
parser.add_argument('--bow_weight_function', default='idf', type=str)
parser.add_argument('--bow_max_cost', default=100, type=int)
parser.add_argument('--bow_bidirectional', default=False, type=bool)

#model - BOW-logistic regression classifier options
parser.add_argument('--bow_lgr_bias', default=True, type=bool)
parser.add_argument('--bow_lgr_lr', default=0.1, type=float)
parser.add_argument('--bow_lgr_regularization', default=True)
parser.add_argument('--bow_lgr_regularization_coef', default=0.1)

#model - BOW-Maximum entropy classifier options
parser.add_argument('--bow_me_step_size', default=0.2, type=float)
parser.add_argument('--bow_me_num_buckets', default=10, type=int)
parser.add_argument('--bow_me_lr', default=0.1, type=float)
parser.add_argument('--bow_me_regularization', default=True, type=bool)
parser.add_argument('--bow_me_regularization_coef', default=0.1)

#directory for data/train/val

def main(args):

	logger.info('Saving the output to %s' % args.output_dir)
	logger.info('CONFIGURATION:')
	logger.info(args)

	logger.info('MODEL:')
	logger.info(args.model_type)
	if  args.model_type == 'BoW':
		logger.info('BASELINE CLASSIFIER:')
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

		model_kwargs = {
					 'vocab': args.vocab,
					 'classifier': classifier,
					 'sim_function': args.bow_sim_function,
					 'weight_function': args.bow_weight_function,
					 'max_cost' : args.bow_max_cost,
					 'bidirectional' : args.bow_bidirectional}	

		if args.train_pickle is None:
			raise ValueError('Bag of Words is trained using only tokenized')


		data_path = os.path.join(args.data_dir, args.train_pickle)
		vocab_path = os.path.join(args.data_dir, args.vocab)


		logger.info('DATA PATH:')
		logger.info(data_path)

		dataset_kwargs = {
				'data_type':'pickle',
				'data_path':data_path,
				'vocab':vocab_path,
				'max_samples':args.max_samples_per_epoch,
				'batch_size': 1,
				'drop_last': False}

		dataset = AlphaLoader(**dataset_kwargs)
		model = BagOfWords(**model_kwargs)

		logger.info('Dataset has %d samples' % len(dataset.dataset))

		#initialize a class to keep track of model progress
		stats = metrics.MetricKeeper(args.eval_measure.split(','))

		# for baseline 
		logger.info('FITTING')

		starttime = time.time()
		model.fit(dataset)
		logger.info('Fitting BoW took {:.2f}s - {:.5f}s per data point'.format(starttime - time.time(), (starttime - time.time()) / len(dataset.dataset)))

		print(model.coded)
		print(model.labels)

		for epoch in range(args.num_epochs):
			L, pred = model.train(dataset)

			#update keepr for log liklihood
			stats.update('loglikelihood',L,epoch)

			#convert to discrete predictions
			y_pred = np.argmax(pred,axis=-1)
			y = model.labels 

			stats.eval(y,y_pred, epoch)
			for eval_name, eval_history in stats.keeper.items():
				logger.info('At epoch {} - {} - {}'.format(epoch, eval_name, eval_history[epoch]))



	# for deep learning models - we can share the same training loop 
	for epochs in range(args.num_epochs):
		#train_loop(model, dataset)
		pass

	stats_name = '_'.join([args.model_type, args.output_name,'stats.json'])
	output_path = os.path.join(args.output_dir, stats_name)

	checkpoint = {
	'stats': stats.keeper,
	'args': args.__dict__
	}
	with open(output_path, 'w') as f:
		json.dump(checkpoint, f, indent=2)

def train_loop(model, dataset):
	NotImplementedError 


if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)
