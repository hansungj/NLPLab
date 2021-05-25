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
from nli.data import AlphaDatasetBaseline
import nli.preprocess as preprocess
import nli.metrics as metrics

from nli.models.BoW import *

from scripts.build_vocab import build_vocabulary

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory 
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--vocab', default=None, type=str)

#directory for data/train/val 
parser.add_argument('--train_tsv', default='data/alphanli/tsv/train.tsv', type=str)
parser.add_argument('--val_tsv', default='data/alphanli/tsv/dev.tsv', type=str)

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
parser.add_argument('--num_epochs', default =1, type=int, help = 'Number of training epochs')
parser.add_argument('--max_samples_per_epoch', type=int, help='Number of samples per epoch')
parser.add_argument('--evaluate', default=True, type=int, help='Decide to evaluate on validation set')
parser.add_argument('--eval_measure', default = 'accuracy', help='Decide on evaluation measure') # put multiple eval measures separated by ','

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

#model - pretrained embedding options



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

		elif args.bow_classifier == 'prc':

			prc_kwargs = {
					'num_features': 2 if args.bow_bidirectional else 1,
					'lr' : args.bow_prc_lr,
					'bias' : args.bow_prc_bias}

			logger.info(prc_kwargs)
			classifier = Perceptron(**prc_kwargs)

		logger.info('TRAIN DATA PATH:')
		logger.info(args.train_tsv)
		logger.info('DEV DATA PATH:')
		logger.info(args.train_tsv)

		train_dataset = AlphaDatasetBaseline(args.train_tsv, args.max_samples_per_epoch)
		logger.info('Train Dataset has %d samples' % len(train_dataset))

		#initialize a class to keep track of model progress
		stats = metrics.MetricKeeper(args.eval_measure.split(','))

		if args.evaluate:
			val_dataset = AlphaDatasetBaseline(args.val_tsv)
			logger.info('Validation Dataset has %d samples' % len(val_dataset))
			val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

			vocab = None
			if args.bow_sim_function in ['cosine', 'euclidian', 'distributional']: #modifications of BoW that require vocab in regular form
				if args.vocab == None: # either vocab is given or we create it from train data
					print("A vocabulary is needed for this configuration.")
					print("No vocabulary was given. Generating vocabulary from the data...")
					vocab = build_vocabulary(data_path = args.train_tsv, out_dir = 'data', vocab_type = 'regular')
				else:
					try:
						vocab = json.load(open(args.vocab , 'r'))
					except TypeError:
						print("A json vocabulary file is expected")

		# for baseline 
		model_kwargs = {
			 'classifier': classifier,
			 'sim_function': args.bow_sim_function,
			 'weight_function': args.bow_weight_function,
			 'max_cost' : args.bow_max_cost,
			 'bidirectional' : args.bow_bidirectional,
			 'lemmatize':args.bow_lemmatize,
			 'vocab':vocab}	#added vocab to BoW args so that it can be used for any sim measuare available
		model = BagOfWords(**model_kwargs)

		logger.info('FITTING')
		starttime = time.time()
		
		pred, L = model.fit_transform( train_dataset, num_epochs=args.num_epochs, ll=True)

		logger.info('Fitting and Transforming: BoW took {:.2f}s - {:.5f}s per data point'.format(starttime - time.time(), 
			(starttime - time.time()) / len(train_dataset)))

		#convert to predictions
		y_pred = np.argmax(pred,axis=-1)
		y = model.labels 

		#update keepr for log liklihood
		for epoch, l in enumerate(L):
			stats.update('loglikelihood',l)
		stats.eval(y,y_pred)

		#print to log
		for eval_name, eval_history in stats.keeper.items():
			logger.info('Train {} - {}'.format( eval_name, eval_history))

		#evaluate on validaiton set 
		if args.evaluate:
			pred = model.transform(val_dataset)

			#convert to predictions
			y_pred = np.argmax(pred,axis=-1)

			y = np.array(val_dataset.dataset['label'])
			val_stats.eval(y, y_pred)

			#print to log
			for eval_name, eval_history in val_stats.keeper.items():
				logger.info('Validation {} - {}'.format( eval_name, eval_history))

	# for other models we need to load vocabulary 

	'''
	1. check here that vocabulary given here is good 
	2. initialize tokenizer with the vocabulary
	3. initialize dataloader  - write a dataloader with tokenizer as the argument / write a collate fn that automatically pads 
	4. write training loop / evaluatation loop 
	'''

	# for deep learning models - we can share the same training loop 
	for epochs in range(args.num_epochs):
		#train_loop(model, dataset)
		pass

	stats_name = '_'.join([args.model_type, args.output_name,'stats.json'])
	output_path = os.path.join(args.output_dir, stats_name)

	checkpoint = {
	'stats': stats.keeper,
	'val_stats': val_stats.keeper if args.evaluate else None,
	'args': args.__dict__
	}

	with open(output_path, 'w') as f:
		json.dump(checkpoint, f, indent=4)




def train(model, dataloader):
	return None


def evaluate(model, dataloader):
	return None


if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)
