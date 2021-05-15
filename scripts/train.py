#generate.py
'''

prepeare data into h5 file which can be used to run the models 

'''
import logging
import os
import numpy as np
import argparse
import pickle

import nli.utils as utils 
from nli.data import AlphaLoader
import nli.preprocess as preprocess

from nli.models.BoW import *

parser = argparse.ArgumentParser()

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

#general training settings 
parser.add_argument('--model_type', default='BoW', type=str)
parser.add_argument('--num_epochs', default = 20, type=int)
parser.add_argument('--max_samples_per_epoch')
parser.add_argument('--eval_measure', default = 'accuracy')

#model - BOW options
parser.add_argument('--bow_classifier', default='maxent', type=str)
parser.add_argument('--bow_sim_function', default='levenshtein', type=str)
parser.add_argument('--bow_weight_function', default=None, type=str)
parser.add_argument('--bow_max_cost', default=100, type=int)
parser.add_argument('--bow_bidirectional', default=False, type=bool)

#model - BOW-logistic regression classifier options
parser.add_argument('--bow_lgr_bias', default=True, type=bool)
parser.add_argument('--bow_lgr_lr', default=0.01)
parser.add_argument('--bow_lgr_regularization', default=True)
parser.add_argument('--bow_lgr_regularization_coef', default=0.1)

#model - BOW-Maximum entropy classifier options
parser.add_argument('--bow_me_step_size', default=0.01)
parser.add_argument('--bow_me_num_buckets', default=10)
parser.add_argument('--bow_me_lr', default=0.01)
parser.add_argument('--bow_me_regularization', default=True, type=bool)
parser.add_argument('--bow_me_regularization_coef', default=0.1)

#directory for data/train/val

def main(args):

	if  args.model_type == 'BoW':

		if args.bow_classifier == 'maxent':
			maxent_kwargs = {
							'num_features': 2 if args.bow_bidirectional else 1,
					 		'num_classes' : 2,
							'step_size':args.bow_me_step_size,
							'num_buckets': args.bow_me_num_buckets,
							'lr' : args.bow_me_lr,
							'reg' : args.bow_me_regularization,
							'reg_lambda' : args.bow_me_regularization_coef}

			classifier = MaxEnt(**maxent_kwargs)

		elif args.bow_classifier == 'lgr':

			lgr_kwargs = {
					'num_features': 2 if args.bow_bidirectional else 1,
					'lr' : args.bow_lgr_lr,
					'bias' : args.bow_lgr_bias,
					'regularization' : args.bow_lgr_regularization,
					'lmda' : bow_lgr_regularization_coef
			}

			classifer = LogisticRegression(**lgr_kwargs)

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

		dataset_kwargs = {
				'data_type':'pickle',
				'data_path':data_path,
				'vocab':vocab_path,
				'max_samples':args.max_samples_per_epoch,
				'eval_measure': args.eval_measure,
				'mode':'baseline'}

		dataset = AlphaLoader(**dataset_kwargs)
		model = BagOfWords(**model_kwargs)

		# for baseline 
		for epochs in range(args.num_epochs):
			model.train(dataset)

	# for deep learning models - we can share the same training loop 
	for epochs in range(args.num_epochs):
		train_loop(model, dataset)

def train_loop(model, dataset):
	NotImplementedError 


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
