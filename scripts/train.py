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
from tqdm import tqdm


import nli.utils as utils 
from nli.data import AlphaDatasetBaseline, AlphaDataset, load_dataloader
import nli.preprocess as preprocess
import nli.metrics as metrics
from nli.tokenization import WhiteSpaceTokenizer
from nli.embedding import build_embedding_glove


from nli.models.BoW import *

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
parser.add_argument('--batch_size', default=128, type=int)
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

#deep learning models 
parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')
parser.add_argument('--tokenizer', default='regular', help='choose tokenizer: regular/bpe/pretrained')
parser.add_argument('--optimizer', default='adam', help='adam/adamW/sgd/..')
parser.add_argument('--beta_1', default=0.99, type=float, help='beta1 for first moment')
parser.add_argument('--beta_2', default=0.999, type=float, help='beta2 for second moment')
parser.add_argument('--weight_decay', default=False, type=bool)
parser.add_argument('--optimizer_eps', default=1e-6, type=float)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--scheduler', default=None, help='')
parser.add_argument('--num_warming_steps', default=None, help='number of warming steps for the scheduler')
parser.add_argument('--dropout', default=0.5, type=float, help='')
parser.add_argument('--grad_norm_clip', default=None, type=float, help='clip the norm')
parser.add_argument('--grad_accumulation_steps', default=None, type=int, help='number of steps to accumulate gradient')

#model -static embedding  
parser.add_argument('--glove_model', default='glove-wiki-gigaword-50', type=str, help='choose from fasttext-wiki-news-subwords-300, conceptnet-numberbatch-17-06-300, word2vec-ruscorpora-300, word2vec-google-news-300, glove-wiki-gigaword-50, glove-wiki-gigaword-100, glove-wiki-gigaword-200, glove-wiki-gigaword-300, glove-twitter-25, glove-twitter-50, glove-twitter-100, glove-twitter-200') 
parser.add_argument('--se_hidden_encoder_size', default=100, type=int, help='hidden size for the encoder')
parser.add_argument('--se_hidden_decoder_size', default=100, type=int, help='hidden size for the decoder')

#model - static embedding Mixture specific
parser.add_argument('--sem_pooling', default='sum', help='choose from sum/product/max -- used to pool the vectors from premise/hyp1/hyp2')

#model -static embedding RNN specific
parser.add_argument('--sernn_num_rnn_layers', default=2, type=int, help='number of rnn layers for encoder') 
parser.add_argument('--sernn_bidirectional', default=False, type=bool, help='bidirectional for encoder or not') 

#directory for data/train/val

def main(args):

	logger.info('Saving the output to %s' % args.output_dir)
	logger.info('CONFIGURATION:')
	logger.info(args)
	logger.info('MODEL:')
	logger.info(args.model_type)
	logger.info('TRAIN DATA PATH:')
	logger.info(args.train_tsv)
	logger.info('DEV DATA PATH:')
	logger.info(args.dev_tsv)

	if  args.model_type == 'BoW':
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


		train_dataset = AlphaDatasetBaseline(args.train_tsv, args.max_samples_per_epoch)
		logger.info('Train Dataset has %d samples' % len(train_dataset))

		#initialize a class to keep track of model progress
		stats = metrics.MetricKeeper(args.eval_measure.split(','))

		if args.evaluate:
			val_dataset = AlphaDatasetBaseline(args.val_tsv)
			logger.info('Validation Dataset has %d samples' % len(val_dataset))
			val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

		# for baseline 
		model_kwargs = {
			 'classifier': classifier,
			 'sim_function': args.bow_sim_function,
			 'weight_function': args.bow_weight_function,
			 'max_cost' : args.bow_max_cost,
			 'bidirectional' : args.bow_bidirectional,
			 'lemmatize':args.bow_lemmatize,}
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

	vocab = json.load(open(args.vocab, 'r'))
	if args.tokenizer == 'regular':
		#sanity check
		assert(vocab['unk_token'] is not None)
		assert(vocab['start_token'] is not None)
		assert(vocab['end_token'] is not None)
		tokenizer = WhiteSpaceTokenizer(vocab)

	'''
	elif ..
	here we will further implement initializing  

	1. word piece tokenizer

	2. pretrained tokenizer from hugging face

	'''

	#initialize dataloader
	train_dataset = AlphaDataset(args.train_tsv, tokenizer, args.max_samples_per_epoch)
	train_loader=load_dataloader(train_dataset, batch_size, shuffle=True, drop_last = True, num_workers = 0)
	stats = metrics.MetricKeeper(args.eval_measure.split(','))

	#initialize val-dataloader
	if args.evaluate:
		val_dataset = AlphaDataset(args.dev_tsv, tokenizer, args.max_samples_per_epoch)
		val_loader=load_dataloader(val_dataset, batch_size, shuffle=True, drop_last = True, num_workers = 0)
		val_stats = metrics.MetricKeeper(args.eval_measure.split(','))

	#initialize model 
	if  args.model_type == 'StaticEmb-mixture':
		embedding_matrix = build_embedding_glove(vocab, args.glove_model)
		model = StaticEmbeddingMixture(embedding_matrix,
				 args.se_hidden_encoder_size,
				 args.se_hidden_decoder_size,
				 args.dropout,
				 args.sem_pooling)

	elif args.model_type == 'StaticEmb-rnn':
		embedding_matrix = build_embedding_glove(vocab, args.glove_model)
		model = StaticEmbeddingRNN (embedding_matrix,
				 args.num_rnn_layers,
				 args.se_hidden_encoder_size,
				 args.se_hidden_decoder_size,
				 args.dropout,
				 args.sernn_bidirectional)

	if args.use_cuda:
		if not torch.cuda_is_available():
			print('use_cuda=True but cuda is not available')
		device = torch.device("cuda")
		model.cuda()
	else:
		device = torch.device('cpu')

	#group parmaeters if we are weight decaying
	if args.weight_decay:
		parameters = prepare_model_parameters_weight_decay(mode.named_parameters())
	else:
		parameters = model.parameters()

	#optimizer 
	if args.optimzer == 'adam':
		torch.optim.Adam(parameters, args.learning_rate, (args.optimizer_beta_1,optimizer_beta_2), args.optimizer_eps)

	#scheduler 
	if args.scheduler:
		pass

	'''
	we might need to write a separate training loop for pretrained BERT models but we will leave it like this for now

	this training loop was written for StaticEmb models 
	'''
	val_loss = 1000 # 1000 for early stop
	for epoch in tqdm(range(args.num_epochs), desc='epoch'):
		labels = []
		pred = []
		train_loss = 0
		for step, batch in enumerate(train_loader):
			hyp1, hyp2, premise, label = batch['hyp1'], batch['hyp2'], batch['obs'], batch['label']
			
			if args.use_cuda:
				hyp1 = hyp1.to(device)
				hyp2 = hyp1.to(device)
				premise = hyp1.to(device)
				label = hyp1.to(device)

			model.train()
			loss, logtis = model(premise, hyp1, hyp2, label)

			logging.info('Loss: {}'.format(loss))
			loss.backward()

			if args.grad_norm_clip:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

			optimizer.step()
			if args.scheduler:
				scheduler.step()
			model.zero_grad()

			#keep things 
			train_loss += loss.mean().item()
			labels.extend(label.tolist())
			pred.extend(logits>0.5).tolist()

		#update keepr for log liklihood
		for epoch, l in enumerate(L):
			stats.update('loglikelihood',train_loss)
		stats.eval(labels,pred)

		if evaluate:
			'''
			here implement 
			1.evaluation  
			2. early stopping
			3. saving best model at a check point pth
			'''
			NotImplementedError 
	#save 
	stats_name = '_'.join([args.model_type, args.output_name,'stats.json'])
	output_path = os.path.join(args.output_dir, stats_name)

	checkpoint = {
	'stats': stats.keeper,
	'val_stats': val_stats.keeper if args.evaluate else None,
	'args': args.__dict__
	}

	with open(output_path, 'w') as f:
		json.dump(checkpoint, f, indent=4)

def prepare_model_parameters_weight_decay(named_parameters):
	no_decay = ['bias', 'LayerNorm.weight']
	groued_params = [
	{'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
	'weight_decay': args.weight_decay},
	{'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)],
	'weight_decay': 0.0}
	]

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
