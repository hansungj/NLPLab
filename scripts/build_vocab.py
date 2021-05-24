'''

build_vocab.py

here we can build either 

1. regular vocabulary 
2. byte pair encoding vocabulary 

'''
import logging
import os
import numpy as np
import argparse
import pickle
import time
import json


from nli.preprocess import token_to_idx, idx_to_token, frequency_count
from nli.utils import open_tsv_file

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

#directory for data/train/val 
parser.add_argument('--data_path', default='data/alphanli/tsv/train.tsv', type=str, help='data to build the vocabulary on')
parser.add_argument('--out_dir', default='data/alphanli/tsv', type=str, help='output directory for vocab json')
parser.add_argument('--vocab_type', default='regular', type=str, help='build regular or bpe')
parser.add_argument('--min_occurence', default=1, type=str, help='minimum occurence of a token to include in a vocabulary')
parser.add_arguent('--vocabulary_size', default = None, type=int)

parser.add_argument('--start_symbol', default='<sos>', type=str, help='start of sentence token')
parser.add_argument('--end_symbol', default='<eos>', type=str, help='end of sentence token')
parser.add_argument('--split_symbol', default='</s>', type=str, help='split token')
parser.add_argument('--null_symbol', default='<unk>', type=str, help='null token')
parser.add_argument('--pad_symbol', default='<pad>', type=str, help='pad token')

def main(args):

	if args.vocab_type == 'regular':
		data = open_tsv_file(args.data_path)
		freq_count = frequency_count(dataset)

		#filter by vocabulary size
		logger.info('Unfiltered vocabulary length: {}'.format(len(freq_count)))
	

		#min occurence 
		if min_occurence is not None:
			for token, count in freq_count.items():
				if count < min_occurence:
					del freq_count[key]

		if vocbulary_size is not None:
			freq_count = sorted(((k,v) in k,v in freq_count.items()), reverse=True, key = lambda x: x[1])[:vocbulary_size]
			freq_count = {k:v for k,v in freq_count}

		token2idx = token_to_idx(freq_count, 
								delimiters = r'\s+', 
								pad_symbol = args.pad_symbol
								start_symbol = args.start_symbol, 
								end_symbol = args.end_symbol, 
								null_symbol = args.null_symbol,
								split_symbol = args.split_symbol)

		idx2token = idx_to_token(token2idx)

		logger.info('filtered vocabulary length: {}'.format(len(token2idx)))

	elif args.vocab_type == 'bpe':
		'''
		build vocabulary here using byte pair encoding

		'''
		NotImplementedError 

	#finally save
	vocab = {
	'token2idx': token2idx
	'idx2token': idx2token,
	'pad_token': args.pad_symbol
	'null_token': args.null_symbol,
	'end_token': args.end_symbol,
	'start_token':args.start_symbol,
	'split_token':args.split_symbol
	}

	with open(os.path.join(args.out_path, 'vocab.json'), 'w') as f:
		json.dump(vocab, f, indent=4)



if __name__ == '__main__':
	args = parser.parse_args()
	#If you set the log level to INFO, it will include INFO, WARNING, ERROR, and CRITICAL messages
	logging.basicConfig(
		level=logging.INFO,
		format='%(name)s: %(asctime)s: %(message)s'
		)
	main(args)