import numpy as np
import re
from collections import defaultdict

def tokenize(sent, 
			delimiters = ' ', 
			start_symbol=True, 
			end_symbol=True,
			lemmatizer=None):

	tokenized = [t for t in re.split(delimiters, sent) if t != ""]
	if lemmatizer is not None:
		tokenized = [lemmatizer.lemmatize(t) for t in tokenized]

	if start_symbol:
		tokenized = ['START'] + tokenized
    
	if end_symbol:
		tokenized = tokenized + ['END']
	return tokenized


def token_to_idx(dataset, 
			delimiters = ' ', 
			start_symbol = False, 
			end_symbol = False, 
			null_symbol = False,
			split_symbol = False,
			lemmatizer = None,
			min_occurence = 1):

	freq_count = defaultdict(int)

	for i in range(len(dataset)):
		_, obs1, obs2, hyp1, hyp2, label = dataset[i]
		textdata = [obs1, obs2, hyp1, hyp2]
		
		for textelem in textdata:
			for token in tokenize(textelem, ' ',  start_symbol, end_symbol, lemmatizer):
				freq_count[token] += 1
	
	tok2idx = {}
	# 0 for pad 
	if null_symbol:
		tok2idx['NULL'] = 1

	if start_symbol:
		tok2idx['START'] = 2

	if end_symbol:
		tok2idx['END'] = 3

	if split_symbol:
		tok2idx['SPLT'] = 4

	for token, count in freq_count.items():
		if count >= min_occurence and (not token in tok2idx.keys()):
			tok2idx[token] = len(tok2idx) + 1
	return tok2idx

def idx_to_token(tok2idx):
	idx2tok = dict()
	for k,v in tok2idx.items():
		idx2tok[v] = k
	return idx2tok

def encode(tokenized_sentence, tok2idx, null_token='NULL'):
	encoded = []

	null_idx = tok2idx[null]
	for token in tokenized_sentence:
		encoded.append(tok2idx.get(token, null_idx))
	return encoded

def decode(encoded, idx2tok, null_token = 'NULL'):
	decoded = []
	for idx in encoded:
		decoded.append(idx2tok.get(idx,null_token))
	return decoded

