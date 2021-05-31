import numpy as np
import re
from collections import defaultdict
import json
import sys


def tokenize(sent,
			delimiters = r'\s+', 
			start_symbol=None, 
			end_symbol=None):
	
	tokenized = re.split(delimiters, sent)
	if start_symbol:
		tokenized.insert(0, start_symbol)
    
	if end_symbol:
		tokenized.append(end_symbol)
	return tokenized

def frequency_count(dataset, lower, delimiters = r'\s+',  start_symbol=True, end_symbol=True):
	freq_count = defaultdict(int)
	for i in range(len(dataset)):
		_, obs1, obs2, hyp1, hyp2, label = dataset[i]
		textdata = [obs1, obs2, hyp1, hyp2]
		for textelem in textdata:
			if lower:
				textelem = textelem.lower()
			for token in tokenize(textelem, delimiters):
				freq_count[token] += 1
	return freq_count

def token_to_idx(freq_count, 
			delimiters = r'\s+', 
			pad_symbol = None,
			start_symbol = None, 
			end_symbol = None, 
			unk_symbol = None,
			split_symbol = None):
	
	# 0 for pad 
	tok2idx = {}
	if pad_symbol:
		tok2idx[pad_symbol] = len(tok2idx)
	
	if unk_symbol:
		tok2idx[unk_symbol] = len(tok2idx)

	if start_symbol:
		tok2idx[start_symbol] = len(tok2idx)

	if end_symbol:
		tok2idx[end_symbol] = len(tok2idx)

	if split_symbol:
		tok2idx[split_symbol] = len(tok2idx)

	for k, v in freq_count.items():
		if not k in tok2idx.keys(): #if we don't check this, we end up with 'END' and 'SPLT' tokens having quite random indeces
			tok2idx[k] = len(tok2idx)

	return tok2idx


def idx_to_token(tok2idx):
	idx2tok = {v:k for k,v in tok2idx.items()} 
	return idx2tok

def encode(tokenized_sentence, tok2idx, unk_token):
	encoded = []
	unk_idx = tok2idx[unk_token]
	for token in tokenized_sentence:
		encoded.append(tok2idx.get(token, unk_idx))
	return encoded

def decode(encoded, idx2tok, unk_token): 
	#unk token needs to be provided
	decoded = []
	for idx in encoded:
		decoded.append(idx2tok.get(idx,unk_token))
	return decoded

if __name__ == '__main__':

	p1 = 'Chad went to get the wheel alignment measured on his car'	
	p2 = 'The mechanic provided a working alignment with new body work'	
	h1 = 'Chad was waiting for his car to be washed'	
	h2 = 'Chad was waiting for his car to be finished'

	label = 1

	dataset = [('id', p1, p2, h1, h2, label)]
	freq_count = frequency_count(dataset, delimiters = r'\s+', start_symbol = True, end_symbol = True)
	print(freq_count)
	tok2idx = token_to_idx(freq_count, start_symbol = True, end_symbol = True)
	print(tok2idx)