import numpy as np
import re

def tokenize(sent, delimiters = ' ', start_symbol, end_symbol):
        
    tokenized = [t for t in re.split(delimiters, sent) if t != ""]
    
    if start_symbol:
        tokenized = ['START'] + tokenized
    
    if end_symbol:
        tokenized = tokenized + ['END']

	return tokenized


def token_to_idx(dataset, delimiters = ' ', start_symbol = False, end_symbol = False, null_symbol = False):
	tok2idx = dict()
    
    if null_symbol:
        tok2idx['NULL'] = 0
        
	for i in range(len(dataset)):
		obs1, obs2, hyp1, hyp2, label = dataset[i]
		textdata = [obs1, obs2, hyp1, hyp2]
		
		for textelem in textdata:
			for token in tokenize(textelem, delimiters = ' ', start_symbol = start_symbol, end_symbol = end_symbol):
				if not token in tok2idx:
					tok2idx[token] = len(tok2idx) + 1
		
	return tok2idx

def idx_to_token(tok2idx):
	idx2tok = dict()
	for k,v in tok2idx.items():
		idx2tok[v] = k
	return idx2tok

def encode(tokenized_sentence, tok2idx):
	encoded = []
	for token in tokenized_sentence:
		encoded.append(tok2idx[token])
	return encoded

def decode(encoded, idx2tok):
	decoded = []
	for idx in encoded:
		decoded.append(idx2tok[idx])
	return decoded

