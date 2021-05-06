import numpy as np
import re


separators = ".,;:!?-'()“"+'"'

def token_to_idx(dataset, mode = 'whitespace'):
	tok2idx = dict()
	for i in range(len(dataset):
		obs1, obs2, hyp1, hyp2, label = dataset[i]
		textdata = [obs1, obs2, hyp1, hyp2]
		
		for textelem in textdata:
		
			if mode == 'whitespace':
				for token in textelem.split():
					if not token in tok2idx:
						tok2idx[token] = len(tok2idx) + 1
                        
			elif mode == 'non-letter_separators':
				for token in re.split('(\W)', textelem):
					if not token in tok2idx:
						tok2idx[token] = len(tok2idx) + 1
			else:
				raise ValueError('This tokenization mode is not supported')
	return tok2idx

def idx_to_token(tok2idx):
	idx2tok = dict()
	for k,v in tok2idx.items():
		idx2tok[v] = k
	return idx2tok

def encode(sentence):
	encoded = []
	for token in sentence.split():
		encoded.append(tok2idx[token])
	encoded = np.array(encoded, dtype=np.int32)
	return encoded

def decode(encoded):
	decoded = []
	for idx in encoded:
		decoded.append(idx2tok[idx])
	return decoded

