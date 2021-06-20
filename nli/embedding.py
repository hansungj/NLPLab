#embedding.py 

import torch
import torch.nn as nn
import gensim
import gensim.downloader as api

def build_embedding_glove(vocab, glove_name, padding_idx, freeze=True):

	glove = api.load(glove_name)
	N = len(vocab['token2idx'])
	H = glove.vector_size
	embedding = nn.Embedding(N, H, padding_idx=padding_idx).requires_grad_(not freeze)

	#initialize embedding matrix
	for token, idx in vocab['token2idx'].items():
		# we initalize the embedding matrix 
		if token in glove:
			embedding.weight.data[idx] = torch.tensor(glove[token].copy())

	return embedding

