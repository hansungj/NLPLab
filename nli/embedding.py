#embedding.py 

import torch
import torch.nn as nn
import gensim
import gensim.downloader as api

def build_embedding_glove(vocab, glove_name):

	glove = api.load(glove_name)
	N = len(vocab['token2idx'])
	H = glove.vector_size
	embedding = nn.Embedding(N, H)

	#initialize embedding matrix
	for token, idx in vocab['token2idx'].items():
		# we initalize the embedding matrix 
		if token in glove:
			embedding.weight[idx, :] = glove[token]

	return embedding

