# 
import torch
import torch.nn as nn
import torch.nn.functional as F 

class StaticEmbeddingMixture(nn.Module):

	def __init__(self,
				 embedding,
				 hidden_encoder_size,
				 hidde_decoder_size,
				 dropout,
				 pooling = 'sum'):

		super().__init__()

		self.embedding = embedding
		hidden_size = embedding.size(1)
		self.encoder_premise = nn.Sequential(
									nn.Linear(hidden_size,hidden_encoder_size),
									nn.ReLU(),
									nn.LayerNorm(),
									nn.Dropout(dropout),
									nn.Linear(hidden_encoder_size,hidden_encoder_size))
		self.encoder_hyp = nn.Sequential(
									nn.Linear(hidden_size,hidden_encoder_size),
									nn.ReLU(),
									nn.Dropout(dropout),
									nn.LayerNorm(),
									nn.Linear(hidden_encoder_size,hidden_encoder_size))

		self.decoder = nn.Sequential(
									nn.Linear(hidden_encoder_size*5,hidde_decoder_size),
									nn.ReLU(),
									nn.Dropout(dropout),
									nn.LayerNorm(),
									nn.Linear(hidde_decoder_size,1))
		self.loss_fn = nn.BCEWithLogitsLoss()

		if pooling == 'sum':
			self.pooling = torch.sum

		elif pooling =='product':
			self.pooling = torch.prod

		elif pooling =='max':
			self.pooling = torch.max

	def forward(self, p, h1, h2, y=None ):

		p = self.embedding(p) # B X L X H
		h1 = self.embedding(h1)
		h2 = self.embedding(h2)

		#pool
		p = self.pooling(p, dim=1) #B x 1 X H
		h1 = self.pooling(h1, dim=1)
		h2 = self.pooling(h2, dim=1)

		#encode 
		p = self.encoder_premise(p)
		h1 = self.encoder_hyp(h1)
		h2 = self.encoder_hyp(h2)

		#concatenate (p - h1, p - h2, p , h1, h2)
		concat = torch.cat([p, h1, h2, p-h1, p-h2],dim=-1)
		logit = self.decoder(concat)

		if y is None:
			return logit

		loss = self.loss_fn(logit, y.view(-1))
		return logit, loss


class StaticEmbeddingRNN(nn.Module):

	def __init__(self,
				 embedding,
				 num_rnn_layers,
				 hidden_encoder_size,
				 hidde_decoder_size,
				 dropout,
				 bidirectional=True):

		super().__init__()

		self.embedding = embedding
		self.encoder_premise = nn.GRU(hidden_size, hidden_encoder_size, num_rnn_layers, batch_first = True, dropout=dropout)
		self.encoder_hyp = nn.GRU(hidden_size, hidden_encoder_size, num_rnn_layers, batch_first = True, dropout=dropout)

		if bidirectional:
			hidden_encoder_size *= 2
		self.decoder = nn.Sequential(
							nn.Linear(hidden_encoder_size*5,hidde_decoder_size),
							nn.ReLU(),
							nn.Dropout(dropout),
							nn.LayerNorm(),
							nn.Linear(hidde_decoder_size,1))
		self.loss_fn = nn.BCEWithLogitsLoss()

	def forward(self, p, h1, h2, y=None):

		p = self.embedding(p)
		h1 = self.embedding(h1)
		h2 = self.embedding(h2)

		#encode 
		p, _ = self.encoder_premise(p) # only use the last 
		h1, _ = self.encoder_hyp(h1)
		h2, _ = self.encoder_hyp(h2)

		#concatenate (p - h1, p - h2, p , h1, h2)
		concat = torch.cat([p, h1, h2, p-h1, p-h2],dim=-1)
		logit = self.decoder(concat)

		if y is None:
			return logit

		loss = self.loss_fn(logit, y.view(-1))
		return logit, loss

class StaticEmbeddingCNN(nn.Module):
	'''
	
	use different sized conv kernels - do some kind of pooling to make them into a single vector 

	'''

	def __init__(self, x):
		pass
	
