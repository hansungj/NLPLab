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

	def __init__(self,
				 embedding,
				 hidden_decoder_size,
				 dropout,
				 kernel_sizes = [3,4,5],
				 num_kernels = [20,20,20]):

		super().__init__()

		self.embedding = embedding 
		embedding_size = embedding.size(1)
		
		self.conv1d_list_prem = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_size,
														out_channels=num_kernels[i],
														kernel_size=kernel_sizes[i])
													for i in range(len(kernel_sizes))
												])
		
		self.conv1d_list_hyp = nn.ModuleList([nn.Conv1d(in_channels=self.embedding_size,
														out_channels=num_kernels[i],
														kernel_size=kernel_sizes[i])
													for i in range(len(kernel_sizes))
												])
		
		self.decoder = nn.Sequential(
							nn.Linear(sum(num_kernels)*5,hidden_decoder_size),
							nn.ReLU(),
							nn.Dropout(dropout),
							nn.LayerNorm(),
							nn.Linear(hidden_decoder_size,1))
		self.loss_fn = nn.BCEWithLogitsLoss()
					
	def forward(self, p, h1, h2, y=None):

		p = self.embedding(p) # B X L X H
		h1 = self.embedding(h1)
		h2 = self.embedding(h2)
		
		#reshape for Conv1d
		p = p.permute(0, 2, 1) # B X H X L
		h1 = h1.permute(0, 2, 1)
		h2 = h2.permute(0, 2, 1)

		#CNN + ReLU 
		p_conv_list = [F.relu(conv1d(p)) for conv1d in self.conv1d_list_prem]
		h1_conv_list = [F.relu(conv1d(h1)) for conv1d in self.conv1d_list_hyp]
		h2_conv_list = [F.relu(conv1d(h2)) for conv1d in self.conv1d_list_hyp]
		
		#max_pooling
		p_pool_list = [F.max_pool1d(p_conv, kernel_size=p_conv.shape[2])
            for p_conv in p_conv_list]
		h1_pool_list = [F.max_pool1d(h1_conv, kernel_size=h1_conv.shape[2])
            for h1_conv in h1_conv_list]
		h2_pool_list = [F.max_pool1d(h2_conv, kernel_size=h2_conv.shape[2])
            for h2_conv in h2_conv_list]
		
		#concatenate to get a single feature vector
		p = torch.cat([p_pool.squeeze(dim=2) for p_pool in p_pool_list],
                         dim=1)
		h1 = torch.cat([h1_pool.squeeze(dim=2) for h1_pool in h1_pool_list],
                         dim=1)
		h2 = torch.cat([h2_pool.squeeze(dim=2) for h2_pool in h2_pool_list],
                         dim=1)			

		#concatenate (p - h1, p - h2, p , h1, h2)
		concat = torch.cat([p, h1, h2, p-h1, p-h2],dim=-1)
		logit = self.decoder(concat)

		if y is None:
			return logit

		loss = self.loss_fn(logit, y.view(-1))
		return logit, loss