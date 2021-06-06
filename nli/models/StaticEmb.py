import torch
import torch.nn as nn
import torch.nn.functional as F 


class Head(nn.Module):

	def __init__(self, 
				 input_size,
				 output_size,
				 activation,
				 dropout):

		super().__init__()
		self.head = nn.Sequential(
							nn.Dropout(dropout),
							nn.LayerNorm(input_size),
							nn.Linear(input_size,output_size),
							activation,
							)

	def forward(self, x):
		return self.head(x)

class StaticEmbeddingMixture(nn.Module):

	def __init__(self,
				 embedding,
				 hidden_encoder_size,
				 hidden_decoder_size,
				 num_encoder_layers,
				 num_decoder_layers,
				 dropout,
				 pooling = 'max'):

		super().__init__()

		self.embedding = embedding
		hidden_size = embedding.weight.size(1)

		self.encoder_premise = nn.ModuleList([Head(hidden_size, hidden_encoder_size, nn.ReLU(), dropout)])
		for _ in range(num_encoder_layers-1):
			self.encoder_premise.append(Head(hidden_encoder_size, hidden_encoder_size, nn.ReLU(), dropout))

		self.encoder_hyp = nn.ModuleList([Head(hidden_size, hidden_encoder_size, nn.ReLU(), dropout)])
		for _ in range(num_encoder_layers-1):
			self.encoder_hyp.append(Head(hidden_encoder_size, hidden_encoder_size, nn.ReLU(), dropout))


		self.decoder = nn.ModuleList([Head(hidden_encoder_size*4, hidden_decoder_size, nn.ReLU(), dropout)])
		for _ in range(num_decoder_layers-2):
			self.decoder.append(Head(hidden_decoder_size, hidden_decoder_size, nn.ReLU(), dropout))
		self.decoder.append(Head(hidden_decoder_size, 1, nn.Identity(), dropout))

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
		for encoder in self.encoder_premise:
			p = encoder(p)

		for encoder in self.encoder_hyp:
			h1 = encoder(h1)
			h2 = encoder(h2)

		#concatenate [p - h1, p - h2, p , h1, h2]
		#or [p*h1, p*h2, torch.abs(p-h1), torch.abs(p-h2)]
		logit = torch.cat([p*h1, p-h1, p*h2, p-h2],dim=-1)

		for decoder in self.decoder:
			logit = decoder(logit)

		if y is None:
			return logit

		loss = self.loss_fn(logit.view(-1), y.view(-1))
		return logit, loss

class StaticEmbeddingRNN(nn.Module):

	def __init__(self,
				 embedding,
				 hidden_encoder_size,
				 hidden_decoder_size,
				 num_encoder_layers,
				 num_decoder_layers,
				 dropout,
				 bidirectional=True):

		super().__init__()

		self.embedding = embedding
		hidden_size = embedding.weight.size(1)
		self.hidden_encoder_size = hidden_encoder_size
		self.hidden_decoder_size = hidden_decoder_size
		self.num_encoder_layers = num_encoder_layers
		self.num_decoder_layers = num_decoder_layers
		self.bidirectional = bidirectional

		self.encoder_premise = nn.GRU(hidden_size, hidden_encoder_size, num_encoder_layers, batch_first = True, dropout=dropout, bidirectional=bidirectional)
		self.encoder_hyp = nn.GRU(hidden_size, hidden_encoder_size, num_encoder_layers, batch_first = True, dropout=dropout, bidirectional=bidirectional)

		if bidirectional:
			hidden_encoder_size *= 2

		self.decoder = nn.ModuleList([Head(hidden_encoder_size*4, hidden_decoder_size, nn.ReLU(), dropout)])
		for _ in range(num_decoder_layers-2):
			self.decoder.append(Head(hidden_decoder_size, hidden_decoder_size, nn.ReLU(),dropout))
		self.decoder.append(Head(hidden_decoder_size, 1, nn.Identity(), dropout))

		self.loss_fn = nn.BCEWithLogitsLoss()

	def forward(self, p, h1, h2, y=None):

		p = self.embedding(p)
		h1 = self.embedding(h1)
		h2 = self.embedding(h2)

		#encode 
		p, _ = self.encoder_premise(p) # only use the last 
		h1, _ = self.encoder_hyp(h1)
		h2, _ = self.encoder_hyp(h2)

		if self.bidirectional:
			B, L, _ = p.size()
			p = p.view(B, L, self.hidden_encoder_size, -1)

			_, L, __ = h1.size()
			h1 = h1.view(B, L, self.hidden_encoder_size, -1)

			_, L, __ = h2.size()
			h2 = h2.view(B, L, self.hidden_encoder_size, -1)

			p = torch.cat([p[:,-1,:,0],p[:,0,:,1]], dim=-1)
			h1 = torch.cat([h1[:,-1,:,0],h1[:,0,:,1]],dim=-1)
			h2 = torch.cat([h2[:,-1,:,0],h2[:,0,:,1]],dim=-1)
		else:
			p = p[:,-1,:]
			h1 = h1[:,-1,:]
			h2 = h2[:,-1,:]

		#concatenate (p - h1, p - h2, p , h1, h2)
		#logit = torch.cat([p, h1, h2, p-h1, p-h2],dim=-1)

		logit = torch.cat([p*h1, p-h1, p*h2, torch.abs(p-h2)],dim=-1)
		


		for decoder in self.decoder:
			logit = decoder(logit)

		if y is None:
			return logit

		loss = self.loss_fn(logit.view(-1), y.view(-1))
		return logit, loss

class StaticEmbeddingCNN(nn.Module):

	def __init__(self,
				 embedding,
				 hidden_decoder_size,
				 dropout,
				 kernel_sizes = [3,4,5],
				 num_kernels = [20,20,20]):

		super().__init__()

		self.embedding = embedding 
		embedding_size = embedding.weight.size(1)
		
		self.conv1d_list_prem = nn.ModuleList([nn.Conv1d(in_channels=embedding_size,
														out_channels=num_kernels[i],
														kernel_size=kernel_sizes[i])
													for i in range(len(kernel_sizes))
												])
		
		self.conv1d_list_hyp = nn.ModuleList([nn.Conv1d(in_channels=embedding_size,
														out_channels=num_kernels[i],
														kernel_size=kernel_sizes[i])
													for i in range(len(kernel_sizes))
												])
		
		self.decoder = nn.Sequential(
							nn.Linear(sum(num_kernels)*5,hidden_decoder_size),
							nn.ReLU(),
							nn.Dropout(dropout),
							nn.LayerNorm(hidden_decoder_size),
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
		logit = logit.view(-1)

		if y is None:
			return logit

		loss = self.loss_fn(logit, y.view(-1))
		return logit, loss
