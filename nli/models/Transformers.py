'''
Author:  Sungjun Han
   
Description: Various Pre-trained transformer models - adds heads for classification  

'''

import transformers
from transformers import AutoModel, AutoConfig
from transformers import GPT2DoubleHeadsModel
import torch.nn as nn
import torch


class PretrainedTransformerCLS(nn.Module):
	'''
	author:  Sungjun Han
	'''
	def __init__(self, model_name):

		super().__init__()

		if 'pretrained_BERTmlm' in model_name:
			model = AutoModel.from_pretrained(model_name)
			self.model = model.base_model
		else:
			self.model = AutoModel.from_pretrained(model_name)
		
		self.config = AutoConfig.from_pretrained(model_name)

		hidden_size = self.config.hidden_size
		self.classifier = nn.Linear(hidden_size, 1)
		self.loss_fn = nn.BCEWithLogitsLoss()


	def forward(self, input_ids, segment_ids, masks,y=None):

		output = self.model(input_ids = input_ids, attention_mask = masks, token_type_ids = segment_ids)
		x = output.last_hidden_state[:,0,:]
		logits = self.classifier(x)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits


class PretrainedTransformerPooling(nn.Module):
	'''
	author:  Sungjun Han
	'''
	def __init__(self, model_name, dropout=0.2):
		super().__init__()
		self.model = AutoModel.from_pretrained(model_name, cache_dir ='../huggingface')
		self.config = AutoConfig.from_pretrained(model_name, cache_dir ='../huggingface')

		hidden_size = self.config.hidden_size
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.Tanh(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size,1)
		)
		self.loss_fn = nn.BCEWithLogitsLoss()

	def forward(self, input_ids, segment_ids, masks, y=None):
		'''
		we pool the representations and use this for prediction 

		this pools the representations - but should only pool the contextual embeddings excluding the padding token 
		'''
		
		output = self.model(input_ids = input_ids, attention_mask = masks, token_type_ids = segment_ids)
		sentence_emb = self.mean_pooling(output.last_hidden_state, masks) # mean pooling supoorted
		logits = self.classifier(sentence_emb)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits

	#Mean Pooling - Take attention mask into account for correct averaging
	# credit: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
	def mean_pooling(self, token_embeddings, attention_mask):
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
	token = torch.rand(2,3,2)
	masks = (torch.rand(2,3) > 0.5)*1
	a = max_pooling(token, masks)
	print(a)