'''
Author:  Sungjun Han
   
Description: Various Pre-trained transformer models - adds heads for classification  

'''


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
		self.model = AutoModel.from_pretrained(model_name)
		self.config = AutoConfig.from_pretrained(model_name)

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
		'''
		
		output = self.model(input_ids = input_ids, attention_mask = masks, token_type_ids = segment_ids)
		sentence_emb = torch.max(output.last_hidden_state, dim =1)[0]
		logits = self.classifier(sentence_emb)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits

class PretrainedDecoderTransformer(nn.Module):
	'''
	author:  Sungjun Han

	This model will 
	1. take the last token hidden embedding and use this for prediction 
	2. language model as an auxiliary objective  

	this model assumes we have the representation format [obs1, obs2, hyp1, hyp2]
	'''
	def __init__(self, model_name,  dropout=0.1):
		super().__init__()
		
		config = AutoConfig.from_pretrained(model_name)
		config.summary_type = "cls_index"
		config.num_labels = 1
		config.summary_first_dropout = dropout
		self.model = GPT2DoubleHeadsModel.from_pretrained(model_name, config=config)
		self.loss_fn = nn.BCEWithLogitsLoss()
	def forward(self, **kwargs):

		labels  = kwargs.pop('mc_labels')
		output = self.model(**kwargs)
		loss_lm = output.loss
		logits = output.mc_logits
		if labels is not None:
			loss_mc = self.loss_fn(logits.view(-1), labels.view(-1))
			return logits, loss_mc, loss_lm
		return logits

class PretrainedDecoderTransformer(nn.Module):
	'''
	author:  Sungjun Han

	This model will assume a dual-encoder archiecture 
	- language modelling auxilary objetive only for the first head 
	'''
	def __init__(self, model_name,  dropout=0.1):
		super().__init__()
		
		config = AutoConfig.from_pretrained(model_name)
		config.summary_type = "cls_index"
		config.num_labels = 1
		config.summary_first_dropout = dropout
		self.model = GPT2DoubleHeadsModel.from_pretrained(model_name, config=config)
		self.loss_fn = nn.CrossEntropyLoss()
	def forward(self, **kwargs):
		'''
		we have the inputs 
	

		'''

		labels  = kwargs.pop('mc_labels')
		output = self.model(**kwargs)

		loss_lm = output.loss
		logits = output.mc_logits
		if labels is not None:
			loss_mc = self.loss_fn(logits.view(-1), labels.view(-1))
			return logits, loss_mc, loss_lm
		return logits