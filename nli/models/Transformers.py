'''

Pretrained bert 

'''


from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch


class PretrainedTransformerCLS(nn.Module):

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

	def __init__(self, model_name, dropout=0.2):

		self.model = AutoModel.from_pretrained(model_name)
		self.config = AutoConfig.from_pretrained(model_name)

		hidden_size = self.config.hidden_size
		self.classifier = nn.Sequential(
			nn.LayerNorm(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, hidden_size),
			nn.GELU(),
			nn.LayerNorm(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size,1)
		)
		self.loss_fn = nn.BCEWithLogitsLoss()

	def forward(self, x, y=None):
		'''
		we pool the representations and use this for prediction 
		'''
		
		output = self.model(input_ids = input_ids, attention_mask = masks, token_type_ids = segment_ids)
		sentence_emb = torch.max(output, dim =-1)
		logits = self.classifier(logits)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits