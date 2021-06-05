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


	def forward(self, x,y=None):

		output = self.model(x)
		x = output.last_hidden_state[:,0,:]
		logits = self.classifier(x)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits


class PretrainedTransformerPooling(nn.Module):

	def __init__(self, model_name):

		self.model = AutoModel.from_pretrained(model_name)
		self.config = AutoConfig.from_pretrained(model_name)

		hidden_size = self.config.hidden_size
		self.classifier = nn.Linear(hidden_size, 1)
		self.loss_fn = nn.BCEWithLogitsLoss()


	def forward(self, x, y=None):
		raise NotImplementedError