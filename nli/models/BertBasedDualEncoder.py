'''
Author: Anastasiia
Bert Based Dual Encoder
https://arxiv.org/pdf/1908.10084.pdf
Figures 1 and 2
'''
import transformers
from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch


class ClassificationHead(nn.Module):

#Author: Sungjun Han
#(same as for GPT2)

	def __init__(self, n_emb, num_layers = 3, dropout = 0.1, n_out = 1):
		super().__init__()
		self.seq = nn.ModuleList( [nn.Sequential(
					nn.LayerNorm(n_emb),
					nn.Dropout(dropout),
					Linear(n_emb, n_emb),
					nn.GELU()) for _ in range(num_layers - 1)])

		self.seq.append(nn.Sequential(
					nn.LayerNorm(n_emb),
					nn.Dropout(dropout),
					Linear(n_emb, n_out)))

	def forward(self, x):
		for layer in self.seq[:-1]:
			x = layer(x) + (x)
		x = self.seq[-1](x)
		return x


class Linear(nn.Linear):

#Author: Sungjun Han
#(same as for GPT2)

	def __init__(self, n_in, n_out):
		super().__init__(n_in, n_out)
		# orthogonal initialization
		nn.init.orthogonal_(self.weight)

class BB_DualEncoder(nn.Module):
	"""
		Description: Bert based dual encoder custom model
		model_name: str; where to import BERT from
		tokenizer: python object; the huggingface tokenizer that is compatible to the chosed model
	"""
	def __init__(self, model_name, tokenizer):

		super().__init__()

		self.config = AutoConfig.from_pretrained(model_name)
		if 'pretrained_BERTmlm' in model_name:
			model = AutoModel.from_pretrained(model_name)
			self.model = model.base_model
		else:
			self.model = AutoModel.from_pretrained(model_name)
		self.model.resize_token_embeddings(len(tokenizer))

		hidden_size = self.config.hidden_size

		self.classifier = ClassificationHead(hidden_size*3, n_out = 1)
		self.loss_fn = nn.BCEWithLogitsLoss()


	def forward(self, input1, input2, segment_ids1, segment_ids2, attention_mask1, attention_mask2, y=None):

		output_source = self.model(input_ids = input1, attention_mask = attention_mask1, token_type_ids = segment_ids1)
		#take CLS token as sentence representation
		x_source = output_source.last_hidden_state[:,0,:]
		output_target = self.model(input_ids = input2, attention_mask = attention_mask2, token_type_ids = segment_ids2)
		x_target = output_target.last_hidden_state[:,0,:]

		# constructing input to the classifier as suggested in https://arxiv.org/pdf/1908.10084.pdf Figure 1
		x = torch.cat([x_source, x_target, x_source-x_target], dim = -1)

		logits = self.classifier(x)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits

