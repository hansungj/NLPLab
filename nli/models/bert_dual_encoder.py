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

	def __init__(self, model_name, tokenizer):

		super().__init__()
		
		tokens_added = 0
		
		#if tokenizer.cls_token == None:
		#	tokens_added += 1
		#if tokenizer.sep_token == None:
		#	tokens_added += 1
		#if tokenizer.eos_token == None:
		#	tokens_added += 1

		#print(tokenizer.vocab_size)
		#print(tokens_added)
		self.config = AutoConfig.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.model.resize_token_embeddings(len(tokenizer))

		hidden_size = self.config.hidden_size

		self.classifier = ClassificationHead(hidden_size*3, n_out = 1)
		self.loss_fn = nn.BCEWithLogitsLoss()


	def forward(self, input1, input2, segment_ids1, segment_ids2, attention_mask1, attention_mask2, y=None):

		output_source = self.model(input_ids = input1, attention_mask = attention_mask1, token_type_ids = segment_ids1)
		x_source = output_source.last_hidden_state[:,0,:]
		output_target = self.model(input_ids = input2, attention_mask = attention_mask2, token_type_ids = segment_ids2)
		x_target = output_target.last_hidden_state[:,0,:]
		
		x = torch.cat([x_source, x_target, x_source-x_target], dim = -1)
		
		logits = self.classifier(x)

		if y is not None:
			#print(logits)
			#print(logits.view(-1))
			#print(y)
			#loss = self.loss_fn(logits, y.view(-1))
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits

