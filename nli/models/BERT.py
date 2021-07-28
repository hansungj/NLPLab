''' Pretrained bert '''

#from transformers import AutoModel, AutoConfig
from transformers import BertConfig, BertForMaskedLM
import torch

class BertMLM(torch.nn.Module):

	def __init__(self, model_name, tokenizer):

		super().__init__()

		#tokens_added = 0
		
		#if tokenizer.cls_token == None:
			#tokens_added += 1
		#if tokenizer.sep_token == None:
			#tokens_added += 1
		#if tokenizer.eos_token == None:
			#tokens_added += 1

		self.config = BertConfig.from_pretrained(model_name, vocab_size = tokenizer.vocab_size + tokens_added) 
		#self.config = BertConfig.from_pretrained(model_name, vocab_size = tokenizer.vocab_size + tokens_added, cache_dir = '../hugginface') 
		#output_hidden_states=True, output_attentions=True)
		
		self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)
		#self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config, cache_dir = '../hugginface')
		
		self.model.resize_token_embeddings(len(tokenizer))

		#wrapped_model = bert_model.base_model


	def forward(self, input_ids, segment_ids, attention_mask, labels=None):

		outputs = self.model(input_ids= input_ids, attention_mask = attention_mask, labels=labels)
		logits = outputs.logits

		if labels is not None:
			loss = outputs.loss
			return logits, loss

		return logits
