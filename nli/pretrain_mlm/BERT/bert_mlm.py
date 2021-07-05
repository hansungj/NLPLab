''' Pretrained bert '''

#from transformers import AutoModel, AutoConfig
from transformers import BertConfig, BertForMaskedLM
import torch

class BertMLM(torch.nn.Module):

	def __init__(self, model_name, tokenizer):

		super().__init__()

		#if model_name == 'bert-base-uncased':
			#self.config = BertConfig.from_pretrained(model_name, vocab_size = tokenizer.vocab_size + 1) #output_hidden_states=True, output_attentions=True)
		#else:
			#self.config = BertConfig.from_pretrained(model_name) 
		
		self.config = BertConfig.from_pretrained(model_name) 
		self.model = BertForMaskedLM.from_pretrained(model_name, config=self.config)
		
		#wrapped_model = bert_model.base_model


	def forward(self, input_ids, segment_ids, attention_mask, labels=None):

		outputs = self.model(input_ids= input_ids, attention_mask = attention_mask, labels=labels)
		logits = outputs.logits

		if labels is not None:
			loss = outputs.loss
			return logits, loss

		return logits
