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

		this pools the representations - but should only pool the contextual embeddings excluding the padding token 
		'''
		
		output = self.model(input_ids = input_ids, attention_mask = masks, token_type_ids = segment_ids)
		sentence_emb = mean_pooling(output[0], masks) # mean pooling supoorted
		logits = self.classifier(sentence_emb)

		if y is not None:
			loss = self.loss_fn(logits.view(-1),y.view(-1))
			return logits, loss

		return logits

#Mean Pooling - Take attention mask into account for correct averaging
# credit: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
def mean_pooling(token_embeddings, attention_mask):
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def max_pooling(token_embeddings, attention_mask):
	# adapted from the above mean pooling 
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	signs = (input_mask_expanded == 0.)* torch.sign(token_embeddings) # only extract the sign from the paddings  
	
	input_mask_expanded = (input_mask_expanded == 0.)*-float('inf') 
	print(input_mask_expanded)
	return token_embeddings * input_mask_expanded * signs 

# class PretrainedDecoderTransformer(nn.Module):
# 	'''
# 	author:  Sungjun Han

# 	This model will 
# 	1. take the last token hidden embedding and use this for prediction 
# 	2. language model as an auxiliary objective  

# 	this model assumes we have the representation format [obs1, obs2, hyp1, hyp2]

# 	-> this model does not work 
# 	'''
# 	def __init__(self, model_name,  dropout=0.1):
# 		super().__init__()
# 		self.model = transformers.GPT2Model.from_pretrained(model_name)
# 		self.loss_fn = nn.BCEWithLogitsLoss()
# 	def forward(self, **kwargs):
# 		labels  = kwargs.pop('mc_labels')
# 		output = self.model(**kwargs)
# 		loss_lm = output.loss
# 		logits = output.mc_logits
# 		if labels is not None:
# 			loss_mc = self.loss_fn(logits.view(-1), labels.view(-1))
# 			return logits, loss_mc, loss_lm
# 		return logits

class PretrainedDecoderTransformerCLS(nn.Module):
	'''
	author:  Sungjun Han

	simple CLS classification without auxilary language modelling 

	todo: here make sure that this works - does not cause .view(-1) error 
	'''
	def __init__(self, model_name,  dropout=0.1):
		super().__init__()
		
		config = AutoConfig.from_pretrained(model_name)
		config.summary_type = "cls_index"
		config.num_labels = 1
		config.summary_first_dropout = dropout
		self.model = transformers.GPT2Model.from_pretrained(model_name)
	def forward(self, **kwargs):

		output = self.model(**kwargs)
		loss = output.loss
		logits = output.logits
		return logits, loss

class PretrainedDecoderTransformerDual(nn.Module):
	'''
	author:  Sungjun Han

	This model will assume a dual-encoder archiecture 
	- language modelling auxilary objetive only for the encoder with the correct hypothesis! 
	'''
	def __init__(self, 
	model_name, 
	vocab_size = None,  
	dropout=0.1,
	pooling_type = None):
		super().__init__()
		
		self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name)
		if vocab_size:
			self.model.resize_token_embeddings(len(vocab_size))
		self.pooling_type = pooling_type # in max, min, mean 

		self.seq_head = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(config.n_embd*2, config.n_embd),
			nn.ELU(),
			nn.Dropout(dropout),
			nn.Linear(config.n_embd, 1)
		) # *2 because [h1-h2, h1*h2]

		self.loss_fn = nn.BCEWithLogitsLoss()
		self.lm_loss_fn = nn.CrossEntropyLoss()
		

	def forward(self, 
			input1,
			input2, 
			length1, 
			length2,
			labels, 
			masks1,
			masks2,
			segment_ids1, 
			segmnet_ids2):
		'''
		here we gather the hidden states which we will use for language modleling 
		'''
		output1 = self.model(input1, labels=None, attention_mask = masks1, token_type_ids = segment_ids1)
		output2 = self.model(input1, labels=None, attention_mask = masks2, token_type_ids = segment_ids2 )

		# apply only the lm model loss on the correct output 
		if self.pooling_type == 'mean': # mean pooling 
			h1 = max_pooling(output1[0], masks1)
			h2 = max_pooling(output2[0], masks2)

		else: # not specified we just  take the cls 
			h1 = output1.hidden_states[:, length1, :]
			h2 = output2.hidden_states[:, length2, :]
		
		
		
		h = torch.cat([torch.abs(h1-h2), h1*h2]) #abs so that the order does not matter 
		logits = self.seq_head(h)

		if labels is not None:
			loss = self.loss_fn(logits.view(-1), labels.view(-1))

			#here we gather the outputs associated with the correct label 
			h = torch.cat([output1.hidden_states.unsqueeze(-1), output2.hidden_states.unsqueeze(-1)], dim=-1)
			h = h[..., labels.view(-1)]
			
			#now use this extracted output for lm modelling 
			lm_logits = self.model.lm_head(h)[..., :-1, :].contiguous()
			lm_labels = torch.cat([input1.unsqueeze(-1), input2.unsqueeze(-1)], dim=-1) # B X L X H X 2
			lm_labels = lm_labels[..., labels]# B X L X H 
			lm_labels = lm_labels[..., 1:].contiguous()# B X L X H-1
		
			lm_loss = self.lm_loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))

			return logits, loss, lm_loss 
			
		return logits
	
	def zero_shot_classify(self, input1, input2, masks1, masks2):
		'''
		classify zero shot way 
		- here we calculate the log likelihood for the hypothesis 

		input 
		'''

		output1 = self.model(input1, labels =None,  **kwargs)
		output2 = self.model(input1, labels =None, **kwargs)

		label1 = input1[..., 1:].contiguous()
		label2 = input2[..., 1:].contiguous()

		logits1 = output1.hidden_states[...,:-1].contiguous()
		logits2 = output2.hidden_states[...,:-1].contiguous()

		ll1 = logits1[..., labels1.view(-1)]
		ll2 = logits2[..., labels2.view(-1)]

		ll1 = torch.sum(ll1*masks1, dim=-1) / torch.sum(masks1, dim=-1)
		ll2 = torch.sum(ll2*masks2, dim=-1) / torch.sum(masks1, dim=-1)

		return (ll1 < ll2).long()
	

if __name__ == '__main__':
	token = torch.rand(2,3,2)
	masks = (torch.rand(2,3) > 0.5)*1
	a = max_pooling(token, masks)
	print(a)