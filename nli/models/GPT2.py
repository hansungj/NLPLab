# def max_pooling(token_embeddings, attention_mask):
# 	# adapted from the above mean pooling 
# 	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
# 	signs = (input_mask_expanded == 0.)* torch.sign(token_embeddings) # only extract the sign from the paddings  
	
# 	input_mask_expanded = (input_mask_expanded == 0.)*-float('inf') 
# 	print(input_mask_expanded)
# 	return token_embeddings * input_mask_expanded * signs 

import transformers
from transformers import GPT2DoubleHeadsModel
import torch.nn as nn
import torch


class ClassificationHead(nn.Module):
    '''
    Description : classiciation head for dual-encoder 
    '''
    def __init__(self, n_emb, num_layers=3, dropout=0.1, n_out =1):
        super().__init__()
        self.seq = nn.ModuleList([ nn.Sequential(
                nn.LayerNorm(n_emb),
                nn.Dropout(dropout),
                Linear(n_emb, n_emb),
                nn.GELU()) for _ in range(num_layers -1)])
        
        self.seq.append(nn.Sequential(
                nn.LayerNorm(n_emb),
                nn.Dropout(dropout),
                Linear(n_emb, n_out)))

    def forward(self, x):
        for layer in self.seq[:-1]:
            x = layer(x) + x
        x = self.seq[-1](x)
        return x 

class Linear(nn.Linear):
    '''
    Description : linear layer for the classification head 
        - initializes the weights orthogonally  
    '''
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)
        # orthogonal initialization 
        nn.init.orthogonal_(self.weight)

class PretrainedDecoderTransformerCLS(nn.Module):
    '''
    author:  Sungjun Han
    Description : 
        simple CLS classification without auxilary language modelling
    model_name : str
    dropout : float 
    '''
    def __init__(self, model_name,  dropout=0.1):
        super().__init__()
        
        config = AutoConfig.from_pretrained(model_name, cache_dir ='../huggingface')
        config.summary_type = "cls_index"
        config.num_labels = 1
        config.summary_first_dropout = dropout
        self.model = transformers.GPT2Model.from_pretrained(model_name, cache_dir ='../huggingface')
    def forward(self, **kwargs):

        output = self.model(**kwargs)
        loss = output.loss
        logits = output.logits
        return logits, loss


class PretrainedDecoderTransformerDual(nn.Module):
    '''
    author:  Sungjun Han

    This model will assume a dual-encoder archiecture  
    - seperate classifier head for each hypothesis then logits are concatenated then put through softmax
    - language modelling auxilary objetive only for the encoder with the correct hypothesis! 

    model_name : str
    vocab_size : int 
    num_layers : int
    dropout : float < 1
    label_mask : bool 
    '''
    def __init__(self, 
    model_name, 
    vocab_size = None,
    num_layers = 3,
    dropout=0.1,
    label_mask=False):
        super().__init__()
        
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name, cache_dir ='../huggingface')
        # self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name, cache_dir ='../huggingface')
        self.model.config.output_hidden_states = True
        config = self.model.config
        self.seq_head1 = ClassificationHead(config.n_embd, num_layers=num_layers, dropout=dropout)
        self.seq_head2 = ClassificationHead(config.n_embd, num_layers=num_layers, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lm_loss_fn = nn.CrossEntropyLoss()
        self.label_mask = label_mask 
        
    def lm_fill(self, inputs, masks):

        L = inputs.size(1)
        masks = masks.expand(-1, L)
        labels = inputs.masked_fill_(masks)
        return labels 

    def forward(self, 
            input1,
            input2, 
            length1, 
            length2,
            labels, 
            masks1,
            masks2,
            segment_ids1, 
            segment_ids2,
            target1, 
            target2):
        output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = target1)
        output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = target2)

        #take the cls 
        B= labels.size(0)
        h1 = output1.hidden_states[-1][torch.arange(B), length1-1]
        h2 = output2.hidden_states[-1][torch.arange(B), length2-1]
    
        h1 = self.seq_head1(h1)
        h2 =  self.seq_head2(h2)
        logits = torch.cat([h1,h2], dim=-1)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, (logits.size(-1))), labels.view(-1))
            lm_loss = output1.loss + output2.loss
            return logits, loss, lm_loss
            # return logits, loss, lm_loss 
            
        return logits

class PretrainedDecoderTransformerDualSingleClassifier(nn.Module):
    '''
    author:  Sungjun Han
    Description:
        This model will assume a dual-encoder archiecture - with single head 
        the hidden vectors are concatenated as follows : [h1, h2, h1-h2]
        - language modelling auxilary objetive only for the encoder with the correct hypothesis! 
    
    model_name : str
    vocab_size : int 
    num_layers : int
    dropout : float < 1
    label_mask : bool 
    '''
    def __init__(self, 
    model_name, 
    vocab_size = None,
    num_layers = 3,
    dropout=0.1,
    label_mask=False):
        super().__init__()
        
        self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name, cache_dir ='../huggingface')
        # self.model = transformers.GPT2LMHeadModel.from_pretrained(model_name, cache_dir ='../huggingface')
        self.model.config.output_hidden_states = True
        config = self.model.config
        self.seq_head = ClassificationHead(config.n_embd*3, num_layers=num_layers, dropout=dropout, n_out = 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lm_loss_fn = nn.CrossEntropyLoss()
        self.label_mask = label_mask 
        
    def lm_fill(self, inputs, masks):
        L = inputs.size(1)
        masks = masks.expand(-1, L)
        labels = inputs.masked_fill_(masks)
        return labels 

    def forward(self, 
            input1,
            input2, 
            length1, 
            length2,
            labels, 
            masks1,
            masks2,
            segment_ids1, 
            segment_ids2,
            target1, 
            target2):
            
        output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = target1)
        output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = target2)

        #take the cls 
        B= labels.size(0)
        h1 = output1.hidden_states[-1][torch.arange(B), length1-1]
        h2 = output2.hidden_states[-1][torch.arange(B), length2-1]
    
        pooled = torch.cat([h1, h2, h1-h2], dim=-1)
        logits = self.seq_head(pooled)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, (logits.size(-1))), labels.view(-1))
            lm_loss = output1.loss + output2.loss
            return logits, loss, lm_loss
        return logits