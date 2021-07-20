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


class PretrainedDecoderTransformer(nn.Module):
    '''
    author:  Sungjun Han

    This model will 
    1. take the last token hidden embedding and use this for prediction 
    2. language model as an auxiliary objective  

    this model assumes we have the representation format [obs1, obs2, hyp1, hyp2]

    -> this model does not work 
    '''
    def __init__(self, model_name,  dropout=0.1):
        super().__init__()
        self.model = transformers.GPT2Model.from_pretrained(model_name)
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

class PretrainedDecoderTransformerCLS(nn.Module):
    '''
    author:  Sungjun Han
    simple CLS classification without auxilary language modelling 
    todo: here make sure that this works - does not cause .view(-1) error 
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

class ClassificationHead(nn.Module):
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
    def __init__(self, n_in, n_out):
        super().__init__(n_in, n_out)
        # orthogonal initialization 
        nn.init.orthogonal_(self.weight)

class PretrainedDecoderTransformerDual(nn.Module):
    '''
    author:  Sungjun Han

    This model will assume a dual-encoder archiecture 
    - language modelling auxilary objetive only for the encoder with the correct hypothesis! 
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
            segment_ids2):

        # using the mask and only calculate the language modelling for the correct 
        if self.label_mask:
            label_mask1 = labels
            label_mask2 = (labels == 0).long()
            label1 = self.lm_fill(input1, label_mask1)
            label2 = self.lm_fill(input2, label_mask2)
            output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = label1)
            output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = label2)
        else:
        
            output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = input1)
            output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = input2)

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

    # def zero_shot_classify(self, input1, input2, masks1, masks2):
    #     '''
    #     classify zero shot way 
    #     - here we calculate the log likelihood for the hypothesis 
    #     input 
    #     '''

    #     output1 = self.model(input1, labels =None,  **kwargs)
    #     output2 = self.model(input1, labels =None, **kwargs)

    #     label1 = input1[..., 1:].contiguous()
    #     label2 = input2[..., 1:].contiguous()

    #     logits1 = output1.hidden_states[...,:-1].contiguous()
    #     logits2 = output2.hidden_states[...,:-1].contiguous()

    #     ll1 = logits1[..., labels1.view(-1)]
    #     ll2 = logits2[..., labels2.view(-1)]

    #     ll1 = torch.sum(ll1*masks1, dim=-1) / torch.sum(masks1, dim=-1)
    #     ll2 = torch.sum(ll2*masks2, dim=-1) / torch.sum(masks1, dim=-1)

    #     return (ll1 < ll2).long()

class PretrainedDecoderTransformerDualSingleClassifier(nn.Module):
    '''
    author:  Sungjun Han

    This model will assume a dual-encoder archiecture 
    - language modelling auxilary objetive only for the encoder with the correct hypothesis! 
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
            segment_ids2):

        # using the mask and only calculate the language modelling for the correct 
        if self.label_mask:
            label_mask1 = labels
            label_mask2 = (labels == 0).long()
            label1 = self.lm_fill(input1, label_mask1)
            label2 = self.lm_fill(input2, label_mask2)
            output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = label1)
            output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = label2)
        else:
        
            output1 = self.model(input1, attention_mask = masks1, token_type_ids = segment_ids1, labels = input1)
            output2 = self.model(input2, attention_mask = masks2, token_type_ids = segment_ids2, labels = input2)

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