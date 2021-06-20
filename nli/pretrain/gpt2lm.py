from transformers import GPT2LMHeadModel, 
import torch.nn as nn
import torch

class GptLmBase(nn.Module):
    '''
    this model will do next sentence prediction using the [cls] token 
    '''
    def __init__(self, 
                model,
                config):
        self.model = model 
        self.cls_classifier = nn.Sequential(
			nn.Linear(config.hidden_size, config.hidden_size),
			nn.GELU(),
			nn.Dropout(config.hidden_dropout_prob),
			nn.Linear(config.hidden_size, config.num_emotion))
        
        self.cls_loss_fn=nn.CrossEntropyLoss()
        self.lm_loss_fn= nn.CrossEntropyLoss()

    def forward(self, **inputs):
        '''
        make sure that the 
        1. target
        2. nsp_label are part of 
        '''
        
        targets = inputs.pop('targets')
        nsp_label = inputs.pop('nsp_label')
        x = self.model(**inputs)
        hidden_states = x.last_hidden_states

        #classifier for next sentence prediction
        cls = hidden_states[:,0,:]
        cls_logits = self.cls_classifier(cls)
        cls_loss = self.cls_loss_fn(cls_logits.view(-1), nsp_label)

        #language modelling 
        lm_loss = self.lm_loss_fn(hidden_states, targets)

        return lm_loss, cls_loss, cls_logits

class GptLmDynamic(nn.Module):

    '''
    this model will do next sentence prediction using the [sep] token 
    '''
    
    def __init__(self,
                model):
        self.model = model

    def forward(self, x):
        NotImplementedError

