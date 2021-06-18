from transformers import GPT2LMHeadModel, 
import torch.nn as nn
import torch

class GptLmBase(nn.Module):
    '''
    this model will do next sentence prediction using the [cls] token 
    '''
    def __init__(self, 
                model):
        self.model = model 

    def forward(self, x):
        NotImplementedError

class GptLmDynamic(nn.Module):

    '''
    this model will do next sentence prediction using the [sep] token 
    '''
    
    def __init__(self,
                model):
        self.model = model

    def forward(self, x):
        NotImplementedError

