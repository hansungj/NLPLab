
#  Pre-training Stage

## Sentence Level Masked Language Modelling

The objective - further pretraining with the objective of modelling a masked target sentence (Sentence Level Masked Language Modelling).

For a target sentence s and its two context sentences c_l and c_r (one to the left and one to the right), the following probability p(s|c_l, c_r) can be computed. Within SL-MLM objective:

p(s|c_l, c_r) = Mul_{s_i in M} p(s_i|c_l_1, ..., c_l_|l|, c_r_1, ..., c_r_|r|, s not in M)

where M is a set of the masked tokens.

## Data preparation

For pretraining we take book texts and prepare it in the following way.
Each datapoint is a tuple (input, target). The input is structured as follows: a CLS token, tokenized c_l, a separation token, tokenized c_r, a separation token, tokenized and mask s, EOS token. The target is of the same length as the input filled with special ignoring tokens IGN expect for the positions at which tokens in the target sentence were masked, those positions are filled with the initail tokens of the target sentence.

Example:
c_l = I have an elder sister.
s = Her name is Sarah.
c_r = She is nice.

input = [ [CLS], CL1, CL2, CL3, CL4, CL5, CL6, [SEP], CL1, CL2, CL3, CL4, CL5, [SEP], S1, [MASK], S3, [MASK], [EOS] ]
target = [ [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], [IGN], S2, [IGN], S4, [IGN] ]

IGN token is needed to ensure that the loss is calculated for the predicted masked tokens only.

## Masking approaches
 We randomly choose 30% of all the tokens for masking. For each of the chosen tokens one of the following three actions is taken: 
 1) the token is replaced with a masking token [MASK] with probability of 80%;
 2) the token is replaced with another token, randomly chosen from the dictionary with probability of 10%; 
 3) the token is left unchanged with probability of 10%.

Another, deprecated version is as follows: each token is replaced with a masking token [MASK] with probability of 15%.
# Fine-tuning Stage
The further pretrained model is supposed to be used in 2 different ways:
1. Replacing BERT base model (made freely available by Huggingface) within our BERT-concat model (see nli/model/Transformers.py)

2. Replace BERT base model (made freely available by Huggingface) within our BERT-concat model (see nli/model/BertBasedDualEncoder.py)


