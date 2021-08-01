<!-- 
#  Pre-training Stage

The objective - further pretraining with next sentence language modelling objective 
hence, we are modelling the conditional probabilty distribution

1. J_NSLM = Sum_k logp(hyp k-th word|hyp < k, observation )

In order to test the effectivenss of this - we also need to compare with further pre-training  with regular language modleling objective 

2. J_LM = Sum_k logp(hyp/obs k-th word|hyp/obs < k )

Finally, in order to evaluate these two pre-trainig objectives- we need to compare with the baseline - without any finetuning 

3. using GPT-2 model without any further pre-training 

my hypothesis is that  even thought (2) is essentially the same as (3) , the corpus which was used to train GPT-2 is different then the BookCorpus.


# Fine-tuning Stage

There are different ways of using the trained language model for classification. 

1. The most interesting perspective is to test the ability of the model's zero shot-capabiltiy. This can be implemented through comparing the probabilty 
assigned to the [obs1 hyp obs2] for each hypothesis and choosing  the hypothesis that scored higher by the languagem model. This can be further made sophisticated
through introducing various features: p(hyp), p(obs1), p(obs2),.... and using these as features for a logistic classifier

2. another way to fine-tune is to use the trained GPT-2 for classification similar to BERT. This model takes the same input 
representation as BERT but now we have the CLS token at the end - [obs1, obs2, hyp1, hyp2, [CLS]] - because GPT2 uses causal attention. A classifier head can be 
inserted on top  of  this CLS token for classificaiton. It also  has been noted that  language modelling objective also helps classification. 

    - however note that this formulation is very ill-suited when we include the language modleling objective, because this makes the model to model the 
    conditiona. probabtily distribution p(hyp1|hyp2) 
    - hence this model should NOT use language modelling objective as an auxilary objective  - or only apply language modleling objective for the observations

3. final way to fine-tune the GPT-2 for classification is to use dual-encoder structure using [obs1, hyp, obs2 [CLS]] or [obs1, obs2, hyp [CLS]] input representation for each 
hypothesis and  this is inputted into the weight-tied GPT-2 encoder. The final [CLS] token representation is then transformed through a independent linear layers and normalized 
by softmax ( this is essentially 3). We could possibly use language modelling objective and only apply it for the correct hypothesis - if needed.


# Hyperparameter optimization
Another consideration to make is to consider the representation format - during pre-training and fine-tuning

1. the order of hypothesis and observations, [observation1, observation2, hypothesis] vs [observation1, hypothesis, observation2]
2. use of delimiters:
    - using simple delimiters and no delimiters between the observations 
    - making the identities of the observations known: "observation 1 : {observation 1} observation 2 : {observation 2} hypothesis : {hypothesis}", 
    or to use simple delimiters such as $ or to introduce new separation token such as [SEP]

 -->
