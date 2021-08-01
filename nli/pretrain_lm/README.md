
#  Pre-training Stage

The objective - further pretraining with next sentence language modelling objective 
hence, we are modelling the conditional probabilty distribution

2. J_LM = Sum_i Sum_t logp(hyp^i_t|obs^i,hyp^i_{<t} )

Finally, in order to evaluate these two pre-trainig objectives- we need to compare with the baseline - without any finetuning 


# Fine-tuning Stage

There are different ways of using the trained language model for classification. 

**side-note**. One way to fine-tune is to use the trained GPT-2 for classification similar to BERT. This model takes the same input 
representation as BERT but now we have the CLS token at the end - [obs1, obs2, hyp1, hyp2, [CLS]] - because GPT2 uses causal attention. A classifier head can be 
inserted on top  of  this CLS token for classificaiton. It also  has been noted that  language modelling objective also helps classification. 

    - however note that this formulation is very ill-suited when we include the language modleling objective, because this makes the model to model the 
    conditiona. probabtily distribution p(hyp1|hyp2) - and we see that Experimentally this way of **fine-tuning does not work**
    
Fine-tune for GPT-2 for classification is to use dual-encoder structure using [obs1, hyp, obs2 [CLS]] or [obs1, obs2, hyp [CLS]] input representation for each 
hypothesis and  this is inputted into the weight-tied GPT-2 encoder. The final [CLS] token representation is then transformed through a independent linear layers and normalized 
by softmax. We also introduce auxiliary objective of language modelling objective and only apply it for the correct hypothesis.

# Control codes : preprocessing
during pretraing and finetuning we make the identities of the observations (premises) and hypotheses explicit by prepending control codes:

    (observation1, observation2, hypothesis) becomes "before : <observation1> | after : <observation2> | hypothesis : <hypothesis>"

We found that such way of representing the input worked well compared to other representations


