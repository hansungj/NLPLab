# Sentence Level Pretraining for Natural Language Inference 
### Abstract 

In this paper we discuss a novel approach to improve the performance of transformer-based models for Natural language inference (NLI) task. We hypothesize the knowledge of the relationships between the premise and the hy- potheses needed to be successful in NLI can be extracted from an unannotated corpus in a self- supervised manner. We propose two training objectives to achieve this: Sentence Level Lan- guage Modelling (SL-LM) and Sentence Level Masked Language Modelling (SL-MLM)1. To show the conceptual validity of this hypothesis we compare performance of transformer-based models with pretraining to the non-pretrained models on a chosen NLI task.

See [paper](https://www.google.com) here.
Contributors:
- Sungjun Han 
- Anastasiia Sirotina

# Repository Structure 
- **scripts** : all runnable python scripts (i.e. train, test, build vocabulary ...) are in this folder
    - scripts/**run_scripts** : all .sh scripts for running gpt2/bert training
    - scripts/**baseline** : all train/test python scripts for BoW and DL baseline models
        - scripts/baseline/**baseline_train_test.py** : train/test python script for the BoW baseline models 
        - scripts/baseline/**dl_baseline_train_test.py** : train/test python script for the DL baseline models 
    - scripts/**bert** : all train/test python scripts for pretraining/fine-tuning BERT
        - scripts/baseline/**advanced_train_test.py** : train/test python script for the BERT based models 
        - scripts/baseline/**pretrain_mlm.py** : train/test python script for the further pretraining BERT 
    - scripts/**gpt2** : all train/test python scripts for pretraining/fine-tuning GPT-2
        - scripts/baseline/**gpt2test.py** : test python script for GPT-2
        - scripts/baseline/**gpt2train.py** : train (fine-tuning) python script for GPT-2
        - scripts/baseline/**pretrain-lm-aux.py** : pretraining (fine-tuning) python script for GPT-2 with auxiliary pretraining objective 
        - scripts/baseline/**preetrain-lm.py** : pretraining (fine-tuning) python script for GPT-2 without auxiliary pretraining objective 
    - scripts/**annotate.py** :  creating manual human annotations for ART
    - scripts/**build_vocab.py** :  building vocabulary for the baselines
- **nli** : auxiliary functions used by the files in scripts
    - nli/**models** : all models are defined in this directory (baseline/GPT2/BERT)
        - nli/models-lm/**BoW.py** : all python object classes for BoW baseline model
        - nli/models-lm/**GPT2.py** : all PyTorch nn.Module classes for GPT2 models 
        - nli/models-lm/**StaticEmb.py** : all PyTorch nn.Module classes for DL baseline models
        - nli/models-lm/**Transformers.py** : all PyTorch nn.Module classes for transformer baseline models  
    - nli/**pretrain-mlm** : all functions used by BERT training/testing for pretraining 
        - nli/pretrain-mlm/**dataloader.py** : holds PyTorch datasset and dataloader objects for pretraining for BERT
    - nli/**pretrain-lm** : all models (baseline/GPT2/BERT)
        - nli/pretrain-lm/**ft_dataloader.py** : holds PyTorch datasset and dataloader objects for finetuning for GPT-2
        - nli/pretrain-lm/**pt_dataloader.py** : holds PyTorch datasset and dataloader objects for pretraining for GPT-2
    - nli/**dataloader.py** : holds PyTorch datasset and dataloader objects for preparing ART for baselines and BERT
    - nli/**embedding.py** : loads GloVe embeddings to be used by DL baselines
    - nli/**metrics.py** : holds an object class that is used to keep evaluation results during training/testing
    - nli/**preprocess.py** : functions used for preprocessing the dataset
    - nli/**similarity.py** : distance and similarity measures for BoW baseline
    - nli/**tokenization.py** : holds a tokenizer object class used by DL baselines
    - nli/**utils.py** : various utility functions 

## Setup
```
conda env create -f environment.yml
conda activate nlplab
pip install -e .
```
Make sure to have a data folder with the aNLI data present: data/alphali/...
<!-- 
## Evaluate 

This allows you to evaluate on predictions written on a file.  
```
python scripts/evaluate.py --label_dir <PROVIDE GROUND TRUTH LABEL FILE PATH> --pred_dir <PROVIDE REDICTION FILE PATH>
``` -->


## Annotate
Try how good you are at this task! This allows you to annotate --max_samples number of randomly selected aNLI data points. 

```
python scripts/annotate.py --max_samples 30 --annot_pred <PROVIDE OUTPUT PATH HERE>
python scripts/evaluate.py --label_dir <PROVIDE OUTPUT PATH FOR ANNOTATION HERE> --pred_dir  <PROVIDE GROUND TRUTH LABEL FILE PATH>
cat eval_result.json
```

## Build a vocabulary 

for deep learnign models - baseline models, we need to make a vocabulary to initilaize our WhiteSpaceTokenizer. For the pre-trained models, we will just use the availble pretrained sub-word tokenizers. 
```
python scripts/build_vocab.py --out_dir <VOCABULARY OUTPUT DIRECTORY>
```

you can choose between two types under --vocab_type

1. regular: 'reg'
2. Byte-Pair-Encoding: 'bpe' #not implemented yet 

you can also set vocabulary specifiction parameters 
1. --min_occurence : minimum occurence for a word type to be included in the vocabulary
2. --vocabulary_size : desired vocabulary size, selects the top frequent word types and filters out the rest 
# Train a base line model

<!-- ## Split Dev into 70/30 Test/Val

Since testing out the prediction on the unknown test set requires submitting the results on the public leaderboard along with the project description (and submission is only allowed every 7 days), we will use the provided dev set for testing by splitting it into a test set and a validation set with random 90/10 split. The reason why we split on dev.tsv, not on train.tsv, is that the answer distribution seems to be very different between the train and dev. Thus using a validation set created from splitting train will result will not work for monitoring the model's generalization capabiltiy. 

We have uploaded the split that we have used for our experiments 
- data/alphanli/tsv/val_split.tsv # validation 
- data/alphanli/tsv/test_split.tsv # test

It can be also be newly created by running parser.py
```
python scripts/parser.py \
--suffix _split \
--split 0.3 \
```
This splits val.tsv into two files: val_split.tsv and test_split.tsv at data/alphanli/tsv -->

## Train baseline model : Bag of Words 
Note that for the baseline BoW model with Maximum entropy classifier - there is no need to run the model more than one epoch.
###  Perceptron using Levenshtein
to run a model that scored 50.97%, 

```
python scripts/baseline/baseline_train_test.py \
--model_type BoW \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--bow_classifier prc \ 
--num_epochs 1 \
--bow_sim_function levenshtein \
--bow_weight_function idf \
--bow_max_cost 100 \ 
--bow_lemmatize True \
--bow_bidirectional False \
--bow_me_num_buckets 30 \
--bow_me_step_size 0.1 \

```
###  Perceptron using Distributional
to run a model that scored 50.79%, 

```
python scripts/baseline/baseline_train_test.py \
--model_type BoW \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--bow_classifier prc \ 
--num_epochs 1 \
--bow_sim_function distributional \
--bow_weight_function idf \
--bow_max_cost 100 \ 
--bow_lemmatize True \
--bow_bidirectional False \
--bow_me_num_buckets 30 \
--bow_me_step_size 0.1 \

```
###  Maximum Entropy using IDF / Levenshtein / Lemmatization 
to run a model that scored 50.13%, 

```
python scripts/baseline/baseline_train_test.py \
--model_type BoW \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--bow_classifier maxent \ 
--num_epochs 1 \
--bow_sim_function levenshtein \
--bow_weight_function idf \
--bow_max_cost 100 \ 
--bow_lemmatize True \
--bow_bidirectional False \
--bow_me_num_buckets 100 \
--bow_me_step_size 0.1 \

```
###  Maximum Entropy using Distributional 
to run a model that scored 51.52%, 

```
python scripts/baseline/baseline_train_test.py \
--model_type BoW \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--bow_classifier maxent \ 
--num_epochs 1 \
--bow_sim_function distributional \
--bow_weight_function idf \
--bow_max_cost 100 \ 
--bow_lemmatize True \
--bow_bidirectional False \
--bow_me_num_buckets 30 \
--bow_me_step_size 0.1 \

```

## Train baseline DL models : FFN/RNN/CNN encoder with FFN decoder 

These models use pre-trained embeddings 

### FFN - train for 50 epochs using SUM pooling method 

this can also be run using early stopping by setting early stopping > 0 

To replicate the baseline accuracy score of 52.73%, run
```
python scripts/baseline/dl_baseline_train_test.py \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--model_type StaticEmb-mixture \
--sem_pooling sum \
--use_cuda True \
--batch_size 128 \
--learning_rate 5e-4 \
--optimizer adam \ 
--se_num_encoder_layers 3 \
--se_num_decoder_layers 3 \
--glove_model glove-wiki-gigaword-50 \
--evaluate true \
--early_stopping 0 \
--num_epochs 50 \
--seed 1234 \
```

### RNN - train for 50 epochs 

this can also be run using early stopping by setting early stopping > 0 

To replicate the baseline accuracy score of 55.10%, run
```
python scripts/baseline/dl_baseline_train_test.py \
--model_type StaticEmb-rnn \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \  
--use_cuda True \
--batch_size 128 \
--learning_rate 5e-4 \
--optimizer adam \ 
--se_num_encoder_layers 2 \
--se_num_decoder_layers 3 \
--glove_model glove-wiki-gigaword-50 \
--sernn_bidirectional true \
--evaluate true \
--early_stopping 0 \
--num_epochs 50 \
--seed 1234 \
```

### CNN - train for 100 epochs 

this can also be run using early stopping by setting early stopping > 0 

To replicate the baseline accuracy score of 56.13%, run
```
python scripts/baseline/dl_baseline_train_test.py \
--model_type StaticEmb-cnn \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--use_cuda True \
--batch_size 128 \
--learning_rate 1e-4 \
--optimizer adam \ 
--glove_model glove-wiki-gigaword-50 \
--evaluate true \
--early_stopping 0 \
--num_epochs 100 \
--seed 1234 \
```

### BERT with CLS token 

this can also be run using early stopping by setting early stopping > 0. We use huggingface implementation for pretrained transformer models.

To replicate the baseline accuracy score of 61.82%, run 

```
python scripts/baseline/advanced_train_test.py \
--model_type pretrained-transformers-cls \
--pretrained_name bert-base-uncased \
--train_tsv data/alphanli/tsv/train.tsv \ 
--test_tsv data/alphanli/tsv/dev.tsv \ 
--batch_size 128 \
--early_stopping 0 \ 
--num_epochs 15 \ 
--evaluate True \
--learning_rate 1e-5 \
--use_cuda True \
--scheduler True \
--weight_decay 0.0 \ 
--seed 1234 \
```
<!-- 
### BERT Based Dual Encoder with a single CLS classifier head

We use huggingface implementation for pretrained transformer models.

To replicate the baseline accuracy score of 75%, run 

```
python scripts/advanced_train_test.py \
--model_type dual_enc_bert \
--pretrained_name bert-base-uncased \
--train_tsv data/alphanli/tsv/train.tsv \ 
--val_tsv data/alphanli/tsv/val_split.tsv \ 
--test_tsv data/alphanli/tsv/test_split.tsv \ 
--batch_size 8 \
--early_stopping 0 \ 
--num_epochs 5 \ 
--evaluate True \
--learning_rate 1e-5 \
--use_cuda True \
--weight_decay 0.0 \ 
--seed 1234 \
``` -->

# Pretrain

all pretrainig configurations are kept in .sh files - **scripts/run_scripts/**

## GPT-2 Sentence Level Language Modelling (SL-LM)

For multi-gpu:
```
scripts/run_scripts/pretrain-lm-multi-gpu.sh
```

For single-gpu:
```
scripts/run_scripts/pretrain-lm-single-gpu.sh
```

## BERT Sentence Level Masked Language Modelling (SL-MLM)

## GPT-2 Fine-tuning 

Training with SL-LM pretrained model 
```
scripts/run_scripts/gpt2-dual-single-gpu.sh --from_pretrained <INSERT_PRETRAINED_PATH>
```

Training without SL-LM pretrained model 
```
scripts/run_scripts/gpt2-dual-single-gpu.sh --from_pretrained None
```

## BERT Fine-tuning 

Training with SL-MLM pretrained model 
```
scripts/run_scripts/bert-dual.sh 
```

Training without SL-MLM pretrained model 
```
scripts/run_scripts/bert-dual-pretrained.sh 
```
