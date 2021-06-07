# NLPLab

## Natural Language Inference 

Contributors:
- Sungjun Han 
- Anastasiia Sirotina

# Setup
```
conda env create -f environment.yml
conda activate nlplab
pip install -e .
```

# Evaluate 

This allows you to evaluate on predictions written on a file.  
```
python scripts/evaluate.py --label_dir <PROVIDE GROUND TRUTH LABEL FILE PATH> --pred_dir <PROVIDE REDICTION FILE PATH>
```


# Annotate
Try how good you are at this task! This allows you to annotate --max_samples number of randomly selected aNLI data points. 

```
python scripts/annotate.py --max_samples 30 --annot_pred <PROVIDE OUTPUT PATH HERE>
python scripts/evaluate.py --label_dir <PROVIDE OUTPUT PATH FOR ANNOTATION HERE> --pred_dir  <PROVIDE GROUND TRUTH LABEL FILE PATH>
cat eval_result.json
```

# Build a vocabulary 

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
# train a base line model

# Split test into 90/10 Val/Test

Since testing out the prediction on the unknown test set requires submitting the results on the public leaderboard along with project description (and only allowed every 7 days ), we will use the split the provided dev set into a test set and a validation set using 90/10 split. This can be done by running parser.py

```
python scripts/parser.py \
--suffix _split \
- split 0.1 \

```

This splits val.tsv into two files: val_split.tsv and test_split.tsv

# Train baseline model : Bag of Words - Maximum Entropy using IDF / Levenshtein / Lemmatization 

Note that for the baseline BoW model with Maximum entropy classifier - there is no need to run the model more than one epoch.
```
python scripts/train.py \
--model_type BoW \
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

# Train baseline DL models : FFN/RNN/CNN encoder with FFN decoder 

These models use pre-trained embeddings 

### FFN - train for 50 epochs using SUM pooling method 

this can also be run using early stopping by setting early stopping > 0 

To replicate the baseline accuracy score of 52.73%, run
```
!python scripts/train.py \
--train_tsv <PROVIDE TRAIN TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/train.tsv>
--val_tsv <PROVIDE VAL TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/val_split.tsv>
--test_tsv <PROVIDE TEST TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/test_split.tsv>
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
!python scripts/train.py \
--model_type StaticEmb-rnn \
--train_tsv <PROVIDE TRAIN TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/train.tsv>
--val_tsv <PROVIDE VAL TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/val_split.tsv>
--test_tsv <PROVIDE TEST TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/test_split.tsv>
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

### CNN
```
python scripts/train.py --model_type StaticEmb-CNN
```

### BERT with CLS token 

this can also be run using early stopping by setting early stopping > 0. We use huggingface implementation for pretrained transformer models.

To replicate the baseline accuracy score of 61.82%, run 

```
python scripts/train.py \
--model_type pretrained-transformers-cls \
--pretrained_name bert-base-uncased \
--train_tsv <PROVIDE TRAIN TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/train.tsv>
--val_tsv <PROVIDE VAL TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/val_split.tsv>
--test_tsv <PROVIDE TEST TSV DIRECTORY HERE - IF NOT PROVIDED WILL BE ASSUMED data/alphanli/tsv/test_split.tsv>
--batch_size 128 \
--early_stopping 0 \ 
--num_epochs 10 \ 
--evaluate True \
--learning_rate 1e-5 \
--use_cuda True \
--scheduler True \
--weight_decay 0.0 \ 
--seed 1234 \
```
