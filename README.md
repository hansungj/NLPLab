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
```
python scripts/evaluate.py --label_dir alphanli/jsonl/train-labels.lst --pred_dir alphanli/jsonl/train-labels.lst
```

# Annotate
Try how good you are at this task!

```
python scripts/annotate.py --max_samples 30
python scripts/evaluate.py --label_dir annot_label.lst --pred_dir annot_pred.lst
cat eval_result.json
```

# Build a vocabulary 

for deep learnign models 
```
python scripts/build_vocab.py --out_dir <VOCABULARY OUTPUT DIRECTORY>
```

you can choose between two types under --vocab_type

1. regular: 'reg'
2. Byte-Pair-Encoding: 'bpe'

you can also set vocabulary specifiction parameters 
1. --min_occurence : minimum occurence for a word type to be included in the vocabulary
2. --vocabulary_size : desired vocabulary size, selects the top frequent word types and filters out the rest 
# train a base line model

# Train baseline model : Bag of Words  

Note that for the baseline BoW model with Maximum entropy classifier - there is no need to run the model more than one epoch.
```
python scripts/train.py --model_type BoW

python scripts/train.py --model_type BoW --num_epochs 1 --output_name idf+lemmatization+bidirectional --bow_weight_function idf --bow_bidirectional True --bow_me_num_buckets 100 --bow_me_step_size 0.1

```

# Train baseline DL models : FFN/RNN/CNN encoder with FFN decoder 

These models use pre-trained embeddings 

FFN
```
python scripts/train.py --model_type StaticEmb-mixture
```

RNN
```
python scripts/train.py --model_type StaticEmb-RNN
```

CNN
```
python scripts/train.py --model_type StaticEmb-CNN
```
