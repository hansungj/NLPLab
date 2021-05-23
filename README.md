# NLPLab

## Natural Language Inference 

Contributors:
- Sungjun Han 
- Anastasiia Sirotina

# setup
```
conda env create -f environment.yml
conda activate nlplab
pip install -e .
```

# evaluate 
```
python scripts/evaluate.py --label_dir alphanli/jsonl/train-labels.lst --pred_dir alphanli/jsonl/train-labels.lst
```

# annotate
Try how good you are at this task!

```
python scripts/annotate.py --max_samples 30
python scripts/evaluate.py --label_dir annot_label.lst --pred_dir annot_pred.lst
cat eval_result.json
```


# train a model

for baseline models 
```
python scripts/train.py --model_type BoW

python scripts/train.py --model_type BoW --num_epochs 1 --output_name idf+lemmatization+bidirectional --bow_weight_function idf --bow_bidirectional True --bow_me_num_buckets 100 --bow_me_step_size 0.1

```


