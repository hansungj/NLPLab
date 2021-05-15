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

# prepare training files 

for baseline models  - outputs in pickle file
```
python scripts/generate.py --model_type string
```

for deep learning models - outputs in h5 file 

```
python scripts/generate.py --model_type vector
```

# train a model

for baseline models 
```
python scripts/train.py --model_type BoW
```


