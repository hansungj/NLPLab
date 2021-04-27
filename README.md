# NLPLab

## Natural Language Inference 

Contributors:
- Sungjun Han 
- Anastasiia Sirotina

# setup
```
conda env create -f environment.yml
conda activate sysgen
pip install -e .
```

# evaluate 
```
python scripts/evaluate.py --label_dir 'data/alphanli/jsonl/train-labels.lst' --pred_label_dir 'data/alphanli/jsonl/train-labels.lst'
```
