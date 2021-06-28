'''
Author: Anastasiia 
Description: for pretraining BERT
'''
# tutorial by James Briggs used:
# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c

from transformers import BertConfig, BertForMaskedLM
from transformers import AdamW
from datasets import load_dataset

import numpy as np
import argparse
from tqdm import tqdm
import logging
import os


from nli.pretrain-mlm.BERT.bert-mlm import BertMLM

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)

argparse.add_argument('--corpus_name', default='bookcorpus', type=str)

parser.add_argument('--pretrained_name', default='bert-base-uncased', type=str, help='can be used to initialize a pretrained model from huggingface')
parser.add_argument('--max_samples_per_epoch', default=1000, type=int, help='Number of samples per epoch')

parser.add_argument('--use_cuda', default=False, type=bool, help = 'activate to use cuda')

#training 
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--shuffle', default=True, type=bool)

argparse.add_argument('--save_model_to', default='nli\pretrain-mlm\BERT\pretrained_model', type=str)

def main(args):

	utils.set_seed(args.seed)
	
	tokenizer = BertTokenizer.from_pretrained(args.pretrained_name)
	
	dataset_init = {'dataset': args.corpus_name,
					'max_samples': args.max_samples_per_epoch,
					'tokenizer' : tokenizer}
	
	train_dataset = MLM_Dataset(**dataset_init)
	
	train_loader = load_dataloader_transformer(
						train_dataset, 
						args.batch_size, 
						shuffle=args.shuffle, 
						drop_last = True, 
						num_workers = args.num_workers)
	
	model = BertMLM(args.pretrained_name)
	
	optim = AdamW(model.parameters(), lr=5e-5)

	if args.use_cuda:
		if not torch.cuda.is_available():
			print('use_cuda=True but cuda is not available')
		device = torch.device("cuda")
	else:
		device = torch.device('cpu')
	
	model.to(device)
	model.train()
	
	for epoch in range(epochs):

		loop = tqdm(train_loader, leave=True)
		
		for batch in loop:

			optim.zero_grad()

			input_ids = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			labels = batch['labels'].to(device)

			outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
			
			loss = outputs.loss

			loss.backward()

			optim.step()

			loop.set_description(f'Epoch {epoch}')
			loop.set_postfix(loss=loss.item())
	
	model.save_pretrained(args.save_model_to)
	
	