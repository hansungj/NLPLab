''' preprocess '''
from transformers import BertTokenizer
import torch
from datasets import load_dataset

def prepare_dataset(corpus):
	preceed = ''
	#dataset = []
	inputs = []
	outputs = []
	corpus = corpus['text']
	for sent in corpus:
		if preceed:
			#dataset.append((preceed, line.strip()))
			inputs.append(preceed)
			outputs.append(sent)
			preceed = sent
		else:
			preceed = sent
	dataset = (inputs, outputs)
	return dataset

def tokenize_and_mask(dataset, tokenizer = 'bert-base-uncased', max_length = 60, padding = 'max_length', masking_prob = 0.15):
	# tutorial by James Briggs used:
	# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
	
	tokenizer = BertTokenizer.from_pretrained(tokenizer)

	inputs = tokenizer(dataset[0], return_tensors='pt', max_length=max_length, truncation=True, padding=padding)
	outputs = tokenizer(dataset[1], return_tensors='pt', max_length=max_length, truncation=True, padding=padding)
	inputs['labels'] = outputs.input_ids.detach().clone()
	
	random_nums = torch.rand(inputs.input_ids.shape)
	#mask a token with prob of 15%, excluding CLS tokens (101), SEP tokens (102), PAD tokens (0)
	#generating a masking matrix 
	masking = (random_nums < masking_prob) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
		
	# which tokens have to be masked according to the masking matrix
	tokens_to_mask = []
	for i in range(masking.shape[0]):
		tokens_to_mask.append([])
		for j in range(masking.shape[1]):
			if masking[i][j] == True:
				tokens_to_mask[i].append(j)
	
	for i in range(inputs.input_ids.shape[0]):
		for j in range(len(tokens_to_mask[i])):
			inputs.input_ids[i][tokens_to_mask[i][j]] = 103
		
	return inputs, masking

def mask(inputs, max_length = 60, padding = 'max_length', masking_prob = 0.15):

	random_nums = torch.rand(inputs.input_ids.shape)
	#mask a token with prob of 15%, excluding CLS tokens (101), SEP tokens (102), PAD tokens (0)
	#generating a masking matrix 
	masking = (random_nums < masking_prob) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
		
	# which tokens have to be masked according to the masking matrix
	tokens_to_mask = []
	for i in range(masking.shape[0]):
		tokens_to_mask.append([])
		for j in range(masking.shape[1]):
			if masking[i][j] == True:
				tokens_to_mask[i].append(j)
	
	for i in range(inputs.input_ids.shape[0]):
		for j in range(len(tokens_to_mask[i])):
			inputs.input_ids[i][tokens_to_mask[i][j]] = 103
		
	return inputs

if __name__ == '__main__':
	dataset_bookcorpus = load_dataset('bookcorpus') #split='train'
	corpus = dataset_bookcorpus['train']
	print(len(corpus))
	print(corpus[4:9])
	corpus = corpus[4:9]
	dataset = prepare_dataset(corpus)
	inputs, masking = tokenize_and_mask(dataset)
	print(inputs)