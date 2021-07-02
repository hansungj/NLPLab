''' preprocess '''
''' drafts, deprecated'''
#from transformers import BertTokenizer
#import torch
#from datasets import load_dataset

#def prepare_dataset(corpus):
#	preceed = ''
#	dataset = []
#	corpus = corpus['text']
#	for sent in corpus:
#		if preceed:
#			dataset.append((preceed, sent))
#			preceed = sent
#		else:
#			preceed = sent
#	return dataset

#def tokenize_and_mask(dataset, max_length = 100, padding = 'max_length', masking_prob = 0.15):
	# tutorial by James Briggs used:
	# https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
	
#	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	#inputs = tokenizer(dataset[0], return_tensors='pt', max_length=max_length, truncation=True, padding=padding)
#	print(dataset[0])
#	inputs = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset[0]))
#	print(inputs)
#	print(tokenizer.cls_token_id)
#	outputs = tokenizer(dataset[1], return_tensors='pt', max_length=max_length, truncation=True, padding=padding)
#	inputs['labels'] = outputs.input_ids.detach().clone()
	
#	random_nums = torch.rand(inputs.input_ids.shape)
	#mask a token with prob of 15%, excluding CLS tokens (101), SEP tokens (102), PAD tokens (0)
	#generating a masking matrix 
#	masking = (random_nums < masking_prob) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
		
	# which tokens have to be masked according to the masking matrix
#	tokens_to_mask = []
#	for i in range(masking.shape[0]):
#		tokens_to_mask.append([])
#		for j in range(masking.shape[1]):
#			if masking[i][j] == True:
#				tokens_to_mask[i].append(j)
	
#	for i in range(inputs.input_ids.shape[0]):
#		for j in range(len(tokens_to_mask[i])):
#			inputs.input_ids[i][tokens_to_mask[i][j]] = 103
		
#	return inputs, masking

#def tokenize_n_mask(dataset, masking_prob = 0.15, ignore_index = -1):
	'''
	For a pair of sentences "He has an elder sister. Her name is Sarah." , we create the following input and output lists:
		input = 	CLS S1_1 S1_2 S1_3 S1_4 S1_5 S1_6 SEP IGN  IGN  IGN  IGN  IGN  IGN
		output = 	IGN IGN  IGN  IGN  IGN  IGN  IGN  IGN S2_1 S2_2 S2_3 S2_4 S2_5 EOS
	Input consist of a CLS token then indexes of the input's token a separation token and N times ignore_index token, where N-1 is the length of output
	Output consist of N times ignore_index token, where N-1 is the length of input, indexes of the input's token a separation token
	and an end os sentence token,
	Ignore_index is a parameter of torch.nn.CrossEntropyLoss that specifies a target value that is ignored and does not contribute to the input gradient. 
	For BERT model Ignore_index = -1.
	
	For performing masking tutorial by James Briggs was used:
	https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
	'''
#	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	
	
#	input_sent = dataset[0]
#	output_sent = dataset[1]
	
#	input = []
#	output = []
	
#	input.append(tokenizer.cls_token)
#	input += tokenizer.tokenize(input_sent)
#	input.append(tokenizer.sep_token)
	
#	random_nums = torch.rand(len(input))
	
	#mask a token with prob of 15%, excluding CLS tokens, SEP tokens, ignored tokens
	#generating a masking filter 
#	masking = (random_nums < masking_prob) * (input != tokenizer.cls_token) \
#		* (input != tokenizer.sep_token) * (input!= tokenizer.pad_token)
		
	# which tokens have to be masked according to the masking filter
#	tokens_to_mask = []
#	for i in range(len(masking)):
#		if masking[i] == True:
#				tokens_to_mask.append(i)
	
#	for token in tokens_to_mask:
#		input[token] = tokenizer.mask_token
		
#	input = tokenizer.convert_tokens_to_ids(input)
	
#	output = tokenizer.tokenize(output_sent)
#	if tokenizer.eos_token != None:
#		output.append(self.tokenizer.eos_token)
#	output = tokenizer.convert_tokens_to_ids(output)
	
#	input += [ignore_index] * len(output)
	
#	output = [ignore_index] * len(input) + output
#	return input, output

#if __name__ == '__main__':
#	dataset_bookcorpus = load_dataset('bookcorpus') #split='train'
#	corpus = dataset_bookcorpus['train']
#	print(len(corpus))
#	print(corpus[3:5])
#	corpus = corpus[3:5]
#	dataset = prepare_dataset(corpus)
#	print(dataset)
#	for datapair in dataset:
#		input, output = tokenize_n_mask(datapair)
#		print(input)
#		print(output)