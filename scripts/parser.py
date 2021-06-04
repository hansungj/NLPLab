
'''

prepare data into h5 file which can be used to run the models 

We will use this file for splitting for train/val split from train.tsv

'''
import numpy as np
import argparse
import os 
from sklearn.model_selection import train_test_split 

parser = argparse.ArgumentParser()

#directory 
parser.add_argument('--tsv_path', default='data/alphanli/tsv/train.tsv', type=str)
parser.add_argument('--output_dir', default='data/alphanli/tsv', type=str)
parser.add_argument('--suffix', default='_split')

#option 
parser.add_argument('--split', default=0.1, type=float)

def main(args):

	if not os.path.exists(args.output_dir):
		os.mkdirs(args.output_dir)

	data = open(args.tsv_path, 'r').readlines()

	train, dev = train_test_split(data, test_size = args.split, shuffle=True)


	with open(os.path.join(args.output_dir, 'train' + args.suffix + '.tsv'), 'w') as f:
		for l in train:
			f.write(l)

	with open(os.path.join(args.output_dir, 'dev' + args.suffix + '.tsv'), 'w') as f:
		for l in train:
			f.write(l)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)


# def main(args):

# 	assert(args.model_type in ['baseline', 'json', 'h5py'])

# 	data_path = os.path.join(args.data_dir, args.tsv_dir)
# 	dataset = utils.open_tsv_file(data_path)
	
# 	elif args.model_type == 'baseline':
# 			#build vocab
# 		if args.lemmatizer == 'wordnet':
# 			lemmatizer = WordNetLemmatizer()
# 		else:
# 			lemmatizer = None

# 		token2idx = preprocess.token_to_idx(
# 									dataset, 
# 									args.delimiter, 
# 									start_symbol = args.start_symbol, 
# 									end_symbol = args.end_symbol, 
# 									null_symbol = args.null_symbol,
# 									lemmatizer = lemmatizer
# 									)
# 		idx2token = preprocess.idx_to_token(token2idx)

# 		vocab = {
# 				'token2idx': token2idx,
# 				'idx2token': idx2token
# 				 }

# 		if args.tokenizer == 'wordnet':
# 			NotImplementedError
# 		else:
# 			tokenizer = preprocess.DefaultTokenizer(args.delimiter,
# 										vocab,
# 										lemmatizer)

# 		dataset_stories = {
# 		'obs':[],
# 		'hyp1':[],
# 		'hyp2':[],
# 		'label':[]
# 		}
# 		for i, (sid, obs1, obs2, hyp1, hyp2, label) in enumerate(dataset):
# 			obs1 = tokenizer.tokenize(obs1)
# 			obs2 = tokenizer.tokenize(obs2)
# 			hyp1 = tokenizer.tokenize(hyp1)
# 			hyp2 = tokenizer.tokenize(hyp2)
# 			obs = obs1 + obs2

# 			dataset_stories['obs'].append(obs)
# 			dataset_stories['hyp1'].append(hyp1)
# 			dataset_stories['hyp2'].append(hyp2)
# 			dataset_stories['label'].append( 0 if int(label) == 1 else 1 )

# 		#save vocab 
# 		with open(os.path.join(args.output_dir,'vocab.json'), 'w') as f:
# 			json.dump(vocab, f, indent=4)

# 		#save here
# 		output_path = os.path.join(args.output_dir, args.output_name + '.json')
# 		with open(output_path, 'wb') as f:
# 			json.dump(dataset_stories, f, indent=4)
	
# 	elif args.model_type == 'h5py':
# 		output_path = os.path.join(args.output_dir, args.output_name + '.h5')
# 		with h5py.File(output_path, 'w') as stories:
# 			num_examples = len(dataset)
# 			label_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
# 			ob_dataset = stories.create_dataset('observation', (num_examples, args.max_ob_len), dtype= np.int64)
# 			hyp1_dataset =stories.create_dataset('hypothesis1', (num_examples, args.max_hyp_len), dtype= np.int64)
# 			hyp2_dataset =stories.create_dataset('hypothesis2', (num_examples, args.max_hyp_len), dtype= np.int64)
# 			label_dataset =stories.create_dataset('label', (num_examples,),dtype= label_dtype)

# 			for i, (sid, obs1, obs2, hyp1, hyp2, label) in enumerate(dataset):

# 				if args.tokenizer == 'default':
# 					obs1 = tokenizer.tokenize(obs1)
# 					obs2 = tokenizer.tokenize(obs2)

# 					if args.add_split_token:
# 						observation = tokenizer.encode([obs1,obs2], max_len=args.max_ob_len, add_split_token=True)

# 					hyp1 = tokenizer.tokenize(hyp1)
# 					hyp1 = tokenizer.encode(hyp1, args.max_hyp_len)

# 					hyp2 = tokenizer.tokenize(hyp2)
# 					hyp2 = tokenizer.encode(hyp2, args.max_hyp_len)

# 					ob_dataset[i] = obs1
# 					hyp1_dataset[i] = hyp1
# 					hyp2_dataset[i] = hyp2
# 					label_dataset[i] = 0 if int(label) == 1 else 1  


# if __name__ == '__main__':
# 	args = parser.parse_args()
# 	main(args)