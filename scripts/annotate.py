
import argparse
import random
import os

import nli.utils as utils 
from nli.data import AlphaDataset

parser = argparse.ArgumentParser()

#directory for data/train/val
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--train_tsv', default='alphanli/tsv/train.tsv', type=str)
parser.add_argument('--val_tsv', default='alphanli/tsv/dev.tsv', type=str)
parser.add_argument('--vocab', default=None)

#directory for annotation output
parser.add_argument('--annot_pred', default='annot_pred.lst', type=str)
parser.add_argument('--annot_label', default='annot_label.lst', type=str)

#annotation variables
parser.add_argument('--max_samples', default=None, type=int)
parser.add_argument('--show_label', default=False, type=bool)

def main():

	tsv_path = os.path.join(args.data_dir, args.train_tsv)

	vocab = args.vocab
	if vocab is not None:
		vocab = os.path.join(args.data_dir,vocab)
	max_samples = args.max_samples

	if max_samples is None:
		raise ValueError('Please provide number of samples')
	
	dataset = AlphaDataset(tsv_path,
				 		   vocab=vocab,
				 		   max_samples=max_samples,
				 		   annotate=True)

	data_choices = range(len(dataset.label))
	predictions = []
	labels = []
	for i in range(max_samples):
		idx = random.choice(data_choices)
		obs1, obs2, hyp1, hyp2, label = dataset[idx]

		labels.append(label)

		print('Annotation example',i, '-'*50)
		print('Observation 1:', obs1)
		print('Observation 2:', obs2)
		print('Press Enter to see the hypotheses')
		input()

		print('Hypothesis 1:', hyp1)
		print('Hypothesis 2:', hyp2)

		print('Please select "1" or "2" and Press Enter')

		while True:
			pred = input()
			if pred in ['1', '2']:
				break
			print('Please select from "1" or "2"')

		print('\n')

		predictions.append(pred)

	#write to prediction path
	with open(os.path.join(args.data_dir, args.annot_pred), 'w') as f:
		for l in predictions:
			f.write('{}\n'.format(l))

	#write to label path
	with open(os.path.join(args.data_dir, args.annot_label), 'w') as f:
		for l in labels:
			f.write('{}\n'.format(l))


if __name__ == '__main__':
	args = parser.parse_args()
	main()