'''
Author: Sungjun Han
Description: can be used to evaluate from a saved prediction file  
'''

import argparse 
import numpy 
import logging 
import json
import os


import nli.utils as utils  
import nli.metrics as metrics 

parser = argparse.ArgumentParser()

#paths
parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument("--label_dir", default='alphanli/jsonl/train-labels.lst', type=str)
parser.add_argument('--pred_dir', default=None, type=str)
parser.add_argument('--eval_pth', default='', type=str)

#eval metric to use 
parser.add_argument('--eval_acc', default=True,type=bool)
parser.add_argument('--eval_precision', default=False,type=bool)
parser.add_argument('--eval_recall', default=False,type=bool)
parser.add_argument('--eval_f1score', default=False,type=bool)
parser.add_argument('--eval_f1score_beta', default=1,type=float)



logger = logging.getLogger(__name__)

def main(args):

	#get path
	label_path = os.path.join(args.data_dir,args.label_dir)
	pred_path = os.path.join(args.data_dir,args.pred_dir)

	y_true = utils.open_label_file(label_path)
	y_pred = utils.open_label_file(pred_path)
	
	result = {}

	if args.eval_acc:
		acc = metrics.accuracy(y_true, y_pred)
		result['accuracy'] = acc

	if args.eval_precision:
		precision = metrics.precision(y_true, y_pred)
		result['precision'] = precision

	if args.eval_recall:
		recall = metrics.recall(y_true, y_pred)
		result['recall'] = recall

	if args.eval_f1score:
		f1score = metrics.f1score(y_true, y_pred, args.eval_f1score_beta)
		result['f1score'] = f1score


	with open(args.eval_pth + 'eval_result.json', 'w') as f:
		json.dump(result, f, indent=2)


if __name__ == "__main__":
	args = parser.parse_args()
	main(args)