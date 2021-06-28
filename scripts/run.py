'''
Author: Anastasiia 
Description: running a trained model for prediction 

Deprecated 
'''

# import nli.utils as utils 
# from nli.data import AlphaDataset
# import nli.preprocess as preprocess
# import nli.models.BoW as bow

# parser = argparse.ArgumentParser()

# #directory for data/train/val
# parser.add_argument('--data_dir', default='data', type=str)
# parser.add_argument('--train_tsv', default='alphanli/tsv/train.tsv', type=str)
# parser.add_argument('--val_tsv', default='alphanli/tsv/dev.tsv', type=str))
# parser.add_argument('--vocab', default=None)

# parser.add_argument('--sim_function' default='levenshtein', type=str))
# parser.add_argument('--weight_function' default=None, type=str))
# parser.add_argument('--classifier', default=None, type=str))
# parser.add_argument('--max_cost ' default=100, type=int))
# parser.add_argument('--bidirectional ' default=False, type=bool))

# #directory for annotation output
# parser.add_argument('--annot_pred', default='annot_pred.lst', type=str)
# parser.add_argument('--annot_label', default='annot_label.lst', type=str)

# parser.add_argument('--max_samples', default=1000, type=int)

# def main():

# 	tsv_path = os.path.join(args.data_dir, args.train_tsv)

# 	vocab = args.vocab
# 	if vocab is not None:
# 		vocab = os.path.join(args.data_dir,vocab)
# 	max_samples = args.max_samples

# 	if max_samples is None:
# 		raise ValueError('Please provide number of samples')
	
# 	dataset = AlphaDataset(tsv_path,
# 				 		   vocab=vocab,
# 				 		   max_samples=max_samples,
# 				 		   annotate=True)
    
#     token2idx = preprocess.token_to_idx(dataset)
#     idx2tok = preprocess.idx_to_token(tok2idx)
    
#     model = BagOfWords(
# 				 args.vocab,
# 				 args.sim_function,
# 				 args.weight_function,
# 				 args.classifier,
# 				 args.max_cost,
# 				 args.bidirectional,
# 				 )
    
#     #dataset_preprocessed = []
# 	labels = []
#     predictions = []
    
#     for i in range(len(dataset.label)):
# 		obs1, obs2, hyp1, hyp2, gold_label = dataset[i]
#         p = preprocess.encode(obs1) + preprocess.encode(obs2)
#         h1 = preprocess.encode(hyp1)
#         h2 = preprocess.encode(hyp2)
#         labels.append(label)

        
#         pred_label = model.inference(h1, h2, p)
#         #dataset_preprocessed.append((p, h1, h2, gold_label, pred_label))
#         predictions.append(pred_label)         

#     with open(os.path.join(args.data_dir, args.annot_pred), 'w') as f:
# 		for l in predictions:
# 			f.write('{}\n'.format(l))

# 	#write to label path
# 	with open(os.path.join(args.data_dir, args.annot_label), 'w') as f:
# 		for l in labels:
# 			f.write('{}\n'.format(l))
            

# if __name__ == '__main__':
# 	args = parser.parse_args()
# 	main()
