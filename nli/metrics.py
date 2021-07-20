#metrics.py 

import numpy as np
from collections import defaultdict


class MetricKeeper(object):

	'''
	Author: Sungjun Han
	Description: Class for holding and evaluting evaluation metrics 
	'''

	def __init__(self, eval_measures=None):
		self.keeper = defaultdict(list)
		self.eval_functions = []
		if eval_measures:
			assert(isinstance(eval_measures, list))
			for eval_name in eval_measures:

				if eval_name == 'accuracy':
					self.eval_functions.append((accuracy, 'accuracy'))
				elif eval_name == 'precision':
					self.eval_functions.append((precision, 'precision'))
				elif eval_name == 'recall':
					self.eval_functions.append((recall, 'recall'))
				elif eval_name == 'fscore':
					self.eval_functions.append((fscore, 'fscore'))

	def eval(self, y, y_pred):
		for eval_f, eval_n in self.eval_functions:
			eval_r = eval_f(y, y_pred)
			self.keeper[eval_n].append(eval_r)

	def update(self, eval_name, eval_r):
		#manually update 
		self.keeper[eval_name].append(eval_r)

	def print(self):
		for key, val in self.keeper.items():
			print('{}={}'.format(key,val[-1]))

def log_likelihood(y, y_pred, k=2):
	'''
	assume that y_pred has dim [N,k]
	'''
	epsilon = 1e-8
	if k==2:
		y_ = np.zeros((len(y),k))
		y_[y] = 1
	# y_pred's are probabilities 
	L = -np.sum(y_*np.log(y_pred+epsilon)) #/len(y)
	return L


def accuracy(y_true, y_pred):
	'''
	Author: Sungjun Han
	Description: calcalates accuracy 
	'''

	assert(len(y_true)==len(y_pred))
	acc = sum(1 if yt==yp else 0 for yt, yp in zip(y_true, y_pred) )/len(y_pred)

	return acc


def confusion_matrix(y_true, y_pred, axis=0):
	'''
	Author: Sungjun Han
	Description: computes confusion matrix 
	
	axis=0 computes ground in terms of hypothesis 1 
	axis=1 computes ground in terms of hypothesis 2 

	'''
	C = [[0]*2 for _ in range(2)]

	assert(axis in [0, 1])

	cvt = lambda x : 1 if x == 0 else 0 

	for yt, yp in zip(y_true, y_pred):
		if yt == yp:
			if yt == axis+1:
				C[axis][axis] += 1
			else:
				C[cvt(axis)][cvt(axis)] += 1

		else:
			if yt == axis+1:
				C[axis][cvt(axis)] += 1
			else:
				C[cvt(axis)][axis] += 1
	
	return C 

def precision(y_true, y_pred, C=None):
	
	#raise NotImplementedError
	if C is None:
		C = confusion_matrix(y_true, y_pred)
	prec = C[0][0] / (C[0][0] + C[1][0])
	
	return prec

def recall(y_true, y_pred, C=None):
	#raise NotImplementedError
	if C is None:
		C = confusion_matrix(y_true, y_pred)
	rec = C[0][0] / (C[0][0] + C[0][1])
	
	return rec

def fscore(y_true, y_pred, beta = 1):
	#raise NotImplementedError
	# prec = C[0][0] / (C[0][0] + C[1][0])
	# rec = C[0][0] / (C[0][0] + C[0][1])

	C = confusion_matrix(y_true,y_pred)
	prec = precision(None,None,C)
	rec = recall(None,None,C)
	
	F1 = (1+beta**2) * prec * rec / (prec*beta**2 + rec)
	
	return F1

