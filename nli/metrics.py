#metrics.py 

import numpy 


def accuracy(y_true, y_pred):

	assert(len(y_true)==len(y_pred))
	acc = sum(1 if yt==yp else 0 for yt, yp in zip(y_true, y_pred) )/len(y_pred)

	return acc


def confusion_matrix(y_true, y_pred, axis=0):
	'''
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

