import numpy as np
from collections import defaultdict

from nli.similarity import levenshtein


def sigmoid(x):
	return np.exp(x)/np.sum(np.exp(x))

def idf(corpus):
	'''
	build idfs
	
	'''
	d = {}
	N = len(corpus)
	for doc_id, doc in enumerate(corpus):
		for token in line:
			if token not in d:
				d[token] = [1, doc_id]

			elif doc_id != d[token][1]:
				d[token][0] += 1
				d[token][doc_id] = doc_id

	d = {word:-np.log(doc_freq/N) for word, doc_freq  in d.items()}
	return d


class BagOfWords(object):

	def __init__(self,
				 vocab,
				 sim_function='levenshtein',
				 weight_function=None,
				 classifier_type = None,
				 max_cost = 100,
				 bidirectional = False,
				 ):


		self.vocab = vocab
		self.sim_function = sim_function
		self.weight_function = weight_function
		self.classifier_type = classifier_type
		self.bidirectional = bidirectional # consider cost(p|h) and cost(h|p)
		self.max_cost = max_cost

		if self.sim_function == 'levenshtein':
			self.sim = levenshtein
		else:
			raise ValueError('we dont recognize this similarity function')


		if self.weight_function in [None, 'idf']:
			self.weight = None
		else:
			raise ValueError('we dont recognize this weight function')

		if self.classifier_type == None:
			self.classifier = sigmoid
		elif self.classifier_type == 'maxent':
			self.classifier = MaxEnt()
		else:
			raise ValueError('we dont recognize this classifier type')


	def alignment_cost(self, w1, w2):
		
		sim =  1 - self.sim(w1, w2)/max(len(w1),len(w2))
		cost = -1*np.log(sim+1)
		return cost 

	def total_cost(self, hypothesis, premise):

		cost = 0
		for h in hypothesis:
			min_cost = min(self.alignment_cost(h,p) if self.weight is None  
							else self.weight[h]*self.alignment_cost(h,p)for p in premise)
			cost += min(self.max_cost, min_cost)

		return cost

	def train(self, corpus):
		#train weight function

		if self.weight_function == 'idf':
			self.weight = idf(corpus)

		#train classifier 
		if self.classifier_type == 'maxent':
			self.classifier.train(corpus)

	def inference(self, h1, h2, p):
		'''
		assume hypothesis and premise are already tokenized
	
		'''
		cost1 = self.total_cost(h1, p)
		cost2 = self.total_cost(h2, p)
		if self.bidirectional:
			cost1 += self.total_cost(p, h1)
			cost2 += self.total_cost(p, h2)

		x = self.classifier([-cost1, -cost2])
		return x

class MaxEnt():


	def __init__(self):
		NotImplementedError

	def train(self):
		NotImplementedError


if __name__ == '__main__':
	model = BagOfWords(
				 None,
				 sim_function='levenshtein',
				 weight_function=None,
				 classifier_type = None,
				 max_cost = 100,
				 bidirectional = False,
				 )

	p1 = 'Chad went to get the wheel alignment measured on his car'	
	p2 = 'The mechanic provided a working alignment with new body work'	
	h1 = 'Chad was waiting for his car to be washed'	
	h2 = 'Chad was waiting for his car to be finished'

	p = p1+' '+p2
	p = p.lower().split(' ')
	h1 = h1.lower().split(' ')
	h2 = h2.lower().split(' ')
	label = 2

	pred = model.inference(h1, h2, p)
	print(pred)