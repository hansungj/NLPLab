import numpy as np
from collections import defaultdict

from nli.similarity import levenshtein


def sigmoid(x):
	return  1 / (1 + np.exp(-x))

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
				 classifier,
				 sim_function='levenshtein',
				 weight_function=None,
				 max_cost = 100,
				 bidirectional = False,
				 ):


		self.vocab = vocab
		self.sim_function = sim_function
		self.weight_function = weight_function
		self.bidirectional = bidirectional # consider cost(p|h) and cost(h|p)
		self.max_cost = max_cost
		self.coded = None

		if self.sim_function == 'levenshtein':
			self.sim = levenshtein
		else:
			raise ValueError('we dont recognize this similarity function')


		if self.weight_function in [None, 'idf']:
			self.weight = None
		else:
			raise ValueError('we dont recognize this weight function')


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

		#go through all of training corpus and pre-calculate features once 
		if self.coded is None:

			num_features = 2 if self.bidirectional else 1
			self.coded = np.array((len(corpus), num_features), dtype=np.float32)
			self.labels = np.array(len(corpus), dtype=np.int32)

			for i, (premise, hyp1, hyp2, label) in enumerate(corpus):
				features = self.features(hyp1, hyp2, premise)
				self.coded[:,i] = features
				self.labels[i] = label

		#train classifier 
		self.classifier.train(self.coded, self.labels)

	def features(self, h1, h2, p):

		features = []
		features.append(self.total_cost(h1, p) - self.total_cost(h2, p))

		if self.bidirectional:
			features.append(self.total_cost(p, h1) - self.total_cost(p, h2)) #p(h|p)

		return features

	def inference(self, h1, h2, p):
		'''
		assume hypothesis and premise are already tokenized
	
		'''
		features = self.features(h1,h2,p)
		x = self.classifier.forward(features)
		return x


class GDClassifier(object):

	def train(self, x, y):
		for x_p, y_p in zip(x,y):
			self.train_step(x,y)

		self.gradient_step()
		self.reset_gradient()

	def gradient_step(self):
		self.weights -= self.lr * self.gradient

	def reset_gradient(self):
		self.gradient = np.zeros_like(self.weights)

	def forward(self, x):
		NotImplementedError 

class LogisticRegression(GDClassifier):

	def __init__(self,
				num_features,
				lr=0.01,
				bias = True,
				regularization = True,
				lmda = 0.1):

		self.lr = lr
		self.bias = bias
		self.lmda = lmda
		self.regularization = regularization

		if bias:
			num_features += 1

		self.weights = np.ones(num_features+1)
		self.reset_gradient()

	def train_step(self, x, y):
		'''
		y(1-y) if i = j
		'''
		if bias:
			x = np.append(x, 1)
		y_hat = self.forward(x)
		self.gradient += -(y*(1-y_hat) + (1-y)*y_hat )*x

		if self.regularization:
			gradient += self.lmda*self.weights

	def forward(self, x):
		x = np.sum(self.weights*x)
		x = 1 / (1+ np.exp(-x))
		return x

class MaxEnt(GDClassifier):
	'''
	Using continous features require more complicated solution 
	Hence we take a buckting approach where for each continous feature 
	which we know to be centered around zero 
	we bucket them into discrete features using a specified stepsize
	https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.9459&rep=rep1&type=pdf

	
	"This indicates that the moment constraint is not strong enough to distinguish these 
	two different feature distributions and the resulting MaxEnt model performs poorly."

	'''

	def __init__(self,
				 num_features,
                 step_size,
				 num_buckets,
				 num_classes=2,
				 lr = 0.01,
				 reg = True,
				 reg_lambda = 0.01):
		
		self.num_features = num_features
		self.num_classes = num_classes
		self.step_size = step_size
		self.num_buckets = num_buckets
		self.lr = lr
		self.reg = reg
		self.reg_lambda = reg_lambda

		
		#number of buckets + 2 to reprsent the ranges at the ends 

		# if isinstance(num_buckets, list): # you can also specify the bucket ranges yourself
		# 	self.weights = np.ones(num_classes, (len(num_buckest)*2+2)*num_features)
		# 	self.custom_buckets = True
		# else:
		
		# the order is [f_1 positive, f_1 negative, f_2 positive, f_2 negative ,... ]
		self.weights = np.ones((num_classes, (num_buckets*2+2)*num_features))
		self.reset_gradient()

	def convert_to_features(self, x, y=None):
		'''
		num_buckets = 2
		step_size = 1
		
		[cost1 cost2]

		w = y=0[ [0,1][1,2][2,inf][-1][-2,-1][-inf,-2] [0,1][1,2][2,inf][-1][-2,-1][-inf,-2],
			y=1  [0,1][1,2][2,inf][-1][-2,-1][-inf,-2] [0,1][1,2][2,inf][-1][-2,-1][-inf,-2]] 

		p = [0.1,
		     0.9]

		cost = 1.5 // step_size  = 1

		assume that y in {0, 1}
		'''

		features = np.zeros_like(self.weights)
		for i, f_value in enumerate(x):

			if abs(f_value) > self.num_buckets*self.step_size:
				idx = self.num_buckets
			else:
				idx = abs(f_value)//self.step_size

			if f_value < 0 : 
				idx += self.num_buckets+1

			idx += i*((num_buckets*2)+2)

			if y is None:
				features[:,idx] = 1 
			else:
				features[y,idx] = 1

		return features 

	def forward(self, x, as_features = False):

		if not as_features:
			x = self.convert_to_features(x) # num_classes * total_num_features 
		x = self.weights*x
		x = np.sum(x,axis=1)
		x = np.exp(x)
		return x / np.sum(x) # outputs vector of size num_classes

	def model_expectation(self, x):

		x = self.convert_to_features(x)
		model_p = self.forward(x)

		expectation = model_p*x
		return expectation

	def empirical_expectation(self, x, y):
		'''
		x already in features 
		y in range(num_classes)

		E[f]
		'''
		expectation = self.convert_to_features(x, y)
		return expectation

	def train_step(self, x, y):
		#compute empirical expectation for the features 
		e_p_emp = self.empirical_expectation(x,y)

		#compute model expectation for the features 
		e_p_model = self.model_expectation(x)

		#compute gradient
		self.gradient += e_p_emp - e_p_model

		#L2 regularization: lambda * W^2, or W^2/(2*sigma^2)
		if self.reg:
			self.gradient -= self.reg_lambda*self.weights


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