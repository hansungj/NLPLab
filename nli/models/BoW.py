import numpy as np
from collections import defaultdict

from nli.similarity import levenshtein, distributional, cosine, euclidian
from nli.metrics import log_likelihood

import json
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm


def sigmoid(x):
	return  1 / (1 + np.exp(-x))

class BagOfWordsWrapper(object):

	def idf(self, corpus):
		'''
		build idfs - for lexical weighting 
		
		'''
		d = {}
		N = len(corpus)
		for doc_id, (p,h1,h2,_) in enumerate(corpus):
			for token in p+h1+h2:
				if token not in d:
					d[token] = [1, doc_id]

				elif doc_id != d[token][1]:
					d[token][0] += 1
					d[token][1] = doc_id

		d = {word:-np.log(doc_freq[0]/N) for word, doc_freq  in d.items()}

		#calculate average weighting to use for OOV words 
		avg = sum(d.values())/len(d)
		
		#save 
		self.weight = d
		self.weight_avg = avg

	def build_vocabulary(self, corpus):
		token2idx = defaultdict(int)
		
		for (p,h1,h2,_) in corpus:
			for sent in (p,h1,h2):
				for token in sent:
					if token not in token2idx.keys():
						token2idx[token] = len(tok2idx)
		
		idx2tok = {v:k for k,v in tok2idx.items()}
		
		vocab = {
		'token2idx': token2idx,
		'idx2token': idx2token
		}
		return vocab
	
	
	def distributional_represent(self, x):
		token2idx = self.vocab['token2idx']
		x_represent = np.zeros(len(token2idx))
		if x in token2idx.keys(): #if there was no such word in our vocab (train data), then we know nothing about its cooccurences and represent it as zero vector
			if token2idx[x] in self.cooccurence_dict.keys():
				for idx in self.cooccurence_dict[token2idx[x]]:
					x_represent[idx-1] = self.cooccurence_dict[token2idx[x]][idx]

		return x_represent

	def build_coocurences(self, corpus, window = 2):
		token2idx = self.vocab['token2idx']
		cooccurence_dict = defaultdict(dict)

		for (p,h1,h2,_) in corpus:
			for sent in (p,h1,h2):
				for i in range(len(sent)):

					cur_token = token2idx[sent[i]]
					
					left_edge = max(i-window,0)
					right_edge = min(i+window,len(sent)-1)

					for l in range(left_edge,i):
						neighbor_token = token2idx[sent[l]]
						if neighbor_token in cooccurence_dict[cur_token]:
							cooccurence_dict[cur_token][neighbor_token] += 1
						else:
							cooccurence_dict[cur_token][neighbor_token] = 1

					for r in range(i+1,right_edge+1):
						neighbor_token = token2idx[sent[r]]
						if neighbor_token in cooccurence_dict[cur_token]:
							cooccurence_dict[cur_token][neighbor_token] += 1
						else:
							cooccurence_dict[cur_token][neighbor_token] = 1
		#print(len(cooccurence_dict))			
		#cooccurence_dict['UNK']
		return cooccurence_dict

class BagOfWords(BagOfWordsWrapper):

	def __init__(self,
				 classifier,
				 sim_function='levenshtein',
				 weight_function=None,
				 cooccurence_dict = None,
				 max_cost = 100,
				 bidirectional = False,
				 lemmatize = False,
				 vocab = None):

		self.classifier = classifier
		self.sim_function = sim_function
		self.weight_function = weight_function
		self.bidirectional = bidirectional # consider cost(p|h) and cost(h|p)
		self.max_cost = max_cost
		self.lemmatize = lemmatize
		self.coded = None
		self.vocab = vocab

		if self.sim_function == 'levenshtein':
			self.sim = levenshtein
		elif self.sim_function == 'distributional':
			self.sim = distributional
		elif self.sim_function == 'cosine':
			self.sim = cosine
		elif self.sim_function == 'euclidian':
			self.sim = euclidian
		else:
			raise ValueError('we dont recognize this similarity function')


		if self.weight_function in [None, 'idf']:
			self.weight = None
		else:
			raise ValueError('we dont recognize this weight function')

		#if lemmatize = initialize wordnet lemmatizer
		if lemmatize:
			self.lemmatizer = WordNetLemmatizer()

	def alignment_cost(self, w1, w2):
		#print('Words are {} and {}'.format(w1,w2))
		if self.sim_function in ['cosine', 'euclidian']:
			w1 = self.distributional_represent(w1)
			w2 = self.distributional_represent(w2)

			if self.sim_function == 'cosine':
				sim = self.sim(w1, w2)
				cost = -1*np.log(sim+1)
			else:
				try:
					sim = 1 / self.sim(w1, w2)
				except ZeroDivisionError:  
					sim = 1
				cost = -1*np.log(sim+1)


		elif self.sim_function == 'distributional':
			w1 = self.distributional_represent(w1)
			token2idx = self.vocab['token2idx']

			if w2 in token2idx.keys():
				w2 = token2idx[w2]
				sim = self.sim(w1, w2)
			else:
				sim = 0 # if the token was not seen in the training data, then it never cooccured with anything in the training data

			try:
				cost = -1*np.log(sim+1)
			except ZeroDivisionError:
				cost = 1

		else:
			if self.lemmatize:
				w1 = self.lemmatizer.lemmatize(w1)
				w2 = self.lemmatizer.lemmatize(w1)

			sim =  1 - self.sim(w1, w2)/max(len(w1),len(w2))
			cost = -1*np.log(sim+1)
		return cost 

	def total_cost(self, hypothesis, premise):

		cost = 0
		for h in hypothesis:
			min_cost = min(self.alignment_cost(h,p) if self.weight is None  
							else self.weight.get(h,self.weight_avg)*self.alignment_cost(h,p)for p in premise)
			cost += min(self.max_cost, min_cost)

		return cost

	def features(self, h1, h2, p):

		features = []
		features.append(self.total_cost(h1, p) - self.total_cost(h2, p))

		if self.bidirectional:
			features.append(self.total_cost(p, h1) - self.total_cost(p, h2)) #p(h|p)

		return features

	def inference(self, features):
		'''
		assume hypothesis and premise are already tokenized
	
		'''
		#features = self.features(h1,h2,p)
		x = self.classifier.forward(features)
		return x

	def fit(self, corpus, num_epochs=1, verbose = True):
		'''
		Corpus needs to be a list of tuples (premise, hyp1, hyp2, label)
		'''
		#train weight function

		if self.sim_function in ['distributional', 'cosine', 'euclidian']:
			self.vocab = self.build_vocabulary(corpus)
			self.cooccurence_dict = self.build_coocurences(corpus)

		if self.weight_function == 'idf':
			self.idf(corpus)

		#go through all of training corpus and pre-calculate features once 
		num_features = 2 if self.bidirectional else 1
		self.coded = np.zeros((len(corpus), num_features), dtype=np.float32)
		self.labels = np.zeros(len(corpus), dtype=np.int32)

		if verbose:
			print('fitting..')

		for i in tqdm(range(len(corpus))):
			premise, hyp1, hyp2, label = corpus[i]
			features = self.features(hyp1, hyp2, premise)
			self.coded[i,:] = features
			self.labels[i] = label

		#train for num_epochs
		for epoch in range(num_epochs):
			self.classifier.train(self.coded, self.labels)

	def fit_transform(self, corpus, num_epochs=1, ll=True, verbose=True):
		'''
		same as fit but output the result along with log liklihood 
		Corpus needs to be a list of tuples (premise, hyp1, hyp2, label)
		'''
		self.fit(corpus)

		log_like = []
		#train for num_epochs
		for epoch in range(num_epochs):
			self.classifier.train(self.coded, self.labels)

			pred = np.zeros((len(corpus), 2))

			if verbose:
				print('predicting..')
			for i in tqdm(range(len(corpus))):
				p = self.inference(self.coded[i,:]) # size num_classes
				pred[i,:] = p

			if ll:
				L = log_likelihood(self.labels, pred)
				log_like.append(L)

		return pred, log_like

	def transform(self, X, verbose=True):
		pred = np.zeros((len(X), 2))

		if verbose:
			print('predicting...')
		for i in tqdm(range(len(X))):
			premise, hyp1, hyp2, label = X[i]
			features = self.features(hyp1, hyp2, premise)
			p = self.inference(features) # size num_classes
			pred[i,:] = p
		return pred


class GDClassifier(object):

	def train(self, x, y):
		N = len(x)
		for x_p, y_p in zip(x,y):
			self.train_step(x_p,y_p, N)

		self.gradient_step()
		self.reset_gradient()

	def gradient_step(self):
		self.weights -= self.lr * self.gradient

	def reset_gradient(self):
		self.gradient = np.zeros_like(self.weights)

	def forward(self, x):
		NotImplementedError 

class Perceptron(GDClassifier):

	def __init__(self,
				num_features,
				lr=1,
				bias = True):

		self.lr = lr
		self.bias = bias

		if bias:
			num_features += 1
		self.weights = np.zeros(num_features)
		self.reset_gradient()

	def train_step(self, x, y, N):
		if self.bias:
			x = np.append(x, 1)
		y_hat = self.forward(x)

		#print("predict: {}".format(y_hat))
		#print("label: {}".format(y))
		#print("data: {}".format(x))
		
		self.gradient += -y_hat*x

		return y_hat

	def forward(self, x):
		x = np.sum(self.weights*x)
		x = np.sign(x)
		return x

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
		self.weights = np.zeros(num_features)
		self.reset_gradient()

	def train_step(self, x, y, N):
		'''
		y(1-y) if i = j
		'''
		if self.bias:
			x = np.append(x, 1)
		y_hat = self.forward(x)

		# print(y_hat)
		# print(y)
		# print(x)
		
		self.gradient += (-(y*(1-y_hat) + (1-y)*y_hat )*x)/N

		if self.regularization:
			self.gradient += self.lmda*self.weights

		return y_hat

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
		self.weights = np.zeros((num_classes, (num_buckets*2+2)*num_features))
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

			idx += i*((self.num_buckets*2)+2)
			idx = int(idx)

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
		model_p = self.forward(x, True)

		expectation = model_p.reshape(2, 1)*x
		return expectation

	def empirical_expectation(self, x, y):
		'''
		x already in features 
		y in range(num_classes)

		E[f]
		'''
		expectation = self.convert_to_features(x, y)
		return expectation

	def train_step(self, x, y, N):
		#compute empirical expectation for the features 
		e_p_emp = self.empirical_expectation(x,y)

		#compute model expectation for the features 
		e_p_model = self.model_expectation(x)

		#compute gradient
		self.gradient -= (e_p_emp - e_p_model)/N

		#L2 regularization: lambda * W^2, or W^2/(2*sigma^2)
		if self.reg:
			self.gradient += (self.reg_lambda*self.weights)/N


if __name__ == '__main__':

	p1 = 'Chad went to get the wheel alignment measured on his car'	
	p2 = 'The mechanic provided a working alignment with new body work'	
	h1 = 'Chad was waiting for his car to be washed'	
	h2 = 'Chad was waiting for his car to be finished'

	p = p1+' '+p2
	p = p.lower().split(' ')
	h1 = h1.lower().split(' ')
	h2 = h2.lower().split(' ')
	label = 1

	#classifier = MaxEnt(num_features=1,
				 #step_size=0.5,
				 #num_buckets=4,
				 #num_classes=2,
				 #lr = 0.01,
				 #reg = True,
				 #reg_lambda = 0.01)

	#classifier = LogisticRegression(num_features=1,
				#lr=0.01,
				#bias = True,
				#regularization = True,
				#lmda = 0.1)

	classifier = Perceptron(num_features=1,
				lr=1,
				bias = True)

	model = BagOfWords({},
				 classifier,
				 sim_function='levenshtein',
				 weight_function=None,
				 max_cost = 100,
				 bidirectional = False)


	for i in range(10):
		model.train([(p, h1, h2, label),(p, h1, h2, label)])

	print(model.coded)
	print(model.labels)
