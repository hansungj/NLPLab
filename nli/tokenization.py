#tokenization - will hold different tokenization objects here

from nli.preprocess import tokenize

class WhiteSpaceTokenizer(object):
	'''
	simple white space tokenizer
	'''

	def __init__(self, 
				 vocab):

		self.vocab = vocab #json.load(open(vocab, 'r'))
		self.lower = vocab['lower']
		self.pad_token = vocab.get('pad_token', None)
		self.unk_token = vocab.get('unk_token', None)
		self.start_token = vocab.get('start_token', None)
		self.end_token = vocab.get('end_token', None)
		self.split_token = vocab.get('split_token', None)

	def tokenize(self, x):
		if self.lower:
			x=x.lower()
		tokens = tokenize(x, r'\s+')
		return tokens 

	def convert_tokens_to_ids(self, tokens):
		unk_token_id = self.vocab['token2idx'][self.unk_token]
		encoded = [self.vocab['token2idx'].get(token, unk_token_id) for token in tokens]
		return encoded

class SubwordTokenizer(object):
	'''
	Word word tokenizer - vocabulary needs to be built using BPE/WordPiece
	'''
	def __init__(self, 
				vocab):

		self.vocab = json.load(open(vocab, 'r'))
		self.pad_token_id = 0
		self.unk_token_id = 1
		self.start_token_id = 2
		self.end_token_id = 3
		self.splt_token_id = 4

	def tokenize(self, x):
		tokens = tokenize(x, r'\s+')

		# here greddily find the best split for each token 
		for token in tokens:
			pass

	def covert_tokens_to_ids(self, tokens):
		pass