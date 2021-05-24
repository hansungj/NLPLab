#tokenization - will hold different tokenization objects here

from preprocess import tokenize

class WhiteSpaceTokenizer(object):
	'''
	simple white space tokenizer
	'''

	def __init__(self, 
				 vocab):

		self.vocab = vocab #json.load(open(vocab, 'r'))
		self.pad_token = vocab.get('pad_token', None)
		self.unk_token = vocab.get('null_token', None)
		self.start_token = vocab.get('null_token', None)
		self.end_token = vocab.get('null_token', None)
		self.splt_token = vocab.get('null_token', None)

	def tokenize(self, x):
		tokens = tokenize(x, r'\s+', start_symbol=None, end_symbol=None)
		return tokens 

	def convert_tokens_to_idx(self, tokens):
		unk_token_id = vocab['token2idx'][self.unk_token]
		encoded = [self.vocab.get(token, )for token in tokens]
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
		tokens = tokenize(x, r'\s+', start_symbol=True, end_symbol=True)

		# here greddily find the best split for each token 
		for token in tokens:
			pass
