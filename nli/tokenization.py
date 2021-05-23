#tokenization - will hold different tokenization objects here



class WhiteSpaceTokenizer(object):
	'''
	simple white space tokenizer
	'''

	def __init__(self, 
				 vocab):

		self.vocab = json.load(open(vocab, 'r'))

	def tokenize(self, x):
		tokens = tokenize(x, r'\s+', start_symbol=True, end_symbol=True)
		return tokens 
