# initialization for models 

from nli.models.BoW import MaxEnt, LogisticRegression, BagOfWords, Perceptron
from nli.models.StaticEmb import StaticEmbeddingMixture, StaticEmbeddingRNN, StaticEmbeddingCNN
from nli.models.Transformers import PretrainedTransformerCLS, PretrainedTransformerPooling, PretrainedDecoderTransformerCLS, PretrainedDecoderTransformerDual

