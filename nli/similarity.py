import numpy as np
import math


def levenshtein(x,y):
	'''
	levenshtein distance 
	'''

	dp = [[0]*(len(y)+1) for _ in range(len(x)+1)] 

	for i in range(1, len(x)+1):
		for j in range(1,len(y)+1):

			if x[i-1] == y[j-1]:
				dp[i][j] = min(dp[i-1][j-1], dp[i-1][j]+1, dp[i][j-1]+1) 
			else:
				dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1) 

	return dp[-1][-1]


def distributional(x,y):
	'''
	check coocurrence value
	'''
	dist = 0
	dist = x[y] / np.sum(x)
	return dist


def cosine(x,y):
	'''
	cosine tokens' distance
	'''
	dist = 0
	dist = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
	return dist


def euclidian(x,y):
	'''
	euclidian tokens' distance
	'''
	dist = 0
	for i in range(len(x)):
		dist += (x[i] - y[i])**2
	dist = math.sqrt(dist)
	return dist


if __name__ == '__main__':

	x = 'chad'
	y = 'to'
	print(levenshtein(x,y))