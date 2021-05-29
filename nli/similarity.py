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
	try:
		dist = x.get(y, 0) / sum(x.values())
	except ZeroDivisionError:
		dist = 0
	return dist


def cosine(x,y):
	'''
	cosine tokens' distance
	'''
	dist = 0
	axes = set(x.keys()) 
	axes = axes.union(set(y.keys()))
	for axis in axes:
		x_ = x.get(axis, 0)
		y_ = y.get(axis, 0)
		dist += x_ * y_
	x_norm = math.sqrt(sum([t*t for t in x.values()]))
	y_norm = math.sqrt(sum([t*t for t in y.values()]))
	dist = dist / (x_norm * y_norm)
	return dist


def euclidian(x,y):
	'''
	euclidian tokens' distance
	'''
	dist = 0
	axes = set(x.keys()) 
	axes = axes.union(set(y.keys()))
	for axis in axes:
		x_ = x.get(axis, 0)
		y_ = y.get(axis, 0)
		dist += (x_ - y_)**2
	dist = math.sqrt(dist)
	return dist


if __name__ == '__main__':

	x = 'chad'
	y = 'to'
	print(levenshtein(x,y))