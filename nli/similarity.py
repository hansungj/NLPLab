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
   
 
def distributional(x,y, distance_type = 'euclidian'):
    '''
    distance between distributional representations 
    '''
    
    dist = 0
    
    if distance_type == 'euclidian':
        for i in len(x_):
            dist += (x - y)**2
        dist = math.sqrt(dist)
        
    elif distance_type == 'cosine':
        dist = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        
    elif distance_type == 'coocurrence':
        dist = x[y] / np.sum(x)
    
    else:
        raise ValueError('This distance type is not supported')
    return dist


if __name__ == '__main__':

	x = 'chad'
	y = 'to'
	print(levenshtein(x,y))