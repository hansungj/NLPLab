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
   
 
 def distributional(x,y, idx2tok, cooccurence_matrix, distance_type = 'eucledian'):
    '''
    distance between distributional representations 
    '''
    
    x_vec = cooccurence_matrix[idx2tok[x]]
    y_vec = cooccurence_matrix[idx2tok[y]]
    dist = 0
    
    if distance_type == 'eucledian':
        for i in len(x_vec):
            dist += (x_vec - y_vec)**2
        dist = math.sqrt(dist)
        
    elif distance_type == 'cosine':
        dist = np.dot(x_vec, y_vec) / (np.linalg.norm(x_vec) * np.linalg.norm(y_vec))
        
    else:
        raise ValueError('This distance type is not supported')
    return dist


if __name__ == '__main__':

	x = 'chad'
	y = 'to'
	print(levenshtein(x,y))