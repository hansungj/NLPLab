#embedding.py 
import numpy as np

def build_coocur_matrix(corp_encoded, token2idx):
    cooccurence_matrix = np.zeros((len(token2idx), len(token2idx)))
    
    for sent_enc in corp_encoded:
        for i in range(len(sent_enc)):
            left_edge = max(i-2,0)
            right_edge = min(i+2,len(sent_enc)-1)
            for l in range(left_edge,i):
                cooccurence_matrix[sent_enc[i]][sent_enc[l]] += 1
            for r in range(i+1,right_edge+1):
                cooccurence_matrix[sent_enc[i]][sent_enc[r]] += 1
                
    return cooccurence_matrix

