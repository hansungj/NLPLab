#embedding.py 
import numpy as np

def build_coocurences(corp_encoded, token2idx, window = 2):
    cooccurence_dict = dict()
    
    for sent_enc in corp_encoded:
        for i in range(len(sent_enc)):
            if not sent_enc[i] in cooccurence_dict:
                cooccurence_dict[sent_enc[i]] = dict()
            left_edge = max(i-window,0)
            right_edge = min(i+window,len(sent_enc)-1)
            
            for l in range(left_edge,i):
                if sent_enc[l] in cooccurence_dict[sent_enc[i]]:
                    cooccurence_dict[sent_enc[i]][sent_enc[l]] += 1
                else:
                    cooccurence_dict[sent_enc[i]][sent_enc[l]] = 1
            for r in range(i+1,right_edge+1):
                if sent_enc[l] in cooccurence_dict[sent_enc[i]]:
                    cooccurence_dict[sent_enc[i]][sent_enc[l]] += 1
                else:
                    cooccurence_dict[sent_enc[i]][sent_enc[l]] = 1
                
    return cooccurence_dict

