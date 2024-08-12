import numpy as np

# ------------ Pooling Functions ------------ # 
def pool_concat(X):
	"""Pools features by concat each features vector head-to-tail. Results in a word-level feature vector of 5x19"""
	return np.array([np.concatenate(submatrices, axis=0) for submatrices in X])

def pool_last(X): 
	"""Pools phonetic feature vectors by only returning the final segment. Code must correctly identify last non-padding segment"""
	new_X = []
	max_segments = len(X[0])
	for word in X:
		curr_idx = max_segments - 1
		while curr_idx >= 0:
			last = word[curr_idx]
			if len(set(last)) > 1 or 0 not in set(last): # Not pad token
				break
			curr_idx -= 1
		new_X.append(last)
	return np.array(new_X)