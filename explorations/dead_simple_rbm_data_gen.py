#!/bin/python
import sys
import numpy as np


if __name__ == "__main__":
	n = int(sys.argv[1])
	path = sys.argv[2]
	lower_cutoffs = {0.0:(0,0,1),
			 0.1:(0,1,0),
			 0.3:(1,0,0),
			 0.4:(1,0,1),
			 0.7:(1,1,0)}
	data = np.zeros((n,3), dtype=np.bool_)
	draws = np.random.rand(n)
	upperbound = 1.0
	cuts = sorted(lower_cutoffs.keys())
	cuts.reverse()
	for k in cuts:
		i = np.logical_and(draws<upperbound,draws>=k)
		pattern = lower_cutoffs[k] 
		print("{} {}'s added".format(np.sum(i), pattern))
		data[i,:] = pattern
		upperbound = k
	np.save(path,data)
		
