#!/bin/python
import numpy as np
from sklearn.neural_network import BernoulliRBM
import sys


if __name__=="__main__":
	data = np.load(sys.argv[1])
	rbm = BernoulliRBM(n_components=1, learning_rate=0.9,batch_size=16,n_iter=20)
	rbm.fit(data)
	print("{}".format(rbm.components_))
	print("{}".format(rbm.intercept_visible_))
	print("{}".format(rbm.intercept_hidden_))
