'''
Created on May 5, 2016

@author: jlovitt
'''

import numpy as np
import sys
import cPickle

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

if __name__ == '__main__':
    cifarpath = sys.argv[1]
    with open(cifarpath, 'rb') as fi:
      datadict = cPickle.load(fi)
    data = datadict['data']
    original_shape = data.shape
    print("Original Data Shape {}".format(original_shape))
    data = np.reshape(data,[len(data), -1])
    white_data = zca_whitening(data)
    white_data = np.reshape(white_data,original_shape)
    newdatadict = {'data':white_data,
                   'labels':datadict['labels']}
    with open("zca_cifar_data", 'w') as fo:
      cPickle.dump(newdatadict, fo)
    