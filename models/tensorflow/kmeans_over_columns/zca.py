'''
Created on May 5, 2016

@author: jlovitt
'''

import numpy as np
import sys
import cPickle
from sklearn.decomposition import PCA
import weights_to_img as w2i
import matplotlib.pyplot as plt

def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector

def zca_whitening(inputs):
    sigma = np.dot(inputs.T, inputs)/inputs.shape[0] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    S_diag = 1.0/np.sqrt(np.diag(S) + epsilon)
    U_S = np.dot(U, S_diag)
    ZCAMatrix = np.dot(U_S, U.T )    #ZCA Whitening matrix
    print("U shape {}".format(U.shape))
    print("S shape {}".format(S.shape))
    print("U_S shape {}".format(U_S.shape))
    print("S_diag shape {}".format(S_diag.shape))
    print("ZCAMatrix shape {}".format(ZCAMatrix.shape))
    print("inputs shape {}".format(inputs.shape))
    return np.dot(inputs, ZCAMatrix )   #Data whitening
  
  
def zca_whitening_sk(inputs):
  pca = PCA(whiten=True)
  transformed = pca.fit_transform(inputs)
  pca.whiten = False
  zca = pca.inverse_transform(transformed)
  return zca

if __name__ == '__main__':
    cifarpath = sys.argv[1]
    with open(cifarpath, 'rb') as fi:
      datadict = cPickle.load(fi)
    data = datadict['data']
    original_shape = data.shape
    print("Original Data Shape {}".format(original_shape))
    data = np.reshape(data,[len(data), -1])
    white_data = zca_whitening(data)
    comp = w2i.tile_imgs(np.transpose(
                                      np.reshape(
                                                 np.append(
                                                           data[0,:], 
                                                           white_data[0,:], 
                                                           axis =0), 
                                                 [2,3,32,32]
                                                 )
                                      ),
                                      [0,2,3,1]
                        )
    plt.imshow(comp)
    plt.show()
    white_data = np.reshape(white_data,original_shape)
    newdatadict = {'data':white_data,
                   'labels':datadict['labels']}
    with open("zca_cifar_data", 'w') as fo:
      cPickle.dump(newdatadict, fo)
    