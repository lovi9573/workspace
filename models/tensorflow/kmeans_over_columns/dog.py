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
from scipy.ndimage.filters import gaussian_filter

  
def dog(imgs, sigma1, sigma2):
  imgsf32 = imgs.astype(np.float32)
  a = gaussian_filter(imgsf32, [0,0,sigma1, sigma1])
  b = gaussian_filter(imgsf32, [0,0,sigma2, sigma2])
  print("MIN {}, MAX {}".format(np.min(imgs), np.max(imgs)))
  print(a.dtype)
  dog = a-b
  m =np.min(dog)
  M = np.max(dog)
  r = M-m
  dog = (dog-m)/r*255.0
  dogi8 = dog.astype(np.uint8)
  return a,b,dogi8


if __name__ == '__main__':
    cifarpath = sys.argv[1]
    with open(cifarpath, 'rb') as fi:
      datadict = cPickle.load(fi)
    data = datadict['data']
    data = np.reshape(data,[len(data), 3,32,32])
    original_shape = data.shape
    print("Original Data Shape {}".format(original_shape))
    a,b,dog_data = dog(data,1,0.5)
    print("Dog Data Shape {}".format(dog_data.shape))
    appended_mat = np.append(
                             np.append(data[0:2], 
                                       a[0:2],
                                       axis=0
                                       ),
                             np.append(b[0:2],
                                       dog_data[0:2], 
                                       axis =0
                                       ),
                             axis=0
                             )
    print(appended_mat.shape)
    comparison_mat = np.transpose(appended_mat,[0,2,3,1])
    comp = w2i.tile_imgs(comparison_mat)
    plt.imshow(comp)
    plt.show()
    newdatadict = {'data':dog_data,
                   'labels':datadict['labels']}
    with open("dog_cifar_data", 'w') as fo:
      cPickle.dump(newdatadict, fo)
    