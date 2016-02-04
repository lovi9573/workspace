'''
Created on Feb 3, 2016

@author: jlovitt
'''
import sys
from dataio import LMDBDataProvider
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from sigmoid_autoencoder import *
import random as rand
import numpy as np
import tensorflow as tf

class Object:
    pass

N_COLUMNS = 3
N_STEPS = 2
LAYERS = [ConvLayerDef(5,4,10)]
DATA_PARAM = Object()
DATA_PARAM.batch_size = 16
TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [0,0,0]
TRANSFORM_PARAM.crop_size = 225
TRANSFORM_PARAM.mirror = False


if __name__ == '__main__':
    if len(sys.argv) != 2:
      print "Usage: python dynamic_columns.py <path to lmdb data>"
      sys.exit(-1)
    DATA_PARAM.source = sys.argv[1]
    dp = LMDBDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    mb = dp.get_mb().next()
    i = np.array([2,1,0])
    for im in mb[0]:
      print "min: {}, max: {}".format(np.min(im),np.max(im))
      plt.imshow(im[:,:,i], cmap='Greys')
      plt.show()