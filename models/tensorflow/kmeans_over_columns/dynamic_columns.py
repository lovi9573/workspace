'''
Created on Jan 13, 2016

@author: jlovitt
'''
import sys
from dataio import LMDBDataProvider
from matplotlib import pyplot as plt
from sigmoid_autoencoder import *
import random as rand
import numpy as np
import tensorflow as tf

class Object:
    pass

N_COLUMNS = 3
N_STEPS = 2
LAYERS = [LayerDef("FC",10)]
DATA_PARAM = Object()
DATA_PARAM.batch_size = 16
TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [0,0,0]
TRANSFORM_PARAM.crop_size = 225
TRANSFORM_PARAM.mirror = False

def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b

def map_img_2_col(columns):
  mapping = {}
  stats = [0]*len(columns)
  outputs = np.zeros([DATA_PARAM.batch_size, len(columns)])
  for mb in dp.get_mb():
    for i,column in columns.iteritems():
      act = column.fwd(mb[0])
      outputs[:,i] = np.mean(act, axis=1)
    maxvals = np.argmax(outputs,axis=1)
    for key,col in zip(mb[2],maxvals):
      mapping[key] = col
      stats[col] += 1
  print "Mapping Stats: ",stats
  return mapping

def encode(map, columns, epochs):
    datas = [np.zeros(dp.shape()) for i in columns]
    indicies = [0 for i in columns]
    for e in range(epochs):
        for mb in dp.get_mb():
            print len(mb)
            for i in range(len(mb[2])):
                dat,label,tag = (mb[0][i], mb[1][i], mb[2][i])
                i = map[tag]
                datas[i][indicies[i],:] = dat
                indicies[i] += 1
                if indicies[i] == DATA_PARAM.batch_size:
                    columns[i].encode_mb(datas[i])
                    indicies[i] = 0
                    print "encode:",i
            

if __name__ == '__main__':
    DATA_PARAM.source = sys.argv[1]
    dp = LMDBDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    imgkeys = dp.get_keys()
    columns = {}
    with tf.Session() as s:
      for i in range(N_COLUMNS):
        columns[i] = AutoEncoder(s,dp)
      print columns
      for l in LAYERS:
        for column in columns.values():
          column.add_layer(l)
        tf.initialize_all_variables().run()
        immap_old = None
        immap = map_img_2_col(columns)
        while(not converged(immap, immap_old)):
          encode(immap, columns, N_STEPS)
#         immap_old = immap
#         immap = map_img_2_col(imgkeys, columns)
#     
# #   dat = dp.get_mb()
# #   for d in dat:
# #       i = d[0][0,:]
# #       print i.shape
# #       plt.imshow(i.reshape((225,225*3)),shape=(225,225*3))
# #       plt.show()
#     for i in range(N_COLUMNS):
#       columns[i] = make_column()