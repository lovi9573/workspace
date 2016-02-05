'''
Created on Jan 13, 2016

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
from weights_to_img import display

class Object:
    pass

N_COLUMNS = 2
N_STEPS = 1
LAYERS = [ConvLayerDef(11,5,1)]
DATA_PARAM = Object()
DATA_PARAM.batch_size = 128
TRANSFORM_PARAM = Object()
TRANSFORM_PARAM.mean_file = ""
TRANSFORM_PARAM.mean_value = [127,127,127]
TRANSFORM_PARAM.crop_size = 225
TRANSFORM_PARAM.mirror = False
PRETRAIN_EPOCHS=0

def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b

def stationary(a,b,thresh):
  if a == None or b == None:
    return False
  matches = 0
  for label,column in a.iteritems():
    if b[label] == column:
      matches += 1
  if float(matches)/float(len(a)) > thresh:
    return True
  return False

def map_img_2_col(columns):
  mapping = {}
  stats = [0]*len(columns)
  outputs = np.zeros([DATA_PARAM.batch_size, len(columns)])
  for mb in dp.get_mb():
    for i,column in columns.iteritems():
      outputs[:,i] = column.loss(mb[0])
      #outputs[:,i] = np.mean(np.max(np.max(act, axis=1), axis=1), axis=1)
    maxvals = np.argmin(outputs,axis=1)
    for key,col in zip(mb[2],maxvals):
      mapping[key] = col
      stats[col] += 1
  #print "Mapping Stats: ",stats
  return mapping

def encode(imap, columns, epochs, epoch_num):
    datas = [np.zeros(dp.shape()) for i in columns]
    indicies = [0 for i in columns]
    for e in range(epoch_num,epoch_num+epochs):
        print "Epoch: {}".format(e)
        for mb in dp.get_mb():
            #print len(mb)
            for i in range(len(mb[2])):
                dat,label,tag = (mb[0][i], mb[1][i], mb[2][i])
                i = imap[tag]
                datas[i][indicies[i],:] = dat
                indicies[i] += 1
                if indicies[i] == DATA_PARAM.batch_size:
                    columns[i].encode_mb(datas[i])
                    indicies[i] = 0
                    #print "encode:",i
 
def mapping_stats(mapping):
  counts = {}
  for val in mapping.values():
    while len(counts) <= val:
      counts[len(counts)] = 0
    counts[val] += 1
  return counts
            

if __name__ == '__main__':
    if len(sys.argv) != 2:
      print "Usage: python dynamic_columns.py <path to lmdb data>"
      sys.exit(-1)
    DATA_PARAM.source = sys.argv[1]
    dp = LMDBDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    imgkeys = dp.get_keys()
    columns = {}
    with tf.Session() as s:
      for i in range(N_COLUMNS):
        columns[i] = AutoEncoder(s,dp)
      print "Columns Initialized"
      #Iterate over layer definitions to build a column
      for l in LAYERS:
        for column in columns.values():
          column.add_layer(l)
        print "{} added".format(l)
        tf.initialize_all_variables().run()
        for i in range(PRETRAIN_EPOCHS):
          for mb in dp.get_mb():
            for column in columns.values():
              column.encode_mb(mb[0])
        print "Columns trained on all data {} epochS".format(PRETRAIN_EPOCHS)
        immap_old = None
        immap = map_img_2_col(columns)
        #Train current layer depth until convergence.
        epoch_num = 0
        while(not stationary(immap, immap_old, 0.95)):
          print "Mapping Distribution",mapping_stats(immap)
          encode(immap, columns, N_STEPS, epoch_num)
          print "Encoding complete"
          for i in range(len(columns)):
            d,r = columns[i].recon(dp.get_mb().next()[0])
            #print(d[1,100:110,100:110,0])
            #print(r[1,100:110,100:110,0])
            plt.imshow(np.append(d[1],r[1],axis=0)[:,:,0], cmap="Greys")
            plt.show()
          epoch_num += N_STEPS
          N_STEPS += 1
          immap_old = immap
          immap = map_img_2_col(columns)
        for i in range(len(columns)):
          print "Example for column {}".format(i)
          key = None
          map_iter = immap.iterkeys()
          try:
            while(key==None):
              k = map_iter.next()
              if immap[k] == i:
                key = k
            sample = dp.get_mb_by_keys([key])
            plt.imshow(sample[0][0,:])
            plt.show()
          except StopIteration:
            pass
          display(s.run(columns[i].layers[-1].W).transpose([3,0,1,2]))
            
#     
# #   dat = dp.get_mb()
# #   for d in dat:
# #       i = d[0][0,:]
# #       print i.shape
# #       plt.imshow(i.reshape((225,225*3)),shape=(225,225*3))
# #       plt.show()
#     for i in range(N_COLUMNS):
#       columns[i] = make_column()