'''
Created on Jan 13, 2016

@author: jlovitt
'''
import sys
from dataio import LMDBDataProvider, CifarDataProvider, MnistDataProvider
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from autoencoder import *
import random as rand
import numpy as np
import tensorflow as tf
from weights_to_img import display
from itertools import islice, cycle
from dnf.cli.output import Output
from PIL import Image
import math
import weights_to_img as w2i
from os import path
from column_definition import LAYERS,DATA_PARAM,TRANSFORM_PARAM,NUM_LABELS

N_LABELED_EXAMPLES = 8
            
def get_label_batch(dp,label,n):
  data = np.zeros(dp.shape())
  examples_found = 0
  repeat = math.ceil(data.shape[0]/n)
  data_i = 0
  batch_iter = dp.get_mb()
  while examples_found < n:
    samples, labels, keys = batch_iter.next()
    label_indices = np.argwhere(labels==label)
    if len(label_indices) >0:
      label_indices_i = 0
      while label_indices_i < len(label_indices) and examples_found < n:
        i = label_indices[label_indices_i]
        repeat =  min(repeat, data.shape[0]-data_i)
        data[data_i:data_i+repeat] = samples[i,:]
        data_i += repeat
        examples_found += 1
        label_indices_i += 1
  return data

def save_recon(column, data, label):
    d,r = column.recon(data)
    s = list(d.shape)
    s[0] = s[0]*2
    d_r_array = np.empty(s,dtype=d.dtype)
    d_r_array[0::2,:,:,:] = d
    d_r_array[1::2,:,:,:] = r
    im = w2i.tile_imgs(d_r_array)
    im.save(IMG_DIR+'pretrain_col'+str(label)+'_img_recon_level'+str(layer_number+1)+'.png')
    top_shape = column.top_shape()
    a = np.zeros(top_shape)
    input_shape = data.shape
    imgs = np.zeros([top_shape[-1]] + list(input_shape[1:]))
    for channel in range(top_shape[-1]):
      b = a.copy()
      if len(top_shape) == 4:
        b[0,top_shape[1]/2,top_shape[2]/2,channel] = 1
      elif len(top_shape) == 2:
        b[0,channel] = 1
      c = column.inject(b)
      imgs[channel,:,:,:] = c[0,:,:,:]
    im = w2i.tile_imgs(imgs, normalize=True)
#       im = Image.fromarray(dp.denormalize(c[0,:]).astype(np.uint8).squeeze(),mode='L')
    im.save(IMG_DIR+'pretrain_col'+str(label)+'_level'+str(layer_number+1)+'.png')    

def save_top(column, data, label):
    t = column.fwd(data)
    top_shape = column.top_shape()
    if len(top_shape) == 4:
      im = w2i.tile_imgs(t)
#       im = Image.fromarray(dp.denormalize(c[0,:]).astype(np.uint8).squeeze(),mode='L')
      im.save(IMG_DIR+'pretrain_col'+str(label)+'_level'+str(layer_number+1)+'_top.png')   


 


def pretrain_label(column,data):
    loss = column.encode_mb(data)
    return loss
  
  
def save_embedding(column,dp,label):
  print("Saving embedding")
  effective_examples = dp.get_n_examples() - dp.get_n_examples()%dp.shape()[0]
  embedding = np.zeros((effective_examples,column.top_shape()[-1]))
  labels = np.zeros((effective_examples), dtype = np.uint32)
  i = 0
  mb_size = dp.shape()[0]
  for mb in dp.get_mb():
    embedding[i:i+mb_size,:] = column.encode_mb(mb[0])
    labels[i:i+mb_size] = mb[1]
  np.save(path.join(CHECKPOINT_DIR,"embedding_label"+str(label)), embedding)
  np.save(path.join(CHECKPOINT_DIR,"embedding_labels_label"+str(label)), labels)
   

if __name__ == '__main__':
    if len(sys.argv) < 3:
      print "Usage: python dynamic_columns.py <path to output root (contains: img, check, log)> <path to data> [<>]"
      sys.exit(-1)
    DATA_PARAM.source = sys.argv[2:]
    BASE_PATH = sys.argv[1]
    LOG_DIR = path.join(BASE_PATH,'log/')
    IMG_DIR =  path.join(BASE_PATH,'img/')
    CHECKPOINT_DIR =  path.join(BASE_PATH,'check/')
    dp = CifarDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    imgkeys = dp.get_keys()
    with tf.Session() as sess:
      for label in range(NUM_LABELS):
        g = tf.Graph()
        s = tf.Session(graph=g)
        column = AutoEncoder(s,g,dp,LOG_DIR, CHECKPOINT_DIR, colnum=label)
        data = get_label_batch(dp,label,N_LABELED_EXAMPLES)
        print "Column Initialized"
        
        
        #Iterate over layer definitions to build a column
        for layer_number,l in enumerate(LAYERS):
          column.add_layer(l['Layerdef'])
          print "{} added".format(l['Layerdef'])
          
          if l.get('Train',True):
            #Pretrain on all data
            if l.get('Pretrain_epochs',0) > 0:
              for i in range(l['Pretrain_epochs']):
                loss = pretrain_label(column, data)
                print("\tAve loss: {}".format(loss))
            elif l.get('Pretrain_epochs',0) == -1:
              loss = 100
              best_loss = loss
              patience = 0
              i = 0
              while patience < l.get("Patience",0):
                best_loss = min(best_loss,loss)
                loss = pretrain_label(column,data)
                if (loss - best_loss)/abs(best_loss)  < -l.get("Patience_delta",0.1):
                  patience = 0
                else:
                  patience += 1
                if loss < best_loss:
                  print("\tAve loss: {} *** {}".format(loss,patience))
                else:
                  print("\tAve loss: {}     {}".format(loss,patience))
                i += 1
            print "Layer {} trained on all data {} epochs".format(layer_number+1,l.get('Pretrain_epochs',0))
            column.save()
#             save_embedding(column,dp, label)
  
            #Visual investigation
            save_recon(column, data, label)   
            save_top(column, data, label)       

     
  