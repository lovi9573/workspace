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











def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b

def stationary(a,b,thresh):
  if thresh*len(a) < 1.0:
    return True
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
  rev_mapping = dict([(col,[]) for col in columns.keys()])
  stats = dict([(col,0) for col in columns.keys()])
  outputs = np.zeros([DATA_PARAM.batch_size, len(columns)])
  for mb in dp.get_mb():
    for i,column in columns.iteritems():
       tmp = column.individual_reconstruction_loss(mb[0])
       outputs[:,i] = tmp
      #outputs[:,i] = np.mean(np.max(np.max(act, axis=1), axis=1), axis=1)
    maxvals = np.argmin(outputs,axis=1)
    for key,col in zip(mb[2],maxvals):
      mapping[key] = col
      stats[col] += 1
      rev_mapping[col].append(key)
  #print "Mapping Stats: ",stats
  return {'key2col':mapping, 'stats':stats, "col2key":rev_mapping}

def encode(imap, columns, epochs, epoch_num):
    datas = [np.zeros(dp.shape()) for i in columns]
    indicies = [0 for i in columns]
    max_examples = max(imap['stats'])
    if min(imap['stats']) == 0:
      max_examples = DATA_PARAM.batch_size
    n_updates = max_examples/DATA_PARAM.batch_size
    report = ["\tTraining column {} on {} examples\n".format(col,len(k)) for col,k in imap['col2key'].iteritems()]
    print(''.join(report))
    for e in range(epoch_num,epoch_num+epochs):
        print "Epoch: {}".format(e)
        for mb in dp.get_mb():
            #print len(mb)
            for i in range(len(mb[2])):
                dat,label,tag = (mb[0][i], mb[1][i], mb[2][i])
                colnum = imap['key2col'][tag]
                datas[colnum][indicies[colnum],:] = dat
                indicies[colnum] += 1
                if indicies[colnum] == DATA_PARAM.batch_size:
                    columns[colnum].encode_mb(datas[colnum])
                    indicies[colnum] = 0
                for colnum,stat in imap['stats'].iteritems():
                  if stat == 0:
                    datas[colnum][indicies[colnum],:] = dat
                    indicies[colnum] += 1
                    if indicies[colnum] == DATA_PARAM.batch_size:
                      columns[colnum].encode_mb(datas[colnum])
                      indicies[colnum] = 0
 
def train(imap, columns, keys, epochs, epoch_num):
    indicies = [0 for i in columns]
    max_examples = max(imap['stats'].values())
    if min(imap['stats'].values()) == 0:
      max_examples = len(keys)
      for colnum,n in imap['stats'].iteritems():
        if n == 0:
          imap['col2key'][colnum] = keys
          print("\tColumn {} has no examples mapped to it.  Training it on all data".format(colnum))
    n_updates = max_examples/DATA_PARAM.batch_size
    for e in range(epoch_num,epoch_num+epochs):
        print "Epoch: {}".format(e)
        losses = dict([(col,0) for col in columns.keys()])
        for update in range(n_updates):
          for colnum,col in columns.iteritems():
            batch_keys = list(islice(cycle(imap['col2key'][colnum]),indicies[colnum],indicies[colnum]+DATA_PARAM.batch_size))
            indicies[colnum] += DATA_PARAM.batch_size
            s,l,k = dp.get_mb_by_keys(batch_keys)
            #print("\tTraining column {} on {} keys".format(colnum,len(batch_keys)))
            losses[colnum] += columns[colnum].encode_mb(s)
    return dict([(col, loss/n_updates) for col,loss in losses.iteritems()])
            

def save_recon(dp, column):
    d,r = column.recon(dp.get_mb().next()[0])
    s = list(d.shape)
    s[0] = s[0]*2
    d_r_array = np.empty(s,dtype=d.dtype)
    d_r_array[0::2,:,:,:] = d
    d_r_array[1::2,:,:,:] = r
    im = w2i.tile_imgs(d_r_array)
    im.save(IMG_DIR+'col_pretrain_img_recon_level'+str(layer_number+1)+'.png')
    top_shape = column.top_shape()
    a = np.zeros(top_shape)
    input_shape = dp.shape()
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
    im.save(IMG_DIR+'col_pretrain_level'+str(layer_number+1)+'_inject.png')    

def save_top(dp, column):
    t = column.fwd(dp.get_mb().next()[0])
    top_shape = column.top_shape()
    if len(top_shape) == 4:
      im = w2i.tile_imgs(t)
#       im = Image.fromarray(dp.denormalize(c[0,:]).astype(np.uint8).squeeze(),mode='L')
      im.save(IMG_DIR+'col_pretrain_level'+str(layer_number+1)+'_top.png')   


 
def mapping_stats(mapping):
  counts = {}
  for val in mapping.values():
    while len(counts) <= val:
      counts[len(counts)] = 0
    counts[val] += 1
  return counts


def pretrain_epoch(column,dp, i):
    print("Pretrain epoch {}".format(i))
    loss = 0
    n = 0
    for mb in dp.get_mb():
      n += 1
      loss += column.encode_mb(mb[0])
    loss /= n
    return loss
  
  
def save_embedding(column,dp):
  print("Saving embedding")
  effective_examples = dp.get_n_examples() - dp.get_n_examples()%dp.shape()[0]
  embedding = np.zeros((effective_examples,column.top_shape()[-1]))
  labels = np.zeros((effective_examples), dtype = np.uint32)
  i = 0
  mb_size = dp.shape()[0]
  for mb in dp.get_mb():
    embedding[i:i+mb_size,:] = column.encode_mb(mb[0])
    labels[i:i+mb_size] = mb[1]
  np.save(path.join(CHECKPOINT_DIR,"embedding"), embedding)
  np.save(path.join(CHECKPOINT_DIR,"embedding_labels"), labels)
   

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
      g = tf.Graph()
      s = tf.Session(graph=g)
      column = AutoEncoder(s,g,dp,LOG_DIR, CHECKPOINT_DIR)
      print "Column Initialized"
      
      
      #Iterate over layer definitions to build a column
      for layer_number,l in enumerate(LAYERS):
        column.add_layer(l['Layerdef'])
        print "{} added".format(l['Layerdef'])
        
        if l.get('Train',True):
          #Pretrain on all data
          if l.get('Pretrain_epochs',0) > 0:
            for i in range(l['Pretrain_epochs']):
              loss = pretrain_epoch(column, dp, i)
              print("\tAve loss: {}".format(loss))
          elif l.get('Pretrain_epochs',0) == -1:
            loss = 100
            best_loss = loss
            patience = 0
            i = 0
            while patience < l.get("Patience",0):
              best_loss = min(best_loss,loss)
              loss = pretrain_epoch(column, dp, i)
              if (loss - best_loss)/abs(best_loss)  < -l.get("Patience_delta",0.1):
                patience = 0
                print("\tAve loss: {} ***".format(loss))
              else:
                print("\tAve loss: {}".format(loss))
                patience += 1
              i += 1
          print "Layer {} trained on all data {} epochs".format(layer_number+1,l.get('Pretrain_epochs',0))
          column.save()
          with open(path.join(IMG_DIR,"col_pretrain_losses".format(layer_number)),"a") as fout:
            fout.write(str(layer_number)+": "+str(loss)+"\n")
#           save_embedding(column,dp)

          #Visual investigation
          save_recon(dp,column)   
          save_top(dp,column)       

     
  