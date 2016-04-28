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
from PIL import Image
import math
import weights_to_img as w2i
from os import path
from column_definition import LAYERS,DATA_PARAM,TRANSFORM_PARAM,NUM_LABELS
from util import save_recon,save_top,save_injection
            

def pretrain_epoch(column,dp, i):
    print("Pretrain epoch {}".format(i))
    loss = 0
    n = 0
    for mb in dp.get_mb():
      n += 1
      loss += column.train_mb(mb[0])
    loss /= n
    return loss
  
 

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
      
      
      #Iterate over layer definitions to greedy train a column
      for layer_number,l in enumerate(LAYERS):
        column.add_layer(l['Layerdef'],l.get('All',{}).get('Freeze',True) )
        column.set_decode(l['Decodedef'])
        column.build()
        print "{} added".format(l['Layerdef'])
        
        l_params = l.get('All',{})
        if l.get('Train',True) and  l_params.get('N_epochs',0) != 0:
          #Pretrain on all data
          if l_params.get('N_epochs',0) > 0:
            for i in range(l_params.get('N_epochs',0)):
              loss = pretrain_epoch(column, dp, i)
              print("\tAve loss: {}".format(loss))
          elif l_params.get('N_epochs',0) == -1:
            loss = 100
            best_loss = loss
            patience = 0
            i = 0
            while patience < l_params.get("Patience",0):
              best_loss = min(best_loss,loss)
              loss = pretrain_epoch(column, dp, i)
              if (loss - best_loss)/abs(best_loss)  < -l_params.get("Patience_delta",0.1):
                patience = 0
              else:
                patience += 1
              if loss < best_loss:
                print("\tAve loss: {} *** {}".format(loss,patience))
              else:
                print("\tAve loss: {}     {}".format(loss,patience))
              i += 1
          print "Layer {} trained on all data {} epochs".format(layer_number+1,l_params.get('N_epochs',0))
          column.save()
          with open(path.join(IMG_DIR,"col_pretrain_losses".format(layer_number)),"a") as fout:
            fout.write(str(layer_number)+": "+str(loss)+"\n")
#           save_embedding(column,dp)

          #Visual investigation
          save_recon(dp.get_mb().next()[0],column,"pretrain", layer_number,IMG_DIR)  
          save_injection(column,"pretrain",layer_number,IMG_DIR) 
          save_top(dp.get_mb().next()[0],column,"pretrain", layer_number,IMG_DIR)       

     
  