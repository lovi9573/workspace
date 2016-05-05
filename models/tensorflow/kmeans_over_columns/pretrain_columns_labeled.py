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
from column_definition import LAYERS,DATA_PARAM,TRANSFORM_PARAM,NUM_LABELS,get_dp,N_LABELED_EXAMPLES
from util import save_recon,save_top,save_injection,get_label_batch



            

if __name__ == '__main__':
    if len(sys.argv) < 3:
      print "Usage: python dynamic_columns.py <path to output root (contains: img, check, log)> <path to data> [<>]"
      sys.exit(-1)
    DATA_PARAM.source = sys.argv[2:]
    BASE_PATH = sys.argv[1]
    LOG_DIR = path.join(BASE_PATH,'log/')
    IMG_DIR =  path.join(BASE_PATH,'img/')
    CHECKPOINT_DIR =  path.join(BASE_PATH,'check/')
    dp = get_dp(DATA_PARAM,TRANSFORM_PARAM )
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
          column.add_layer(l['Layerdef'],l.get('Labeled',{}).get('Freeze',True))
          column.set_decode(l['Decodedef'])
          column.build()
          print "{} added".format(l['Layerdef'])
          
          l_params = l.get('Labeled',{})
          if l.get('Train',True):
            #Pretrain on all data
            if l_params.get('N_epochs',0) > 0:
              for i in range(l_params.get('N_epochs',0)):
                loss = column.train_mb(data)
                print("\tAve loss: {}".format(loss))
            elif l_params.get('N_epochs',0) == -1:
              loss = 100
              best_loss = loss
              patience = 0
              i = 0
              while patience < l_params.get("Patience",0):
                best_loss = min(best_loss,loss)
                loss = column.train_mb(data)
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
#             save_embedding(column,dp, label)
  
            #Visual investigation
            save_recon(dp.get_mb().next()[0],column,label, layer_number,IMG_DIR)  
            save_injection(column,label,layer_number,IMG_DIR) 
            save_top(dp.get_mb().next()[0],column,label, layer_number,IMG_DIR)       

     
  