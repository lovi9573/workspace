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



# number of columns to route data to dynamically
N_COLUMNS = 10

# Number of batches to train between data reroute's
TRAIN_BATCHES = 1

# Growth rate for TRAIN_BATCHES every training cycle
D_TRAIN_BATCHES = 2








def converged(a, b):
  if a == None or b == None:
    return False
  else:
    return a == b

def stationary(a,b,thresh):
  if a == None or b == None:
    return False,None
  matches = 0
  for label,column in a.iteritems():
    if b[label] == column:
      matches += 1
  r = float(matches)/float(len(a))
  if r > thresh:
    return True,r
  return False,r

def map_img_2_col(columns):
  """
  Computes a mapping from image to column where each image is mapped to the column that encodes it with the least error.
  
  @param: columns The columns to obtain error from.
  """
  key2col = {}
  col2keys = dict([(col,[]) for col in columns.keys()])
  col2key_count = dict([(col,0) for col in columns.keys()])
  outputs = np.zeros([DATA_PARAM.batch_size, len(columns)])
  for mb in dp.get_mb():
    for i,column in columns.iteritems():
       tmp = column.per_example_reconstruction_loss(mb[0])
       outputs[:,i] = tmp
      #outputs[:,i] = np.mean(np.max(np.max(act, axis=1), axis=1), axis=1)
    maxvals = np.argmin(outputs,axis=1)
    for key,col in zip(mb[2],maxvals):
      key2col[key] = col
      col2key_count[col] += 1
      col2keys[col].append(key)
  #print "Mapping Stats: ",stats
  return {'key2col':key2col, 'n_examples':col2key_count, "col2key":col2keys}

 
def train(imap, columns, keys, batches):
    """
    Trains each column on the examples mapped to it.
    Training is done such that all columns get an equal number of updates.
    This number is governed by the column with the most examples mapped to it.  
    That column will go through it's examples once while columns with fewer examples will cycle through theirs.
    
    @param: imap  The image to column mapping structure
    @param: columns A list of columns
    @param: keys The list of image keys
    @param: batches The number of batches to train on 
    
    @return: A dictionary of columns to average loss over the training epochs.
    """
    per_column_batch_key_index = [0 for i in columns]
    max_examples = max(imap['n_examples'].values())
    if min(imap['n_examples'].values()) == 0:
      max_examples = len(keys)
      for colnum,n in imap['n_examples'].iteritems():
        if n == 0:
          imap['col2key'][colnum] = keys
          print("\tColumn {} has no examples mapped to it.  Training it on all data".format(colnum))
#     n_updates = max_examples/DATA_PARAM.batch_size
    for b in range(batches):
#         print "Training Batch: {}".format(b)
        losses = dict([(col,0) for col in columns.keys()])
        for colnum,col in columns.iteritems():
          batch_keys = list(islice(cycle(imap['col2key'][colnum]),per_column_batch_key_index[colnum],per_column_batch_key_index[colnum]+DATA_PARAM.batch_size))
          per_column_batch_key_index[colnum] += DATA_PARAM.batch_size
          s,l,k = dp.get_mb_by_keys(batch_keys)
          #print("\tTraining column {} on {} keys".format(colnum,len(batch_keys)))
          losses[colnum] += columns[colnum].train_mb(s)
    return losses            

def get_mapped_batch(dp, column_num, immap):
    keys = immap['col2key'][column_num]
    if len(keys) > dp.shape()[0]:
      keys = keys[:dp.shape()[0]]
    if len(keys)>0:
      sample,l,k = dp.get_mb_by_keys(keys)
      return sample,l,k
    else:
      return np.zeros(0),np.zeros(0),[]
    
def save_recon(dp, columns, immap):
    for n,column in columns.iteritems():
      #Reconstruct mapped examples
      mapped_samples,_,_ = get_mapped_batch(dp, n, immap)
      if len(mapped_samples) == dp.shape()[0]:
        d,r = column.fwd_back(mapped_samples)
        s = list(d.shape)
        s[0] = s[0]*2
        d_r_array = np.empty(s,dtype=d.dtype)
        d_r_array[0::2,:,:,:] = d
        d_r_array[1::2,:,:,:] = r
        im = w2i.tile_imgs(d_r_array)
        im.save(IMG_DIR+'col'+str(n)+'_mapped_recon_level'+str(layer_number+1)+'.png')
        
def save_injection(dp,columns,immap):
    for n,column in columns.iteritems():
      #Reconstruct an injected value
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
      im.save(IMG_DIR+'col'+str(n)+'_level'+str(layer_number+1)+'_decode.png')    

def save_top(dp, columns,immap):
    for n,column in columns.iteritems():
      mapped_samples,_,_ = get_mapped_batch(dp, n, immap)
      mapped_batch = np.zeros(dp.shape())
      mapped_batch[0:len(mapped_samples),:] = mapped_samples
      t = column.fwd(mapped_batch)
      top_shape = column.top_shape()
      if len(top_shape) == 4:
        im = w2i.tile_imgs(t)
  #       im = Image.fromarray(dp.denormalize(c[0,:]).astype(np.uint8).squeeze(),mode='L')
        im.save(IMG_DIR+'col'+str(n)+'_level'+str(layer_number+1)+'_top.png')  
      elif len(top_shape) == 2:
        t = t.reshape([1]+top_shape+[1])
        im = w2i.tile_imgs(t)
        im.save(IMG_DIR+'col'+str(n)+'_level'+str(layer_number+1)+'_top.png')  
      else:  
        print("Top not saved.  Dimenstions {}".format(top_shape)) 

def save_exemplars(dp, columns,immap):
      for i in range(len(columns)):
        sample,l,k = get_mapped_batch(dp, i, immap)
        if k != None and len(k) > 0:  
          im = w2i.tile_imgs(sample)
          im.save(IMG_DIR+"col"+str(i)+"_exemplars.png")
        #display(dp.denormalize(s.run(columns[i].layers[-1].W).transpose([3,0,1,2])))

 
def mapping_stats(mapping):
  counts = {}
  for val in mapping.values():
    while len(counts) <= val:
      counts[len(counts)] = 0
    counts[val] += 1
  return counts


def pretrain_epoch(columns,dp, i):
    print("Pretrain epoch {}".format(i))
    losses = dict([(col,0) for col in columns.keys()])
    n = 0
    for mb in dp.get_mb():
      n += 1
      for colnum,column in columns.iteritems():
        losses[colnum] += column.train_mb(mb[0])
    losses = dict([(col,v/n) for col,v in losses.iteritems()])
    return losses
  
  
def save_embedding(column,dp):
  print("Saving embedding")
  effective_examples = dp.get_n_examples() - dp.get_n_examples()%dp.shape()[0]
  embedding = np.zeros((effective_examples,column.top_shape()[-1]))
  labels = np.zeros((effective_examples), dtype = np.uint32)
  i = 0
  mb_size = dp.shape()[0]
  for mb in dp.get_mb():
    embedding[i:i+mb_size,:] = column.fwd(mb[0])
    labels[i:i+mb_size] = mb[1]
  np.save(path.join(CHECKPOINT_DIR,"embedding"), embedding)
  np.save(path.join(CHECKPOINT_DIR,"embedding_labels"), labels)
   
   
def save_column_means(dp,columns,immap):
  #generate mean image for each column
  means = np.zeros([N_COLUMNS]+list(dp.shape()[1:]))
  for i,(col,keys) in enumerate(immap['col2key'].iteritems()):
    mb = dp.get_mb_by_keys(keys)
    x = np.mean(mb[0],axis=0)
    means[i,:,:,:]  = x
  im = w2i.tile_imgs(means)
  im.save(IMG_DIR+"mean_imgs.png")

def print_column_entropy(dp,columns,immap):
  #Get column entropy
  columnlabels = dict([(col,[0]*NUM_LABELS) for col in columns.keys()])
  for key,col in immap['key2col'].iteritems():
    _,l,_ = dp.get_mb_by_keys([key])
    columnlabels[col][l[0]] += 1
  output = ""
  for col,dat in columnlabels.iteritems():
    output += "Column: "+str(col)+", "
    output += str(dat) + ", "
    s = reduce(add,dat)
    entropies = [0]*len(dat)
    for i in range(len(entropies)):
      v = dat[i]
      if v != 0:
        entropies[i] = -float(v)/s*math.log(float(v)/s)
    entropy = reduce(add, entropies )
    output += "Entropy: {}\n".format(entropy)
  return output


def print_class_entropy(dp,columns,immap):  
  #Get class assignment entropy
  class_assignments = {}
  for key,col in immap['key2col'].iteritems():
    _,l,_ = dp.get_mb_by_keys([key])
    if l[0] not in class_assignments:
      class_assignments[l[0]] = [0]*len(columns.keys())
    class_assignments[l[0]][col] += 1
  output = ""
  for cl,dat in class_assignments.iteritems():
    output += "Class: "+str(cl)+", "
    output += str(dat)+", "
    s = reduce(add,dat)
    entropies = [0]*len(dat)
    for i in range(len(entropies)):
      v = dat[i]
      if v != 0:
        entropies[i] = -float(v)/s*math.log(float(v)/s)
    entropy = reduce(add, entropies )
    output += "Entropy: {}\n".format(entropy)
  return output  

def accuracy(dp,columns,immap): 
  #print accuracy
  #Determined by assuming each column represents it's majority class.
  #Each non-majority class assigned to it will be an error.
  #Get class assignment entropy
  class_assignments = {}
  for key,col in immap['key2col'].iteritems():
    _,l,_ = dp.get_mb_by_keys([key])
    if l[0] not in class_assignments:
      class_assignments[l[0]] = [0]*len(columns.keys())
    class_assignments[l[0]][col] += 1 
  errors = 0
  for true_class in class_assignments:
    errors += reduce(add,class_assignments[true_class]) - max(class_assignments[true_class])
  return 1.0 - float(errors)/dp.get_n_examples()

if __name__ == '__main__':
    if len(sys.argv) < 3:
      print "Usage: python dynamic_columns.py <path to output dirs> <path to data> [<>]"
      sys.exit(-1)
    DATA_PARAM.source = sys.argv[2:]
    BASE_PATH = sys.argv[1]
    #BASE_PATH = "/home/jlovitt/storage/models/kmeans_over_columns"
    LOG_DIR = path.join(BASE_PATH,'log/')
    IMG_DIR =  path.join(BASE_PATH,'img/')
    CHECKPOINT_DIR =  path.join(BASE_PATH,'check/')
    dp = CifarDataProvider(DATA_PARAM,TRANSFORM_PARAM )
    imgkeys = dp.get_keys()
    columns = {}
    with tf.Session() as sess:
      for i in range(N_COLUMNS):
        g = tf.Graph()
        s = tf.Session(graph=g)
        columns[i] = AutoEncoder(s,g,dp,LOG_DIR, CHECKPOINT_DIR, colnum=i)
      print "Columns Initialized"
      
      #Helper Function
      def col2num(col):
        for i,column in columns.iteritems():
          if column == col:
            return i
        return None
      
      #Iterate over layer definitions to build a column
      for layer_number,l in enumerate(LAYERS):
        for column in columns.values():
          column.add_layer(l['Layerdef'], l.get('Mapped',{}).get('Freeze',True))
          column.set_decode(l['Decodedef'])
          column.build()
        print "{} added".format(l['Layerdef'])
        
        if l.get('Use_To_Map_Samples',False):
          
          immap_old = {'key2col':None}
          immap = map_img_2_col(columns)
          
          #Train current layer depth until convergence.
          epoch_num = 0
          n_batches = TRAIN_BATCHES
          stationary_mapping = False
          while(not stationary_mapping and epoch_num < 5):
            print("========= Epoch {} ========".format(epoch_num))
            print("Mapping Distribution " + str(immap['n_examples']))
            loss = train(immap, columns, imgkeys, n_batches)
            print("Encoding loss on mapped examples {}").format(loss)

            epoch_num += n_batches*(float(dp.shape()[0])/dp.get_n_examples())
            n_batches *= D_TRAIN_BATCHES
            immap_old = immap
            immap = map_img_2_col(columns)
            stationary_mapping,stationary_rate = stationary(immap['key2col'], immap_old['key2col'], l.get('Convergence_threshold',0.0))
            print("{} of the examples were stationary in column mapping".format(stationary_rate))
            print("Accuracy: {}".format(accuracy(dp,columns,immap)))
            
            #Visual investigation
            save_recon(dp,columns,immap)   
            save_top(dp,columns,immap)       
            save_exemplars(dp, columns,immap)
            save_column_means(dp,columns,immap)
            
          for column in columns.values():
            column.save()
            
          if N_COLUMNS > 1:
            col_ent = print_column_entropy(dp,columns,immap)
            class_ent = print_class_entropy(dp,columns,immap)
            
            print(col_ent)
            print(class_ent)
            #print column mapping
            with open(IMG_DIR+"col2key",'w') as fout:
              fout.write(col_ent)
              fout.write(class_ent)
              fout.write(str(accuracy(dp,columns,immap)))
              fout.write(str(immap['col2key']))
              
            #generate mean image for each column
            save_column_means(dp,columns,immap)
            print("Accuracy: {}".format(accuracy(dp,columns,immap)))


# def accuracy(column_entropies, immap):
   




