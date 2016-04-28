'''
Created on Apr 27, 2016

@author: jlovitt
'''
import numpy as np
import weights_to_img as w2i
from sys import path
import math


def save_recon(data, column, columnuid, layeruid, save_path):
    #Reconstruction of a batch
    d,r = column.fwd_back(data)
    s = list(d.shape)
    s[0] = s[0]*2
    d_r_array = np.empty(s,dtype=d.dtype)
    d_r_array[0::2,:,:,:] = d
    d_r_array[1::2,:,:,:] = r
    im = w2i.tile_imgs(d_r_array)
    im.save(save_path+'col'+str(columnuid)+'_img_recon_level'+str(layeruid)+'.png')

def save_injection(column, columnuid, layeruid, save_path):
    #Reconstruction of an injected filter
    top_shape = column.top_shape()
    a = np.zeros(top_shape)
    input_shape = column.bottom_shape()
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
    im.save(save_path+'col'+str(columnuid)+'_level'+str(layeruid)+'_inject.png')    

def save_top(data, column, columnuid,  layeruid, save_path):
    t = column.fwd(data)
    top_shape = column.top_shape()
    if len(top_shape) == 4:
      im = w2i.tile_imgs(t)
#       im = Image.fromarray(dp.denormalize(c[0,:]).astype(np.uint8).squeeze(),mode='L')
      im.save(save_path+'col'+str(columnuid)+'_level'+str(layeruid)+'_top.png')
      
def save_embedding(column,dp,uid, save_path):
  print("Saving embedding")
  effective_examples = dp.get_n_examples() - dp.get_n_examples()%dp.shape()[0]
  embedding = np.zeros((effective_examples,column.top_shape()[-1]))
  labels = np.zeros((effective_examples), dtype = np.uint32)
  i = 0
  mb_size = dp.shape()[0]
  for mb in dp.get_mb():
    embedding[i:i+mb_size,:] = column.encode_mb(mb[0])
    labels[i:i+mb_size] = mb[1]
  np.save(path.join(save_path,"embedding_level"+str(uid)), embedding)
  np.save(path.join(save_path,"embedding_labels_level"+str(uid)), labels)
    
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