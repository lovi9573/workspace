'''
Created on Jan 13, 2016

@author: jlovitt
'''

import sys,os,gc
import time
import lmdb
import numpy as np
import numpy.random
import scipy.io as si
from PIL import Image
from google.protobuf import text_format
import cPickle

from caffe import *


class LMDBDataProvider:
    ''' Class for LMDB Data Provider. 

    .. note::
        Layer type in Caffe's configure file: DATA

    '''

    def __init__(self, data_param, transform_param, mm_batch_num=1):
        bp = BlobProto()
        if len(transform_param.mean_file) == 0:
            self.mean_data = np.ones([3, 256, 256], dtype=np.float32)
            assert(len(transform_param.mean_value) == 3)
            self.mean_data[0] = transform_param.mean_value[0]
            self.mean_data[1] = transform_param.mean_value[1]
            self.mean_data[2] = transform_param.mean_value[2]           
        else:
            with open(transform_param.mean_file, 'rb') as f:
                bp.ParseFromString(f.read())
            mean_narray = np.array(bp.data, dtype=np.float32)
            h_w = np.sqrt(np.shape(mean_narray)[0] / 3)
            self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, h_w, h_w])
        self.source = data_param.source
        self.batch_size = data_param.batch_size / mm_batch_num
        self.crop_size = transform_param.crop_size
        self.mirror = transform_param.mirror

    def normalize(self,raw_image):
      return (raw_image.astype(np.float32) - self.mean_data)/127.0
    
    def denormalize(self,normal_image):
      return (normal_image*127.0 +127).astype(np.uint8)

    def get_n_examples(self):
      env = lmdb.open(self.source, readonly=True)
      with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            return len(cursor)
     
    def get_keys(self):
      env = lmdb.open(self.source, readonly=True)
      with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            keys = []
            it = cursor.iternext(keys=True,values=False)
            for k in it:
              keys.append(k)
      return keys
  
    def shape(self):
        return (self.batch_size, self.crop_size, self.crop_size,3)

    def get_mb_by_keys(self,keys):
      env = lmdb.open(self.source, readonly=True)
      samples = np.zeros([len(keys), self.crop_size ** 2 * 3], dtype=np.float32)
      num_label = -1      
      with env.begin(write=False, buffers=False) as txn:
        for i, key in zip(range(len(keys)), keys):
          raw_dat = txn.get(key)
          d = Datum()
          d.ParseFromString(raw_dat) 
          ori_size = np.sqrt(len(d.data) / 3)
          im = self.normalize(np.fromstring(d.data, dtype=np.uint8).reshape([3, ori_size, ori_size]))
          [crop_h, crop_w] = np.random.randint(ori_size - self.crop_size, size=2)
          im_cropped = im[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
          if self.mirror == True and numpy.random.rand() > 0.5:
            im_cropped = im_cropped[:,:,::-1] 
          samples[i, :] = np.transpose(im_cropped,[1,2,0]).reshape(self.crop_size ** 2 * 3).astype(np.float32)
          
          '''
          #output
          imgdata = np.zeros([self.crop_size, self.crop_size, 3], dtype=np.uint8)
          imgdata[:,:,0] = im_cropped[2,:,:]
          imgdata[:,:,1] = im_cropped[1,:,:]
          imgdata[:,:,2] = im_cropped[0,:,:]
          cropimg = Image.fromarray(imgdata)
          nnn = '/home/tianjun/tests/img_%d.jpg' % (count)
          cropimg.save(nnn, format = 'JPEG')
          '''
          
          if num_label == -1:
            num_label = len(d.label)
            labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
          labels[i, :] = d.label
      _shape = list(self.shape())
      _shape[0] = len(keys)
      return (np.reshape(samples, _shape), labels, keys)              
          
          
    def get_mb(self, phase = 'TRAIN'):
        ''' Get next minibatch
        '''
        env = lmdb.open(self.source, readonly=True)
        samples = np.zeros([self.batch_size, self.crop_size ** 2 * 3], dtype=np.float32)
        keys = []
        num_label = -1
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                ori_size = np.sqrt(len(d.data) / 3)
                im = self.normalize(np.fromstring(d.data, dtype=np.uint8).reshape([3, ori_size, ori_size]))
                if phase == 'TRAIN':
                    [crop_h, crop_w] = np.random.randint(ori_size - self.crop_size, size=2)
                else:
                    crop_h = (ori_size - self.crop_size) / 2
                    crop_w = (ori_size - self.crop_size) / 2
                
                im_cropped = im[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
                if self.mirror == True and numpy.random.rand() > 0.5:
                    im_cropped = im_cropped[:,:,::-1]
                
                samples[count, :] = np.transpose(im_cropped,[1,2,0]).reshape(self.crop_size ** 2 * 3).astype(np.float32)
                keys.append(key)
               
                '''
                #output
                imgdata = np.zeros([self.crop_size, self.crop_size, 3], dtype=np.uint8)
                imgdata[:,:,0] = im_cropped[2,:,:]
                imgdata[:,:,1] = im_cropped[1,:,:]
                imgdata[:,:,2] = im_cropped[0,:,:]
                cropimg = Image.fromarray(imgdata)
                nnn = '/home/tianjun/tests/img_%d.jpg' % (count)
                cropimg.save(nnn, format = 'JPEG')
                '''

                if num_label == -1:
                    num_label = len(d.label)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                labels[count, :] = d.label
                
                count = count + 1
                if count == self.batch_size:
                    yield (np.reshape(samples, self.shape() ), labels, keys)
                    keys = []
                    if phase == 'CHECK':
                        while True:
                            yield(np.reshape(samples, self.shape() ), labels, keys)
                    
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                    count = 0
        if count != self.batch_size:
          pass
#             delete_idx = np.arange(count, self.batch_size)
#             yield (np.delete(samples, delete_idx, 0), np.delete(labels, delete_idx, 0), keys)

    def get_multiview_mb(self):
        '''  Multiview testing will get better accuracy than single view testing. For each image,
        it will crop out the left-top, right-top, left-down, right-down, central patches and their
        hirizontal flipped version. The final prediction is averaged according to the 10 views.
        Thus, for each original batch, get_multiview_mb will produce 10 consecutive batches for the batch.
        '''

        env = lmdb.open(self.source, readonly=True)
        view_num = 10
        ori_size = -1
        samples = np.zeros([view_num, self.batch_size, self.crop_size ** 2 * 3], dtype=np.float32)
        num_label = -1
        count = 0
        with env.begin(write=False, buffers=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                d = Datum()
                d.ParseFromString(value)
                if ori_size == -1:
                    ori_size = np.sqrt(len(d.data) / 3)
                    diff_size = ori_size - self.crop_size
                    start_h = [0, diff_size, 0, diff_size, diff_size/2]
                    start_w = [0, 0, diff_size, diff_size, diff_size/2]

                im = self.normalize(np.fromstring(d.data, dtype=np.uint8).reshape([3, ori_size, ori_size]))
                
                for i in range(view_num):
                    crop_h = start_h[i/2]
                    crop_w = start_w[i/2]
                    im_cropped = im[:, crop_h:crop_h+self.crop_size, crop_w:crop_w+self.crop_size]
                    if i%2 == 1:
                        im_cropped = im_cropped[:,:,::-1]
                    samples[i, count, :] = np.transpose(im_cropped,[1,2,0]).reshape(self.crop_size ** 2 * 3).astype(np.float32)
                   
                if num_label == -1:
                    num_label = len(d.label)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                labels[count, :] = d.label
                
                count = count + 1
                if count == self.batch_size:
                    for i in range(view_num):
                        yield (samples[i,:,:], labels)
                    labels = np.zeros([self.batch_size, num_label], dtype=np.float32)
                    count = 0
        if count != self.batch_size:
            delete_idx = np.arange(count, self.batch_size)
            left_samples = np.delete(samples, delete_idx, 1)
            left_labels = np.delete(labels, delete_idx, 0)
            for i in range(view_num):
                yield (left_samples[i,:,:], left_labels)
                
class CifarDataProvider:
  
  def __init__(self, data_param, transform_param, mm_batch_num=1):
    bp = BlobProto()
    if len(transform_param.mean_file) == 0:
        self.mean_data = np.ones([3, 32, 32], dtype=np.float32)
        assert(len(transform_param.mean_value) == 3)
        self.mean_data[0] = transform_param.mean_value[0]
        self.mean_data[1] = transform_param.mean_value[1]
        self.mean_data[2] = transform_param.mean_value[2]           
    else:
        with open(transform_param.mean_file, 'rb') as f:
            bp.ParseFromString(f.read())
        mean_narray = np.array(bp.data, dtype=np.float32)
        h_w = np.sqrt(np.shape(mean_narray)[0] / 3)
        self.mean_data = np.array(bp.data, dtype=np.float32).reshape([3, h_w, h_w])
    self.files = data_param.source
    self.batch_size = data_param.batch_size / mm_batch_num
    self.crop_size = transform_param.crop_size
    self.mirror = transform_param.mirror
    self._dat = {"filename":""}

  def get_n_examples(self):
    return 10000*len(self.files)
   
  def get_keys(self):
    keys = []
    for f in self.files:
      for i in xrange(10000):
            keys.append(f+"_ex{:0>4}".format(i))
    return keys

  def shape(self):
      return (self.batch_size, self.crop_size, self.crop_size,3)

  def normalize(self,raw_image):
    return (raw_image.astype(np.float32) - self.mean_data)/127.0
  
  def denormalize(self,normal_image):
    return (normal_image*127.0 +127).astype(np.uint8)

  def get_mb_by_keys(self,keys):
    samples = np.zeros([len(keys), self.crop_size,self.crop_size,3], dtype=np.float32)
    labels = np.zeros([len(keys)],dtype=np.uint8)
    sorted_keys = sorted(keys)
    for n,key in enumerate(sorted_keys):
      fname,i = key.split("_ex")
      if fname != self._dat['filename']:
        fo = open(fname, 'rb')
        self._dat = cPickle.load(fo)
        self._dat['filename'] = fname
        fo.close()
      mb = self._dat["data"][i,:]
      mb_4 = mb.reshape([1,3,32,32])
      mb_n = self.normalize(mb_4)
      mb_t = mb_n.transpose([0,2,3,1])
      dx,dy = np.random.randint(32 - self.crop_size+1, size=2)
      samples[n,:] = mb_t[:,dx:dx+self.crop_size,dy:dy+self.crop_size,:]
      labels[n] = self._dat["labels"][int(i)]
      return (samples,labels,keys)
    

  def get_mb(self):
    samples = np.zeros([self.batch_size, self.crop_size,self.crop_size,3], dtype=np.float32)
    labels = np.zeros([self.batch_size])
    for filename in self.files:
      fo = open(filename, 'rb')
      datadict = cPickle.load(fo)
      fo.close()
      i = 0
      while i < (10000 - self.batch_size):
        mb = datadict["data"][i:i+self.batch_size,:]
        mb_4 = mb.reshape([self.batch_size,3,32,32])
        mb_n = self.normalize(mb_4)
        mb_t = mb_n.transpose([0,2,3,1])
        dx,dy = np.random.randint(32 - self.crop_size+1, size=2)
        samples = mb_t[:,dx:dx+self.crop_size,dy:dy+self.crop_size,:]
        labels = datadict["labels"][i:i+self.batch_size]
        keys = []
        for n in range(self.batch_size):
          keys.append(filename+"_ex{:0>4}".format(i+n))
        yield (samples,labels,keys)
        i += self.batch_size
          
        
          
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  class Object:
    pass
  DATA_PARAM = Object()
  DATA_PARAM.batch_size = 16
  TRANSFORM_PARAM = Object()
  TRANSFORM_PARAM.mean_file = ""
  TRANSFORM_PARAM.mean_value = [127,127,127]
  TRANSFORM_PARAM.crop_size = 16
  TRANSFORM_PARAM.mirror = False
  DATA_PARAM.source = sys.argv[1:]
  dp = CifarDataProvider(DATA_PARAM,TRANSFORM_PARAM )
  for d,l,k in dp.get_mb(8):
    print k
#     plt.imshow(dp.denormalize(d[0]))
#     plt.colorbar()
#     plt.show()      