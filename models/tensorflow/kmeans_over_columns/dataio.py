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
import gzip
import weights_to_img as w2i

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
    self.cache_data()
    self.data = self.normalize(self.data)
    self.data = self.data.reshape([-1,3,32,32])
    self.data = self.data.transpose([0,2,3,1])

  def cache_data(self):
    self.data = None
    for filename in self.files:
      fo = open(filename, 'rb')
      datadict = cPickle.load(fo)
      fo.close()
      if not self.data:
        self.data = datadict['data']
        self.labels = np.asarray(datadict['labels'],dtype=np.int)
        self.keys = [ filename+"_ex{:0>4}".format(n) for n in range(len(datadict['data']))]
      else:
        self.data = np.append(self.data, datadict['data'], axis=0)
        self.labels = np.append(self.labels,datadict['labels'], axis=0)
        self.keys += [ filename+"_ex{:0>4}".format(n) for n in range(len(datadict['data']))]

  def get_n_examples(self):
    return len(self.labels)
  
   
  def get_keys(self):
    return self.keys

  def shape(self):
      return (self.batch_size, self.crop_size, self.crop_size,3)

  def normalize(self,raw_image):
    return (raw_image.astype(np.float32))/255.0
  
  def denormalize(self,normal_image):
    return (normal_image*255.0).astype(np.uint8)

  def get_mb_by_keys(self,keys):
    samples = np.zeros([len(keys), self.crop_size,self.crop_size,3], dtype=np.float32)
    labels = np.zeros([len(keys)],dtype=np.uint8)
    sorted_keys = sorted(keys)
    for n,key in enumerate(sorted_keys):
      i = self.keys.index(key)
      mb = self.data[i,:,:,:]
#       mb_4 = mb.reshape([1,3,32,32])
#       mb_n = self.normalize(mb_4)
#       mb_t = mb_4.transpose([0,2,3,1])
      dx,dy = np.random.randint(32 - self.crop_size+1, size=2)
      samples[n,:,:,:] = mb[dx:dx+self.crop_size,dy:dy+self.crop_size,:]
      labels[n] = self.labels[i]
    return (samples,labels,keys)
    

  def get_mb(self):
    samples = np.zeros([self.batch_size, self.crop_size,self.crop_size,3], dtype=np.float32)
    lbls = np.zeros([self.batch_size])
    i = 0
    while i < (self.get_n_examples() - self.batch_size):
      mb = self.data[i:i+self.batch_size,:]
#       mb_4 = mb.reshape([self.batch_size,3,32,32])
#       mb_n = self.normalize(mb_4)
#       mb_t = mb_4.transpose([0,2,3,1])
      dx,dy = np.random.randint(32 - self.crop_size+1, size=2)
      samples = mb[:,dx:dx+self.crop_size,dy:dy+self.crop_size,:]
      lbls = self.labels[i:i+self.batch_size]
      keys = self.keys[i:i+self.batch_size]
      yield (samples,lbls,keys)
      i += self.batch_size
          
  
  
  
class MnistDataProvider:
  
  def __init__(self, data_param, transform_param, mm_batch_num=1):
    self.files = data_param.source
    self.batch_size = data_param.batch_size / mm_batch_num
    self.crop_size = transform_param.crop_size
    self.mirror = transform_param.mirror
    self._data = self.extract_data(self.files[0],self.get_n_examples())
    self._labels = self.extract_labels(self.files[1], self.get_n_examples())
  
  
  def extract_data(self,filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    Values are rescaled from [0, 255] down to [0, 1.0].
    """
    print('Extracting', filename)
    dim = self.shape()[1]
    with gzip.open(filename) as bytestream:
      bytestream.read(16)
      buf = bytestream.read(dim * dim * num_images)
      data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
#       data = data  /  255
      data = data.reshape(num_images, dim, dim, 1)
      data = self.normalize(data)
      return data   
    
  def extract_labels(self, filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
      bytestream.read(8)
      buf = bytestream.read(1 * num_images)
      labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return labels
  
  def get_n_examples(self):
    return 60000
   
  def get_keys(self):
    return range(self.get_n_examples())

  def shape(self):
      return (self.batch_size, 28,28,1)

  def normalize(self,raw_image):
    return (raw_image.astype(np.float32))/255.0
  
  def denormalize(self,normal_image):
    return (normal_image*255.0).astype(np.uint8)

  def get_mb_by_keys(self,keys):
    samples = np.zeros([len(keys)] +list(self.shape()[1:]), dtype=np.float32)
    labels = np.zeros([len(keys)],dtype=np.uint8)
    sorted_keys = sorted(keys)
    for n,key in enumerate(sorted_keys):
      i = int(key)
      samples[n,:,:,:] = self._data[i,:,:,:]
      labels[n] = self._labels[i]
    return (samples,labels,keys)
    

  def get_mb(self):
    samples = np.zeros(self.shape(), dtype=np.float32)
    labels = np.zeros([self.batch_size])
    i = 0
    while i < (self.get_n_examples() - self.batch_size):
      samples[:,:,:,:] = self._data[i:i+self.batch_size,:,:,:]
      labels = self._labels[i:i+self.batch_size]
      keys = range(i,i+self.batch_size)
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
  TRANSFORM_PARAM.crop_size = 32
  TRANSFORM_PARAM.mirror = False
  DATA_PARAM.source = sys.argv[1:]
  dp = CifarDataProvider(DATA_PARAM,TRANSFORM_PARAM )
  dat,l,k = dp.get_mb().next()
  dat,l,k = dp.get_mb_by_keys(k)
  im = w2i.display(dp.denormalize(dat))
  for i in range(0):
    d = dat[i,:]
    print d.shape
    plt.imshow(dp.denormalize(d))
    plt.colorbar()
    plt.show()      
    
    