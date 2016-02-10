"""
Running kmeans over columns of 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import tensorflow.python.platform
from google.protobuf.text_format import Merge
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from operator import mul,add




class FCLayerDef(object):
    
    def __init__(self, outdim):
      self._outdim = outdim
      self._indim = None
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def instance(self):
        return FCLayer()

    def __str__(self):
      return "Fully Connected layer with output size {}".format(self._outdim)
  
class ConvLayerDef(object):

    def __init__(self,filterdim, stride, outdim):
      self._outdim = outdim
      self._filterdim = filterdim
      self._stride = stride
      self._indim = None
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def filterdim(self):
        return [self._filterdim,self._filterdim]
      
    def strides(self):
        return [1,self._stride,self._stride,1]
      
    def instance(self):
        return ConvLayer()
  
    def __str__(self):
      return "Convolutional layer {}x{}x{} stride {}".format(self._filterdim, self._filterdim,self._outdim,self._stride)
  
class CorruptionLayerDef(object):
    def __init__(self,corruptionlevel):
      self._corruptionlevel = corruptionlevel
  
    def corruptionlevel(self):
      return self._corruptionlevel
      
    def instance(self):
        return CorruptionLayer()
  
    def __str__(self):
      return "Corruption layer at p({})".format(self._corruptionlevel)
    
  
"""
==========================================================================================================
  Layer Implementations
==========================================================================================================
"""
    
class Layer(object):
  def params(self):
    return []

class DataLayer(Layer):
    
    def __init__(self,dp):
        self.dp = dp
        self.datalayer = tf.placeholder(tf.float32, dp.shape(), "data")
        self._recon = None
        self.next = None
        self._back = None

    def set_next(self,n):
        '''
        :param l: The layer that feeds into this one
        :type l: Layer
        '''     
        self.next = n
        if n != self:
            self.next._set_prev(self) 
        
    def top(self):
        return self.datalayer
    
    def build_rev(self):
        self._recon = self.next.recon()
        
    def recon(self):
        return self._recon
      
    def build_back(self):
      self._back = self.next.back()
      
    def back(self):
      return self._back
    
    def bottom_feed(self):
      return self.datalayer 
    
    def params(self):
      return []
 

 
class FeedThroughLayer(Layer):
    
  def __init__(self):
      self._top = None
      self._recon = None
      self._back = None
      self.next = None
      self.prev = None
      self.d = None

  def set_params(self,d,n):
      '''
      :param d: A definition of layer parameters
      :param n: The layer number
      :type d: FCLayerDef
      :type n: Integer
      '''
      self.d = d
      self.n = n
      self.build()        
      
  def set_next(self,n):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      self.next = n
      self.next._set_prev(self)
      if n != self:
          self.next._set_prev(self)    
      
  def _set_prev(self,l):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      self.prev = l
      self.build()   
   
  def top(self):
      return self._top
  
  def recon(self):
      return self._recon  
  
  def back(self):
    return self._back
      

class CorruptionLayer(FeedThroughLayer):
  noiselevel = 0.3
  
  def build(self):
        '''
        Constructs the computation graph for this layer and all subsequent layers.
        '''
        if self.prev and self.next and self.d and self.prev != self: 
          self._indim = self.prev.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.noise = tf.random_uniform(self._indim,
                                         minval = -1,
                                         maxval = 1,
                                         dtype = tf.float32)
          self.p = tf.random_uniform(self._indim,
                                             minval = 0,
                                             maxval = 1,
                                             dtype=tf.float32)
          self.mask = tf.to_float(self.p > self.d.corruptionlevel())
          self.invmask = tf.to_float(self.p <= self.d.corruptionlevel())
          self._top = tf.mul(self.prev.top(),self.invmask) + tf.mul(self.noise, self.mask)
          if self.next != self:
            self.next.build()
          else:
            self.build_loop()

  def build_loop(self):
      self._recon = self.top()
      self.prev.build_rev()
      
  def build_rev(self):
      self._recon = self.next.recon()
      self.prev.build_rev()
  
  def inject(self,i):
      self._back = i
      self.prev.build_back()
      
  def build_back(self):
      self._back = self.next.back()
      self.prev.build_back()
  
  def params(self):
    return []            
  


class FCLayer(FeedThroughLayer): 
        
    def build(self):
        '''
        Constructs the computation graph for this layer and all subsequent layers.
        '''
        if self.prev and self.next and self.d and self.prev != self: 
          self._indim = self.prev.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.W = tf.Variable(
                               tf.random_uniform([inflat, self.d.outdim()],
                                                 minval=-4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 maxval=4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 dtype=tf.float32
                                                 ),
                               name='W_'+str(self.n))
          self.bias = tf.Variable(
                                  tf.zeros([self.d.outdim()]),
                                  name='b_'+str(self.n))
          #print(self.W.get_shape().with_rank(2))
          flat_in = tf.reshape(self.prev.top(),[self._indim[0],-1])
          self._top = tf.tanh(tf.matmul(flat_in,self.W))
          if self.next != self:
              self.next.build()
          else:
              self.build_loop()
                
    def build_loop(self):
        self._recon = tf.reshape(tf.tanh(tf.matmul(self.top(),tf.transpose(self.W))),self._indim)
        self.prev.build_rev()
        
    def build_rev(self):
        self._recon = tf.reshape(tf.tanh(tf.matmul(self.next.recon(),tf.transpose(self.W))),self._indim)
        self.prev.build_rev()
 
    def inject(self,i):
        self._back = tf.reshape(tf.tanh(tf.matmul(i,tf.transpose(self.W))),self._indim)
        self.prev.build_back()
        
    def build_back(self):
        self._back = tf.reshape(tf.tanh(tf.matmul(self.next.back(),tf.transpose(self.W))),self._indim)
        self.prev.build_back()

    def params(self):
      return [self.W, self.bias]
    
    


class ConvLayer(FeedThroughLayer):

        
    def build(self):
        '''
        Constructs the computation graph for this layer and all subsequent layers.
        '''
        if self.prev and self.next and self.d and self.prev != self: 
          indim = self.prev.top().get_shape().as_list()
          inflat = reduce(mul,indim[1:])
          dims = self.d.filterdim()+[indim[3], self.d.outdim()]
          self.W = tf.Variable(
                               tf.truncated_normal(dims,
                                                 stddev=math.sqrt(1.0/(reduce(mul,self.d.filterdim()))),
                                                 dtype=tf.float32
                                                 ),
                               name='W_'+str(self.n))
          self.bias = tf.Variable(
                                 tf.truncated_normal([self.d.outdim()],
                                                 stddev=0.1,
                                                 dtype=tf.float32
                                                 )
                                  )
          self._top = tf.tanh(tf.nn.conv2d(self.prev.top(), self.W, self.d.strides(), "SAME")+self.bias)
          if self.next != self:
              self.next.build()
          else:
              self.build_loop()

    def build_loop(self):
        self._recon = tf.tanh(tf.nn.deconv2d(self.top(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
        self.prev.build_rev()
        
    def build_rev(self):
        self._recon = tf.tanh(tf.nn.deconv2d(self.next.recon(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
        self.prev.build_rev()
 
    def inject(self,i):
        self._back = tf.tanh(tf.nn.deconv2d(i, tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
        self.prev.build_back()
        
    def build_back(self):
        self._back = tf.tanh(tf.nn.deconv2d(self.next.back(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
        self.prev.build_back()
        
    def params(self):
      return [self.W,self.bias]

class PoolingLayer(Layer):
  
  def build(self):
      '''
      Constructs the computation graph for this layer and all subsequent layers.
      '''
      if self.prev and self.next and self.d and self.prev != self: 
        indim = self.prev.top().get_shape().as_list()
        inflat = reduce(mul,indim[1:])
                                
        self._top = tf.nn.max_pool(self.prev.top(),
                                   ksize=[1,self.d.size,self.d.size,1],
                                   strides=[1,self.d.stride,self.d.stride,1],
                                   padding=[1,self.d.padding,self.d.padding,1],
                                   name = "Pool {}".format(self.n))
        if self.next != self:
            self.next.build()
        else:
            self.build_loop()

  def build_loop(self):
      self._recon = tf.tanh(tf.nn.deconv2d(self.top(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
      self.prev.build_rev()
      
  def build_rev(self):
      self._recon = tf.tanh(tf.nn.deconv2d(self.next.recon(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
      self.prev.build_rev()
    
  def inject(self,i):
      self._back = tf.tanh(tf.nn.deconv2d(i, tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
      self.prev.build_back()
      
  def build_back(self):
      self._back = tf.tanh(tf.nn.deconv2d(self.next.back(), tf.transpose(self.W, [1,0,2,3] ), self.prev.top().get_shape().as_list(), self.d.strides()))
      self.prev.build_back()
      
  def params(self):
    return []
  


class AutoEncoder(object):
    
    def __init__(self,s, dp):
        self.dp = dp
        self.s = s
        self.layers = [DataLayer(self.dp)]
        self.bottom_feed = self.layers[0].bottom_feed()
        self.LEARNING_RATE=0.7
        self.alpha = 0.1
        self.freeze = False
        self.sparsity_target = 0.05
        self.sparsity_lr = 6

    def kl(self,p, p_hat):
      a = p*tf.log(tf.div(p,p_hat))
      b = (1-p)*tf.log(tf.div((1-p),(1-p_hat)))
      return tf.reduce_mean(a+b)
        
    def add_layer(self,definition):
      #Insert into stack
      l = definition.instance()
      l.set_params(definition, len(self.layers))
      l.set_next(l)
      self.layers[-1].set_next(l)
      self.layers.append(l)
      l.build()
      
      #Reference key layer inputs/outputs
      self._top = self.layers[-1].top()
      self.injection = tf.placeholder(tf.float32, self._top.get_shape().as_list(), "data")
      l.inject(self.injection)
      self._recon = self.layers[0].recon()
      depth = min(len(self.layers)-1,10)
      self._rep_recon = self.layers[-depth].recon()
      self.ground_truth = (self.layers[-depth]).prev.top()
      self.back = self.layers[0].back()
      
      #Build loss and optimization functions
      self._individual_reconstruction_loss = tf.reduce_mean(tf.pow(self._recon - self.bottom_feed,2),
                                                      reduction_indices=[1,2,3] )
      reconstruction_loss = tf.reduce_mean(tf.pow(self._rep_recon - self.ground_truth,2))
      sparsity_loss = reduce(add, 
                             [tf.reduce_mean(tf.pow(tf.sub(
                                      self.sparsity_target,
                                      tf.abs(
                                             tf.reduce_mean(
                                                            l.top(),
                                                            reduction_indices=[0,1,2]
                                                            )
                                             )
                                      ),2))
                               for l in self.layers])
      params = [tf.reduce_sum(tf.pow(w,2)) for l in self.layers for w in l.params()]
      weight_decay = 0
      if len(params) > 0:
        paramsize = reduce(add,[reduce(mul,w.get_shape().as_list())  for l in self.layers for w in l.params() ])
        weight_decay = reduce(add,params)/paramsize
        self._loss = reconstruction_loss + self.alpha*weight_decay + self.sparsity_lr*sparsity_loss
        # Optimization
        parameterlayers = self.layers
        if self.freeze:
          parameterlayers = [self.layers[-1]]
        self.optimizer = tf.train.MomentumOptimizer(self.LEARNING_RATE,0.9,use_locking=True) \
                                  .minimize(self._loss, var_list=[w for l in parameterlayers for w in l.params()])
    
    def top_shape(self):
      return self.injection.get_shape().as_list()
    
    def fwd(self,data):
      return self.s.run(self._top, feed_dict={self.bottom_feed:data})
  
    def inject(self,data):
      return self.s.run(self.back, feed_dict={self.injection:data})
    
    def recon(self,data):
          feed_dict = {self.bottom_feed:data}
          _recon = self.s.run(self._recon,feed_dict=feed_dict)
          return (data,_recon)    
       
    def loss(self,data):
        feed_dict = {self.bottom_feed:data}
        l = self.s.run(self._loss,feed_dict=feed_dict)
        return l
      
    def individual_reconstruction_loss(self,data):
        feed_dict = {self.bottom_feed:data}
        l = self.s.run(self._individual_reconstruction_loss,feed_dict=feed_dict)
        return l    
        
    def encode_mb(self,data): 
          feed_dict = {self.bottom_feed:data}
          dummy, l = self.s.run([self.optimizer,self._loss],feed_dict=feed_dict)
          #print("_loss: {}".format(_loss))
          return l
          

#tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
#FLAGS = tf.app.flags.FLAGS

# 
# def main(argv=None):  # pylint: disable=unused-argument
#   # Get the data.
#   train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
#   test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
# 
#   # Extract it into numpy arrays.
#   train_data = extract_data(train_data_filename, 60000)
# 
#   # Generate a validation set.
#   train_data = train_data[:-VALIDATION_SIZE, :]
#   num_epochs = NUM_EPOCHS
#   train_size = train_data.shape[0]
#   print("Training data samples: {}".format(train_size))
# 
#   #Setup persistent vars
#   visible = tf.placeholder(tf.float32,shape=(BATCH_SIZE,N_VISIBLE))
#   corruption_level = tf.placeholder(tf.float32,shape=())
#   weights = tf.Variable(
#       tf.random_uniform([NUM_HIDDEN, N_VISIBLE],
#                           minval=-4.0*math.sqrt(6.0/(NUM_HIDDEN + N_VISIBLE)),
#                           maxval=4.0*math.sqrt(6.0/(NUM_HIDDEN + N_VISIBLE)),
# 			  dtype=tf.float32,
#                           seed=SEED))
#   bias_h = tf.Variable(tf.zeros([NUM_HIDDEN]))
#   bias_v = tf.Variable(tf.zeros([N_VISIBLE]))
# 
#   def v_h(v):
#     h = tf.sigmoid(tf.matmul(v,weights,transpose_b=True) +bias_h)
#     return h
#   
#   def h_r(h):
#     r = tf.sigmoid(tf.matmul(h,weights) + bias_v )
#     return r
# 
#   def kl(p, p_hat):
#     a = p*tf.log(tf.div(p,p_hat))
#     b = (1-p)*tf.log(tf.div((1-p),(1-p_hat)))
#     return tf.reduce_mean(a+b)
#   
#   def cross_entropy(v,r):
#     e = 0.00001
#     return -tf.reduce_mean(tf.reduce_sum((v*tf.log(r+e) + (1-v)*tf.log(1-r-e)), reduction_indices=1))
#   
#   def weight_decay(W,b_v,b_h):
#     return tf.reduce_sum(tf.pow(W,2)) + tf.reduce_sum(tf.pow(b_h,2))+tf.reduce_sum(tf.pow(b_v,2))
# 
#   def bernoulli(shape, thresh):
#     n = tf.random_uniform(shape)
#     return tf.to_float(n > thresh)
# 
#   def model(d, train=False):
#     """The Model definition."""
#     data = d*bernoulli(tf.shape(d),corruption_level)
#     h = v_h(data) 
#     r = h_r(h) 
#     return data, h, r
#   
#   batch = tf.Variable(0)
#   sparsity_learning_rate= tf.train.exponential_decay(
# 		SPARSITY_LR,
# 		batch * BATCH_SIZE,
# 		train_size,
# 		0.4,
# 		staircase=True)
# 
#   # Build computation graph
#   v, h, r = model(visible, True)
#   print("Using sparsity target: {}".format(SPARSITY_TARGET))
#   err_cost = cross_entropy(v,r)
#   sparsity = sparsity_learning_rate*kl(SPARSITY_TARGET,tf.reduce_mean(h,0))
#   weight_cost = LAMBDA*weight_decay(weights,bias_v,bias_h)
#   _loss = err_cost + sparsity + weight_cost
#           
#   
#   # Learning rate scheduling
#   learning_rate = tf.train.exponential_decay(
#       LR,  # Base learning rate.
#       batch * BATCH_SIZE,  # Current index into the dataset.
#       train_size,  # Decay step.
#       0.999,  # Decay rate.
#       staircase=True)
#  
# 	
# 
# 
#   # Create a local session to run this computation.
#   with tf.Session() as s:
#     tf.initialize_all_variables().run()
#     print('Initialized!')
#     costs = {"err":[],
#              "sparsity": [],
#              "weight": []}
#     display_step = 1
#     for step in xrange(num_epochs * train_size // BATCH_SIZE):
#       offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
#       batch_data = train_data[offset:(offset + BATCH_SIZE), :]
#       feed_dict = {visible: batch_data, corruption_level:step/(num_epochs * train_size // BATCH_SIZE)}
#       _, l, lr,slr = s.run(
#           [optimizer, _loss, learning_rate,sparsity_learning_rate],
#           feed_dict=feed_dict)
#       e = s.run(err_cost,feed_dict=feed_dict)
#       sp = s.run(sparsity,feed_dict=feed_dict)
#       w = s.run(weight_cost,feed_dict=feed_dict)
#       costs["err"].append(e)
#       costs["sparsity"].append(sp)
#       costs["weight"].append(w)
#       if step % REPORT == 0:
#         print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
#         print('Minibatch _loss:(err,sparsity,weight) %.3f: %3f,%3f,%3f   learning rate: %.6f, sparsity_lr %.6f' % (l, e, sp, w, lr, slr))
#       if step == display_step or (step+1)%(1*train_size // BATCH_SIZE) == 0:
#         #display(s.run(visiblevar).reshape([BATCH_SIZE,28,28,1]))
#         ht = s.run(h,feed_dict=feed_dict).reshape([1,BATCH_SIZE,NUM_HIDDEN,1])
#         display(ht)
#         if SPARSITY:
#           bh = s.run(tf.reduce_mean(h,0),feed_dict=feed_dict).reshape([NUM_HIDDEN,1,1,1])
#           display(bh)
#         vt = s.run(v,feed_dict=feed_dict).reshape([BATCH_SIZE,28,28,1])
#         rt = s.run(r,feed_dict=feed_dict).reshape([BATCH_SIZE,28,28,1])
#         display(numpy.append(vt,rt,0))
#         w = s.run(weights,feed_dict=feed_dict).reshape([NUM_HIDDEN,28,28,1])
#         b_v = s.run(bias_v,feed_dict=feed_dict).reshape([1,28,28,1])
#         #display(numpy.append(w, b_v, 0))
# 	display(w)
#         plt.plot(costs["err"],'r', costs["sparsity"],'b', costs["weight"],'k')
#         x0,x1,y0,y1 = plt.axis()
#         plt.axis((x0,x1,y0,0.5*y1))
#         plt.show()
#         sys.stdout.flush()
#         #display_step += int(REPORT*math.log(4*step+5))
#         

if __name__ == '__main__':
  class Object:
    pass
  from dataio import LMDBDataProvider, CifarDataProvider
  import numpy as np
  N_COLUMNS = 3
  N_STEPS = 1
  DATA_PARAM = Object()
  DATA_PARAM.batch_size = 128
  TRANSFORM_PARAM = Object()
  TRANSFORM_PARAM.mean_file = ""
  TRANSFORM_PARAM.mean_value = [127,127,127]
  TRANSFORM_PARAM.crop_size = 32
  TRANSFORM_PARAM.mirror = False
  DATA_PARAM.source = sys.argv[1:]
  dp = CifarDataProvider(DATA_PARAM,TRANSFORM_PARAM )
  with tf.Session() as s:
    ae = AutoEncoder(s,dp)
    ae.add_layer(ConvLayerDef(3,1,32))
    ae.add_layer(ConvLayerDef(3,1,32))
    ae.add_layer(ConvLayerDef(3,1,32))
    #ae.add_layer(FCLayerDef(64))
    tf.initialize_all_variables().run()
    for mb in dp.get_mb():
      for e in range(200):
        print("Epoch {}".format(e))
        print(ae.encode_mb(mb[0]))
        if e%10 == 0:
          d,r = ae.recon(mb[0])
          plt.imshow(dp.denormalize(np.append(d[0],r[0],axis=0)[:,:,::-1]), cmap="Greys")
          plt.colorbar()
          plt.show()
        
  
