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
from operator import mul




class LayerDef(object):
    
    def __init__(self,t, outdim):
      self._type = t
      self._outdim = outdim
      self._indim = None
        
    
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def instance(self):
      if self._type == "FC":
        return FCLayer()
    
class Layer(object):
    pass

class DataLayer(Layer):
    
    def __init__(self,dp):
        self.dp = dp
        self.datalayer = tf.placeholder(tf.float32, dp.shape(), "data")
        self._recon = None
        self.next = None

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
    
    def bottom(self):
      return self.datalayer 
      

class FCLayer(Layer):

    def __init__(self):
        self._top = None
        self.next = None
        self.prev = None
        self.d = None
    
    def set_params(self,d):
        '''
        :param d: A definition of layer parameters
        :type d: LayerDef
        '''
        self.d = d
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
        
    def build(self):
        if self.prev and self.next and self.d and self.prev != self: 
          indim = self.prev.top().get_shape().as_list()
          inflat = reduce(mul,indim[1:])
          self.W = tf.Variable(
                               tf.random_uniform([inflat, self.d.outdim()],
                                                 minval=-4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 maxval=4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 dtype=tf.float32,
                                                 seed=0)
                               )
          self.bias = tf.Variable(tf.zeros([self.d.outdim()]))
          self._top = tf.sigmoid(tf.matmul(self.prev.top(),self.W)+self.bias)
          if self.next != self:
              self.next.build()
          else:
              self.build_loop()
                
    def build_loop(self):
        self._recon = tf.sigmoid(tf.matmul(self.top(),tf.transpose(self.W)))
        self.prev.build_rev()
        
    def build_rev(self):
        self._recon = tf.sigmoid(tf.matmul(self.next.recon(),tf.transpose(self.W)))
        self.prev.build_rev()
        
    def top(self):
        return self._top
    
    def recon(self):
        return self._recon
    
    


class ConvLayer(Layer):
    
    def __init__(self,):
        pass


class AutoEncoder(object):
    
    def __init__(self, dp):
        self.dp = dp
        self.column = [DataLayer(self.dp)]
        self.bottom = self.column[0].bottom()
        
    def add_layer(self,definition):
        l = definition.instance()
        l.set_params(definition)
        l.set_next(l)
        self.column[-1].set_next(l)
        self.column.append(l)
        l.build()
        self.top = self.column[-1].top()
        self.recon = self.column[0].recon()
    
    def fwd(self,data):
      pass
        
            

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
#   loss = err_cost + sparsity + weight_cost
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
#   # Optimization
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate,use_locking=True).minimize(loss,
#                                                        var_list=[weights], #TODO: put bias' back in.
#                                                        global_step=batch)
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
#           [optimizer, loss, learning_rate,sparsity_learning_rate],
#           feed_dict=feed_dict)
#       e = s.run(err_cost,feed_dict=feed_dict)
#       sp = s.run(sparsity,feed_dict=feed_dict)
#       w = s.run(weight_cost,feed_dict=feed_dict)
#       costs["err"].append(e)
#       costs["sparsity"].append(sp)
#       costs["weight"].append(w)
#       if step % REPORT == 0:
#         print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
#         print('Minibatch loss:(err,sparsity,weight) %.3f: %3f,%3f,%3f   learning rate: %.6f, sparsity_lr %.6f' % (l, e, sp, w, lr, slr))
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
    pass
