"""
Running kmeans over columns of 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
from os import path
import tensorflow.python.platform
from google.protobuf.text_format import Merge
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from operator import mul,add
from scipy import stats


class LayerDef():
  
  def __init__(self,params):
    self.lr = params.get('lr',0.1)
    self.ALPHA = params.get('ALPHA',0.0)
    self.sparsity_target = params.get('sparsity_target',0.01)
    self.sparsity_lr = params.get('sparsity_lr',0.0)
    self.activation_entropy_lr = params.get('activation_entropy_lr',0.0)
    self.activation_function = params.get('activation_function',tf.sigmoid)
    


class FCLayerDef(LayerDef):
    
    def __init__(self, outdim, **params):
      LayerDef.__init__(self,params)
      self._outdim = outdim
      self._indim = None
      self._tied_weights = params.get('tied_weights', True)
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def instance(self,g):
        return FCLayer(g)

    def __str__(self):
      return "Fully Connected layer with output size {}".format(self._outdim)
  
class ConvLayerDef(LayerDef):

    def __init__(self,filterdim, stride, outdim, **params):
      LayerDef.__init__(self,params)
      self._outdim = outdim
      self._filterdim = filterdim
      self._stride = stride
      self._padding = params.get('padding','VALID')
      self._tied_weights = params.get('tied_weights', True)
      self._indim = None
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def filterdim(self):
        return [self._filterdim,self._filterdim]
      
    def strides(self):
        return [1,self._stride,self._stride,1]
      
    def instance(self, g):
        return ConvLayer(g)
  
    def __str__(self):
      return "Convolutional layer {}x{}x{} stride {}".format(self._filterdim, self._filterdim,self._outdim,self._stride)
  
class CorruptionLayerDef(LayerDef):
    def __init__(self,corruptionlevel, **params):
      LayerDef.__init__(self,params)
      self._corruptionlevel = corruptionlevel
  
    def corruptionlevel(self):
      return self._corruptionlevel
      
    def instance(self, g):
        return CorruptionLayer(g)
  
    def __str__(self):
      return "Corruption layer at p({})".format(self._corruptionlevel)
    
class FeedThroughLayerDef(LayerDef):
    def __init__(self, **params):
      LayerDef.__init__(self,params)
      
    def instance(self):
        return FeedThroughLayer()
  
    def __str__(self):
      return "Feed Through layer"
  
"""
==========================================================================================================
  Layer Implementations
==========================================================================================================
"""
    
class Layer(object):
  def params(self):
    return []

class DataLayer(Layer):
    
    def __init__(self,dp,g):
        self.dp = dp
        with g.as_default():
          self.datalayer = tf.placeholder(tf.float32, dp.shape(), "data")
        self._recon = None
        self._recon = None
        self._inject_recon = None

    def get_top(self,n):
        '''
        :param l: The layer that feeds into this one
        :type l: Layer
        '''     
        self._recon = n
        if n != self:
            self._recon.set_bottom(self) 
        
    def top(self):
        return self.datalayer
    
    def truth(self):
        return self.top()
    
    def build_rev(self):
        self._recon = self._recon.get_recon()
        
    def get_recon(self):
        return self._recon
      
    def build_back(self):
      self._inject_recon = self._recon.get_inject_recon()
      
    def get_inject_recon(self):
      return self._inject_recon
    
    def bottom_feed(self):
      return self.datalayer 
    
    def params(self):
      return []
 

 
class FeedThroughLayer(Layer):
    
  def __init__(self,g):
      self.g = g
      self._top = None
      self._recon = None
      self._inject_recon = None
      self._recon = None
      self._bottom = None
      self.d = None

  def set_params(self,d,n):
      '''
      :param d: A definition of layer parameters
      :param _uid: The layer number
      :type d: FCLayerDef
      :type _uid: Integer
      '''
      self.d = d
      self._uid = n
      self.build_fwd()        
      
  def get_top(self,n):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      self._recon = n
      self._recon.set_bottom(self)
      if n != self:
          self._recon.set_bottom(self)    
      
  def set_bottom(self,l):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      self._bottom = l
      #self.build_fwd()   
      
  def build_fwd(self):
      if self._bottom and self._recon and self._bottom != self: 
        self._top = self._bottom.top()
        if self._recon != self:
          self._recon.build_fwd()
        else:
          self.build_loop()

  def build_loop(self):
      self._recon = self.top()
      self._bottom.build_rev()
      
  def build_rev(self):
      self._recon = self._recon.get_recon()
      self._bottom.build_rev()
  
  def set_inject_embedding(self,i):
      self._inject_recon = i
      self._bottom.build_back()
      
  def build_back(self):
      self._inject_recon = self._recon.get_inject_recon()
      self._bottom.build_back()
   
  def top(self):
      return self._top
    
  def truth(self):
      return self.top()
  
  def get_recon(self):
      return self._recon  
  
  def get_inject_recon(self):
    return self._inject_recon
  
  def params(self):
    return []    
  
  def __str__(self):
    return self.d.__str__() 
      

class CorruptionLayer(FeedThroughLayer):
  noiselevel = 0.3
  
  def build_fwd(self):
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self._bottom and self._recon and self.d and self._bottom != self: 
          self._indim = self._bottom.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.noise = tf.random_uniform(self._indim,
                                         minval = -1,
                                         maxval = 1,
                                         dtype = tf.float32)
          self.p = tf.random_uniform(self._indim,
                                             minval = 0,
                                             maxval = 1,
                                             dtype=tf.float32)
          self.mask = tf.to_float(self.p < self.d.corruptionlevel())
          self.invmask = tf.to_float(self.p >= self.d.corruptionlevel())
          self._top = tf.mul(self._bottom.top(),self.invmask) + tf.mul(self.noise, self.mask)
          if self._recon != self:
            self._recon.build_fwd()
          else:
            self.build_loop()

    

class FCLayer(FeedThroughLayer): 
        
    def build_fwd(self):
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self._bottom and self._recon and self.d and self._bottom != self: 
          self._indim = self._bottom.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.W = tf.Variable(
                               tf.random_uniform([inflat, self.d.outdim()],
                                                 minval=-4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 maxval=4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 dtype=tf.float32
                                                 ),
                               name='Weights_'+str(self._uid))
          self.bias = tf.Variable(
                                  tf.zeros([self.d.outdim()]),
                                  name='bias_'+str(self._uid))
          self.rev_bias = tf.Variable(
                                  tf.zeros([self._indim[-1]]),
                                  name='rev_bias_'+str(self._uid))
          flat_in = tf.reshape(self._bottom.top(),[self._indim[0],-1])
          self._top = self.d.activation_function(tf.add(tf.matmul(flat_in,self.W),self.bias))
          if self._recon != self:
              self._recon.build_fwd()
          else:
              self.build_loop()
                
    def build_loop(self):
        self._recon = self.compute_back(self.top()) 
        self._bottom.build_rev()
        
    def build_rev(self):
        self._recon = self.compute_back(self._recon.get_recon()) 
        self._bottom.build_rev()
 
    def set_inject_embedding(self,i):
        self._inject_recon = self.compute_back(i)
        self._bottom.build_back()
        
    def build_back(self):
        self._inject_recon = self.compute_back(self._recon.get_inject_recon())
        self._bottom.build_back()
    
    def compute_back(self,top):
      with self.g.as_default():
        return self.d.activation_function(tf.add(tf.reshape(tf.matmul(top,tf.transpose(self.W)),self._indim),self.rev_bias))

    def params(self):
      return [self.W, self.bias, self.rev_bias]
    
    


class ConvLayer(FeedThroughLayer):

        
    def build_fwd(self):
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self._bottom and self._recon and self.d and self._bottom != self: 
          indim = self._bottom.top().get_shape().as_list()
          inflat = reduce(mul,indim[1:])
          dims = self.d.filterdim()+[indim[3], self.d.outdim()]
          self.W = tf.Variable(
                               tf.truncated_normal(dims,
                                                 stddev=math.sqrt(3.0/(reduce(mul,self.d.filterdim()))),
                                                 dtype=tf.float32
                                                 ),
                               name='W_'+str(self._uid))
          if not self.d._tied_weights:
            self.rev_W = tf.Variable(
                     tf.truncated_normal(dims,
                                       stddev=math.sqrt(3.0/(reduce(mul,self.d.filterdim()))),
                                       dtype=tf.float32
                                       ),
                     name='W_rev_'+str(self._uid))
          self.bias = tf.Variable(
                                 tf.zeros([self.d.outdim()],
                                          name='bias_'+str(self._uid))
                                  )
          self.rev_bias = tf.Variable(
                                 tf.zeros([indim[-1]],
                                          name='rev_bias_'+str(self._uid))
                                  )
          self._top = self.d.activation_function(tf.nn.conv2d(self._bottom.top(), self.W, self.d.strides(),self.d._padding, name="Conv_"+str(self._uid)+"_top")+self.bias)
          if self._recon != self:
              self._recon.build_fwd()
          else:
              self.build_loop()

    def build_loop(self):
        self._recon = self.compute_back(self.top())
        self._bottom.build_rev()
        
    def build_rev(self):
        self._recon = self.compute_back(self._recon.get_recon())
        self._bottom.build_rev()
 
    def set_inject_embedding(self,i):
        self._inject_recon = self.compute_back(i)
        self._bottom.build_back()
        
    def build_back(self):
        self._inject_recon = self.compute_back(self._recon.get_inject_recon())
        self._bottom.build_back()
        
    def compute_back(self, top):
      with self.g.as_default():
        w = self.W
        if not self.d._tied_weights:
          w = self.rev_W
        return self.d.activation_function(tf.nn.deconv2d(top, tf.transpose(w, [1,0,2,3] ), self._bottom.top().get_shape().as_list(), self.d.strides(), padding=self.d._padding)+self.rev_bias)
          
    def params(self):
      ret =  [self.W,self.bias, self.rev_bias]
      if not self.d._tied_weights:
        ret += [self.rev_W]
      return ret

class PoolingLayer(Layer):
  
  def build_fwd(self):
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      if self._bottom and self._recon and self.d and self._bottom != self: 
        indim = self._bottom.top().get_shape().as_list()
        inflat = reduce(mul,indim[1:])
                                
        self._top = tf.nn.max_pool(self._bottom.top(),
                                   ksize=[1,self.d.size,self.d.size,1],
                                   strides=[1,self.d.stride,self.d.stride,1],
                                   padding=[1,self.d.padding,self.d.padding,1],
                                   name = "Pool {}".format(self._uid))
        if self._recon != self:
            self._recon.build_fwd()
        else:
            self.build_loop()

  def build_loop(self):
      self._recon = tf.sigmoid(tf.nn.deconv2d(self.top(), tf.transpose(self.W, [1,0,2,3] ), self._bottom.top().get_shape().as_list(), self.d.strides()))
      self._bottom.build_rev()
      
  def build_rev(self):
      self._recon = tf.sigmoid(tf.nn.deconv2d(self._recon.get_recon(), tf.transpose(self.W, [1,0,2,3] ), self._bottom.top().get_shape().as_list(), self.d.strides()))
      self._bottom.build_rev()
    
  def set_inject_embedding(self,i):
      self._inject_recon = tf.sigmoid(tf.nn.deconv2d(i, tf.transpose(self.W, [1,0,2,3] ), self._bottom.top().get_shape().as_list(), self.d.strides()))
      self._bottom.build_back()
      
  def build_back(self):
      self._inject_recon = tf.sigmoid(tf.nn.deconv2d(self._recon.get_inject_recon(), tf.transpose(self.W, [1,0,2,3] ), self._bottom.top().get_shape().as_list(), self.d.strides()))
      self._bottom.build_back()
      
  def params(self):
    return []

def entropy(a):
  return -tf.mul(a,tf.log(a+0.000001))

"""
======================================================================================================
DECODER
======================================================================================================
Caution!  This part was thrown together as an afterthought.
"""


class DecoderFCLayerDef(LayerDef):
    
    def __init__(self, outdim, **params):
      LayerDef.__init__(self,params)
      self._outdim = outdim
      self._indim = None
      self._tied_weights = params.get('tied_weights', True)
      self.output_shape = params.get('output_shape',None)
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def instance(self,g):
        return FCLayer(g)

    def __str__(self):
      return "Fully Connected layer with output size {}".format(self._outdim)

class DecoderFCLayer(FeedThroughLayer): 
        
    def build_fwd(self):
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self._bottom  and self.d and self._bottom != self: 
          self._indim = self._bottom.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.W = tf.Variable(
                               tf.random_uniform([inflat, self.d.outdim()],
                                                 minval=-4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 maxval=4.0*math.sqrt(6.0/(inflat+ self.d.outdim())),
                                                 dtype=tf.float32
                                                 ),
                               name='Weights_'+str(self._uid))
          self.bias = tf.Variable(
                                  tf.zeros([self.d.outdim()]),
                                  name='bias_'+str(self._uid))
          flat_in = tf.reshape(self._bottom.top(),[self._indim[0],-1])
          self._top = tf.reshape(self.d.activation_function(tf.add(tf.matmul(flat_in,self.W),self.bias)), self.d.output_shape)


    def set_inject_embedding(self,i):
      flat_in = tf.reshape(i,[self._indim[0],-1])
      self._inject_recon = tf.reshape(self.d.activation_function(tf.add(tf.matmul(flat_in,self.W),self.bias)), self.d.output_shape)
        
               
    def params(self):
      return [self.W, self.bias]







class AutoEncoder(object):
    
    def __init__(self,s,g, dp, log_path, checkpoint_path, colnum=-1):
        self.dp = dp
        self.s = s
        self.g = g
        self.layeruid = 0
        self.encode_layers = [DataLayer(self.dp,g)]
        self.bottom_feed = self.encode_layers[0].bottom_feed()
        self.LEARNING_RATE=0.9
        self.MOMENTUM = 0.9
        self.ALPHA = 0.0 # mnist: 0.3
        self.freeze = False
        self.representation_loss = False
        self.EPSILON  = 0.0000001
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.summaryid = 0
        self.summarize = False
        self.coluid = colnum
        

    def kl(self,p, p_hat):
      a = p*tf.log(tf.div(p,p_hat+self.EPSILON)+self.EPSILON)
      b = (1-p)*tf.log(tf.div((1-p),(1-p_hat+self.EPSILON))+self.EPSILON)
      return tf.reduce_mean(a+b)
    
    def cross_entropy(self,p, p_hat):
      a = tf.mul(p,tf.log(self.EPSILON+p_hat))
      b = tf.mul((1-p),tf.log(1-p_hat+self.EPSILON))
      return a+b
    
    def save(self):
      with self.g.as_default():
        parameterlayers = self.encode_layers
        if self.freeze:
          parameterlayers = [self.encode_layers[-1]]
        for i,layer in enumerate(parameterlayers):
          params = [w for w in layer.params()]
          if len(params) != 0:
            saver = tf.train.Saver(params)
            prefix = ''
            if self.coluid >=0:
              prefix = 'col'+str(self.coluid)+"_"
            saver.save(self.s,path.join(self.checkpoint_path,prefix+'layer'+str(i)))
     
    def restore(self, params):
      with self.g.as_default():
        if len(params) == 0:
          return False
#         paramsdict = {w.name.strip(":[0..9]+"):w for w in params}
        saver = tf.train.Saver(params)
        if os.path.isfile(path.join(self.checkpoint_path,'col'+str(self.coluid)+"_"+'layer'+str(self.layeruid))):
          saver.restore(self.s,path.join(self.checkpoint_path,'col'+str(self.coluid)+"_"+'layer'+str(self.layeruid)))
          return path.join(self.checkpoint_path,'col'+str(self.coluid)+"_"+'layer'+str(self.layeruid))
        elif os.path.isfile(path.join(self.checkpoint_path,'layer'+str(self.layeruid))):
          saver.restore(self.s,path.join(self.checkpoint_path,'layer'+str(self.layeruid)))
          return path.join(self.checkpoint_path,'layer'+str(self.layeruid))
        else:
          return False
   
    def add_layer(self,definition):
      with self.g.as_default():
        self.save()
        #Get hyperparameters
        self.layeruid+=1
        
        #Insert into stack
        l = definition.instance(self.g)
        l.set_params(definition, self.layeruid)
        l.get_top(l)
        self.encode_layers[-1].get_top(l)
        self.encode_layers.append(l)
        l.build_fwd()
        
        #Throw in a makeshift decoder layer
        self.decoder = DecoderFCLayer(self.g)
        self.decoder.set_params(
                                DecoderFCLayerDef(
                                           reduce(mul,self.bottom_feed.get_shape().as_list()[1:]), output_shape = self.dp.shape(),sparsity_target=0.0, sparsity_lr=0.0 , activation_entropy_lr=0.0, tied_weights=False), 
                                self.layeruid+100)
        self.decoder.set_bottom(self.encode_layers[-1])
        self.decoder.build_fwd()
        
        #Reference new layer inputs/outputs
        self._top = self.encode_layers[-1].top()
        self.injection = tf.placeholder(tf.float32, self._top.get_shape().as_list(), "top_data_injection")
#         l.set_inject_embedding(self.injection)
#         self._recon = self.encode_layers[0].get_recon()
        self.decoder.set_inject_embedding(self.injection)
        self._recon = self.decoder.top()
        depth = 1 #min(len(self.encode_layers)-1,10)
        self._rep_recon = self.encode_layers[-depth].get_recon()
        self._rep_ground_truth = (self.encode_layers[-depth])._bottom.truth()
#         self._inject_recon = self.encode_layers[0].get_inject_recon()
        self._inject_recon = self.decoder.get_inject_recon()
        
        
        #Build loss and optimization functions
        self._per_example_reconstruction_loss = tf.reduce_mean(
                                                              -self.cross_entropy(
                                                                                  self.bottom_feed ,
                                                                                  self._recon),
                                                              reduction_indices=range(
                                                                                      1,
                                                                                      self._recon.get_shape().ndims) 
                                                              )
        if self.representation_loss:
          self._loss = tf.reduce_mean(
                                             tf.abs(
                                                                 self._rep_ground_truth -
                                                                 self._rep_recon))
        else:
          self._loss = tf.reduce_mean(
                                             tf.abs(
                                                                 self.bottom_feed-
                                                                 self._recon))
        
        #Sparsity
        if definition.sparsity_lr > 0.0:
          per_channel_mean_activation = tf.reduce_mean(
                                                       self._top,
                                                       reduction_indices=range(l.top().get_shape().ndims-1)
                                                       )
          
          sparsity_loss = tf.reduce_mean(
                                         self.kl(
                                                             definition.sparsity_target,
                                                             per_channel_mean_activation
                                                            )
                                        )
          self._loss += definition.sparsity_lr*sparsity_loss
          
          
        #Activation Entropy
        if definition.activation_entropy_lr > 0.0:
          per_channel_activation_entropy = tf.reduce_mean(entropy(self._top),reduction_indices=range(l.top().get_shape().ndims-1))
          
          entropy_loss = definition.activation_entropy_lr*tf.reduce_mean(per_channel_activation_entropy)
          self._loss += entropy_loss
          
          
        #Parameter groups
        newparameters=self.encode_layers[-1].params()
        existingparameters=[w for l in self.encode_layers[0:-1] for w in l.params()]
        if self.freeze:
          trainableparameters=newparameters
        else:
          trainableparameters=newparameters+existingparameters
        implicitparameters=[]
        #Restore from checkpoint or Initialize new Variables
        restore_file =  self.restore(newparameters)
        if restore_file:
          uninitializedparameters=[]
          print("Layer {} restored from checkpoint {}".format(str(self.layeruid), restore_file))
        else:
          print("No Checkpoint found for layer {}".format(str(self.layeruid)))
          uninitializedparameters=newparameters
        
        #Parameter groups hack for decoder
        newparameters += self.decoder.params()
        uninitializedparameters += self.decoder.params()
        trainableparameters += self.decoder.params()
        self._loss += 0.0000001*reduce(add, [tf.reduce_sum(tf.abs(p)) for p in self.decoder.params()])
        
        #Weight Decay
        if self.ALPHA >0 and len(trainableparameters) > 0:
          weightsmagnitude = [tf.reduce_sum(tf.abs(w)) for w in trainableparameters]
          paramsize = reduce(add,[reduce(mul,w.get_shape().as_list()) for w in trainableparameters ])
          weight_decay = reduce(add,weightsmagnitude)/paramsize
          self._loss += self.ALPHA*weight_decay
        # Optimization
        if len(trainableparameters) > 0:
          self.optimizer = tf.train.MomentumOptimizer(self.LEARNING_RATE,self.MOMENTUM,use_locking=True)
          self.optimizer_objective = self.optimizer.minimize(self._loss, var_list=trainableparameters)

          optimizer_slots = [x  for x in [self.optimizer.get_slot(v,n) for v in trainableparameters for n in self.optimizer.get_slot_names()] if x != None]
          implicitparameters += optimizer_slots
          uninitializedparameters += optimizer_slots
        if len(uninitializedparameters) > 0:
          tf.initialize_variables(uninitializedparameters).run(session=self.s)
        else:
          print("WARNING: No optimizer parameters for layer {}.".format(str(self.layeruid)))
  #       tf.train.SummaryWriter.add_graph(self.s.graph_def, self.layeruid)
        if self.summarize:
          tf.histogram_summary("representation_at_top"+str(self.layeruid), self._rep_ground_truth)
          tf.scalar_summary("reconstruction_loss"+str(self.layeruid),self._loss)
          if definition.sparsity_lr > 0.0:
            tf.histogram_summary("per_channel_mean_activation"+str(self.layeruid) , per_channel_mean_activation)
            tf.scalar_summary("sparsity_loss"+str(self.layeruid), definition.sparsity_lr*sparsity_loss)
          if definition.activation_entropy_lr > 0.0:
            tf.histogram_summary("per_channel_activation_entropy_loss"+str(self.layeruid), per_channel_activation_entropy)
            tf.scalar_summary("activation_entropy_loss"+str(self.layeruid), entropy_loss)
          if len(trainableparameters) > 0:
            self.writer = tf.train.SummaryWriter(self.log_path,self.s.graph_def)
            summarylist = [tf.histogram_summary(str(self.layeruid)+"_"+p.name,p) for i,p in enumerate(trainableparameters)]            
          self. summaries = tf.merge_all_summaries()
  
    def top_shape(self):
      return self.injection.get_shape().as_list()
    
    def fwd(self,data):
      return self.s.run(self._top, feed_dict={self.bottom_feed:data})
  
    def set_inject_embedding(self,data):
      return self.s.run(self._inject_recon, feed_dict={self.injection:data})
    
    def get_recon(self,data):
          feed_dict = {self.bottom_feed:data}
          _recon = self.s.run(self._recon,feed_dict=feed_dict)
          return (data,_recon)    
       
    def loss(self,data):
        feed_dict = {self.bottom_feed:data}
        l = self.s.run(self._loss,feed_dict=feed_dict)
        return l
      
    def per_example_reconstruction_loss(self,data):
        feed_dict = {self.bottom_feed:data}
        l = self.s.run(self._per_example_reconstruction_loss,feed_dict=feed_dict)
        return l    
        
    def train_mb(self,data): 
          feed_dict = {self.bottom_feed:data}
          if self.summarize:
            summary_str,_,dummy, l = self.s.run([self.summaries, self._recon,self.optimizer_objective,self._loss],feed_dict=feed_dict)
            self.writer.add_summary(summary_str,self.summaryid)
            self.summaryid +=1
          else:
            _,dummy, l = self.s.run([self._recon,self.optimizer_objective,self._loss],feed_dict=feed_dict)
          return l
          

    

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
          d,r = ae.get_recon(mb[0])
          plt.imshow(dp.denormalize(np.append(d[0],r[0],axis=0)[:,:,::-1]), cmap="Greys")
          plt.colorbar()
          plt.show()
        
  
