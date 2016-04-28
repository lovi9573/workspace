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



EPSILON  = 0.0000001

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
      
    def instance(self,g,uid, freeze):
        return FCLayer(self,g,uid, freeze)

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
      self._recon_shape = params.get('recon_shape',None)
        
    def outdim(self):
        return self._outdim

    def indim(self):
        return self._indim
      
    def filterdim(self):
        return [self._filterdim,self._filterdim]
      
    def strides(self):
        return [1,self._stride,self._stride,1]
      
    def instance(self, g, uid, freeze):
        return ConvLayer(self,g,uid, freeze)
      
    def recon_shape(self):
      return self._recon_shape
  
    def __str__(self):
      return "Convolutional layer {}x{}x{} stride {}".format(self._filterdim, self._filterdim,self._outdim,self._stride)
  
class CorruptionLayerDef(LayerDef):
    def __init__(self,corruptionlevel, **params):
      LayerDef.__init__(self,params)
      self._corruptionlevel = corruptionlevel
  
    def corruptionlevel(self):
      return self._corruptionlevel
      
    def instance(self, g, uid, freeze):
        return CorruptionLayer(self,g,uid, freeze)
  
    def __str__(self):
      return "Corruption layer at p({})".format(self._corruptionlevel)
    
class FeedThroughLayerDef(LayerDef):
    def __init__(self, **params):
      LayerDef.__init__(self,params)
      
    def instance(self,g,uid, freeze):
        return FeedThroughLayer(self)
  
    def __str__(self):
      return "Feed Through layer"
  
"""
==========================================================================================================
  Layer Implementations
==========================================================================================================
Pathways:
=== top === embedding == inject_embedding ==
|    ^         |               |             |
|    |         \/              \/            |
== bottom == get_recon ====== inject_recon =====
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


  def get_top(self):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      return self.datalayer
  
  def bottom_feed(self):
    return self.datalayer 
  
  def params(self):
    return []
 

 
class FeedThroughLayer(Layer):
    
  def __init__(self,d,g,uid, freeze):
      self.g = g
      self._bottom = None
      self._top = None
      self._embedding = None
      self._recon = None
      self._inject_embedding = None
      self._inject_recon = None
      self.d = d
      self._uid = uid
      self._freeze = freeze
      self._params = []
      
  def set_bottom(self,l):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      self._bottom = l

  def get_top(self):
      '''
      :param l: The layer that feeds into this one
      :type l: Layer
      '''     
      return self._top    
      
  def set_embedding(self,e):
    self._embedding = e

  def get_recon(self):
      return self._recon  
      
  def set_inject_embedding(self,i):
      self._inject_embedding = i
  
  def get_inject_recon(self):
    return self._inject_recon
      
  def build_fwd(self):
      if self._bottom: 
        self._top = self._bottom
        
  def build_back(self):
      if self._embedding and self._inject_embedding:
        self._recon = self._embedding
        self._inject_recon = self._inject_embedding
   
  def params(self):
    return self._params  
  
  def uid(self):
    return self._uid 
  
  def freeze(self):
    return self._freeze
  
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

    

class FCLayer(FeedThroughLayer): 
        
    def init(self,din,dout):
    
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self.d: 
          self._indim = self._bottom.top().get_shape().as_list()
          inflat = reduce(mul,self._indim[1:])
          self.W = tf.Variable(
                               tf.random_uniform([din, dout],
                                                 minval=-4.0*math.sqrt(6.0/(din+ dout)),
                                                 maxval=4.0*math.sqrt(6.0/(din+ dout)),
                                                 dtype=tf.float32
                                                 ),
                               name='Weights_'+str(self._uid))

    def build_fwd(self):
      d_in = self._bottom.get_shape().as_list()
      d_flat = reduce(mul,d_in)
      self.init(d_flat,self.d.outdim())
      self.bias = tf.Variable(
                              tf.zeros([self.d.outdim()]),
                              name='bias_'+str(self._uid)
                              )
      flat_in = tf.reshape(self._bottom,[d_in[0],-1])
      self._top = self.d.activation_function(tf.add(tf.matmul(flat_in,self.W),self.bias))
      self._params = [self.W, self.bias]
       
    def build_back(self):
      d_in = self._embedding.get_shape().as_list()
      d_flat = reduce(mul,d_in)
      self.init(self.d.outdim(),d_flat)      
      self.rev_bias = tf.Variable(
                              tf.zeros([self.d.outdim()]),
                              name='rev_bias_'+str(self._uid)
                              )
      self._inject_recon = self._compute_back(self._inject_embedding)
      self._recon = self._compute_back(self._embedding)
      self._params = [self.W, self.rev_bias]
    
    def _compute_back(self,top):
      with self.g.as_default():
        return self.d.activation_function(tf.add(tf.reshape(tf.matmul(top,tf.transpose(self.W)),self._indim),self.rev_bias))

    
    


class ConvLayer(FeedThroughLayer):

    def init(self,cbottom, ctop, fwd):
        
      '''
      Constructs the computation graph for this layer and all subsequent encode_layers.
      '''
      with self.g.as_default():
        if self.d:
          if fwd:
            var_prefix = "encode_"
          else:
            var_prefix = 'decode_' 
          dims = self.d.filterdim()+[cbottom, ctop]
          self.W = tf.Variable(
                               tf.truncated_normal(dims,
                                                 stddev=math.sqrt(3.0/(reduce(mul,self.d.filterdim()))),
                                                 dtype=tf.float32
                                                 ),
                               name=var_prefix+'W_'+str(self._uid))
          if fwd:
            self.bias = tf.Variable(
                                   tf.zeros([ctop],
                                            name=var_prefix+'bias_'+str(self._uid))
                                    )
          else:
            self.bias = tf.Variable(
                                   tf.zeros([cbottom],
                                            name=var_prefix+'rev_bias_'+str(self._uid))
                                    )
          self._params = [self.W, self.bias]
          
    def build_fwd(self):
      cin = self._bottom.get_shape().as_list()[-1]
      self.init(cin, self.d.outdim(),True)
      self._top = self.d.activation_function(tf.nn.conv2d(self._bottom, self.W, self.d.strides(),self.d._padding, name="Conv_"+str(self._uid)+"_top")+self.bias)
 
        
    def build_back(self):
      cin = self._embedding.get_shape().as_list()[-1]
      self.init(self.d.outdim(), cin,False)
      self._inject_recon = self._compute_back(self._inject_embedding)
      self._recon = self._compute_back(self._embedding)
        
    def _compute_back(self, top):
      with self.g.as_default():
        return self.d.activation_function(tf.nn.deconv2d(top, self.W, self.d.recon_shape(), self.d.strides(), padding=self.d._padding)+self.bias)
          


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
=============  Helper Functions ============
"""
def kl(p, p_hat):
  a = p*tf.log(tf.div(p,p_hat+EPSILON)+EPSILON)
  b = (1-p)*tf.log(tf.div((1-p),(1-p_hat+EPSILON))+EPSILON)
  return tf.reduce_mean(a+b)

def cross_entropy(p, p_hat):
  a = tf.mul(p,tf.log(EPSILON+p_hat))
  b = tf.mul((1-p),tf.log(1-p_hat+EPSILON))
  return a+b



"""
=============  Autoencoder Container ============
"""



class AutoEncoder(object):
    
    def __init__(self,s,g, dp, log_path, checkpoint_path, colnum=-1):
        self.dp = dp
        self.s = s
        self.g = g
        self.log_path = log_path
        self.checkpoint_path = checkpoint_path
        self.coluid = colnum
        self.layeruid = 0
        self.encode_layers = [DataLayer(self.dp,g)]
        self.decode_layers = []
        self.bottom_feed = self.encode_layers[0].bottom_feed()
        self.LEARNING_RATE=0.9
        self.MOMENTUM = 0.9
        self.ALPHA = 0.1 # mnist: 0.3
        self.freeze = False
        self.summaryid = 0
        self.summarize = False
        self.isDecoderValid = False
        

    
    def save(self):
      with self.g.as_default():
        if self.freeze:
          parameterlayers = [self.encode_layers[-1]]
        else:
          parameterlayers = self.encode_layers
        for layer in parameterlayers:
          params = [w for w in layer.params()]
          #tack the decode variable onto the last encode layer
          if layer == self.encode_layers[-1]:
            params += [w for dl in self.decode_layers for w in dl.params()]
          if len(params) != 0:
            saver = tf.train.Saver(params)
            prefix = ''
            if self.coluid >=0:
              prefix = 'col'+str(self.coluid)+"_"
            saver.save(self.s,path.join(self.checkpoint_path,prefix+'layer'+str(layer.uid())))
     
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

    def add_layer(self,definition, freeze=True):
      with self.g.as_default():
        self.save()
        #Get hyperparameters
        self.layeruid+=1
        
        #Insert into stack
        l = definition.instance(self.g, self.layeruid, freeze)
        l.set_bottom(self.encode_layers[-1].get_top())
        self.encode_layers.append(l)
        l.build_fwd()
        self._top = self.encode_layers[-1].get_top()
        self.isDecoderValid = False

        
        
        
    def set_decode(self, decode_layerdefs):
      with self.g.as_default():
        #establish first decoder layer
        l = decode_layerdefs[0].instance(self.g,str(self.layeruid)+"_"+str(1), False)
        #tie top of encoder and top of injection path to decoder layer 1
        self.decode_layers.append(l)
        self.injection = tf.placeholder(tf.float32, self._top.get_shape().as_list(), "top_data_injection")
        l.set_embedding(self._top)
        l.set_inject_embedding(self.injection)
        #Create graph
        l.build_back()
        #attach remaining decode_layers
        for i,ldef in enumerate(decode_layerdefs[1:]):
          l = ldef.instance(self.g,str(self.layeruid)+"_"+str(1+i))
          l.set_embedding(self.decode_layers[-1].get_recon())
          l.set_inject_embedding(self.decode_layers[-1].get_inject_recon())
          self.decode_layers.append(l)
          #Create graph
          l.build_back()
        self._recon = self.decode_layers[-1].get_recon()
        self._inject_recon = self.decode_layers[-1].get_inject_recon()
        self.isDecoderValid = True
   
        
    def build(self):
      with self.g.as_default():
        #Build loss and optimization functions
        self._per_example_reconstruction_loss = tf.reduce_mean(
                                                              -cross_entropy(
                                                                                  self.bottom_feed ,
                                                                                  self._recon),
                                                              reduction_indices=range(
                                                                                      1,
                                                                                      self._recon.get_shape().ndims) 
                                                              )
        self._loss = tf.reduce_mean(
                                             tf.abs(
                                                                 self.bottom_feed-
                                                                 self._recon))
        
        #Sparsity
#         if definition.sparsity_lr > 0.0:
#           per_channel_mean_activation = tf.reduce_mean(
#                                                        self._top,
#                                                        reduction_indices=range(l.top().get_shape().ndims-1)
#                                                        )
#           
#           sparsity_loss = tf.reduce_mean(
#                                          self.kl(
#                                                              definition.sparsity_target,
#                                                              per_channel_mean_activation
#                                                             )
#                                         )
#           self._loss += definition.sparsity_lr*sparsity_loss
          
          
        #Activation Entropy
#         if definition.activation_entropy_lr > 0.0:
#           per_channel_activation_entropy = tf.reduce_mean(entropy(self._top),reduction_indices=range(l.top().get_shape().ndims-1))
#           
#           entropy_loss = definition.activation_entropy_lr*tf.reduce_mean(per_channel_activation_entropy)
#           self._loss += entropy_loss
          
          
        #Parameter groups
        newparameters=self.encode_layers[-1].params()+[ p for dl in self.decode_layers for p in dl.params() ]
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
          print("Layer {} and decoder restored from checkpoint {}".format(str(self.layeruid), restore_file))
        else:
          print("No Checkpoint found for layer {}".format(str(self.layeruid)))
          uninitializedparameters=newparameters
        
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
#           if definition.sparsity_lr > 0.0:
#             tf.histogram_summary("per_channel_mean_activation"+str(self.layeruid) , per_channel_mean_activation)
#             tf.scalar_summary("sparsity_loss"+str(self.layeruid), definition.sparsity_lr*sparsity_loss)
#           if definition.activation_entropy_lr > 0.0:
#             tf.histogram_summary("per_channel_activation_entropy_loss"+str(self.layeruid), per_channel_activation_entropy)
#             tf.scalar_summary("activation_entropy_loss"+str(self.layeruid), entropy_loss)
          if len(trainableparameters) > 0:
            self.writer = tf.train.SummaryWriter(self.log_path,self.s.graph_def)
            summarylist = [tf.histogram_summary(str(self.layeruid)+"_"+p.name,p) for i,p in enumerate(trainableparameters)]            
          self. summaries = tf.merge_all_summaries()
  
    def top_shape(self):
      return self.injection.get_shape().as_list()
    
    def bottom_shape(self):
      return self.bottom_feed.get_shape().as_list()
    
    def fwd(self,data):
      return self.s.run(self._top, feed_dict={self.bottom_feed:data})
  
    def inject(self,data):
      return self.s.run(self._inject_recon, feed_dict={self.injection:data})
    
    def fwd_back(self,data):
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
        
  
