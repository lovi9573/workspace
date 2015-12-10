"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.8%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to exectute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys

import tensorflow.python.platform
import rbmconfig_pb2 as proto
from google.protobuf.text_format import Merge
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from weights_to_img import display
import math
import matplotlib.pyplot as plt

config = proto.RbmParameters()

with open(sys.argv[1],'r') as fin:
  Merge(fin.read(),config)

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = config.train_data_filename
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = config.batch_size
NUM_EPOCHS = config.epochs
NUM_HIDDEN = config.num_hidden
N_VISIBLE = IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS
SAMPLE_HIDDEN = config.sample_hiddens
SAMPLE_VISIBLE = config.sample_visibles
N_GIBBS_STEPS = config.gibbs_sampling_steps
REPORT=config.synchronization_period
LR = config.learning_rate
MOMENTUM = config.momentum
PERSISTENT_CHAIN = config.persistent_gibbs_chain
SPARSITY = config.use_sparsity_target
SPARSITY_TARGET = config.sparsity_target
SPARSITY_LR = config.sparsity_learning_rate
SPARSITY_DECAY = config.sparsity_decay
LAMBDA = config.weight_decay_coef



#tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
#FLAGS = tf.app.flags.FLAGS


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(WORK_DIRECTORY):
    os.mkdir(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [0, 1.0].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = data / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE*IMAGE_SIZE)
    print(data.shape[0])
    return data



def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_data = extract_data(train_data_filename, 60000)
  num_epochs = NUM_EPOCHS
  train_size = train_data.shape[0]
  print("Training data samples: {}".format(train_size))

  #Setup persistent vars
  visible = tf.placeholder(tf.float32,shape=(BATCH_SIZE,N_VISIBLE))
  weights = tf.Variable(
      tf.random_uniform([NUM_HIDDEN, N_VISIBLE],
                          minval=-4.0*math.sqrt(6.0/(NUM_HIDDEN + N_VISIBLE)),
                          maxval=4.0*math.sqrt(6.0/(NUM_HIDDEN + N_VISIBLE)),
			                    dtype=tf.float32,
                          seed=SEED))
  bias_h = tf.Variable(tf.zeros([NUM_HIDDEN],dtype=tf.float32))
  bias_v = tf.Variable(tf.zeros([N_VISIBLE],dtype=tf.float32))

  def corrupt_input(input, thresh):
    n = tf.random_uniform(tf.shape(input))
    return tf.to_float(n > thresh)*input

  def v_h(v):
    h = tf.sigmoid(tf.matmul(v,weights,transpose_b=True) +bias_h)
    return h
  
  def h_r(h):
    r = tf.sigmoid(tf.matmul(h,weights) + bias_v )
    return r

  def kl(p, p_hat):
    a = p*tf.log(tf.div(p,p_hat))
    b = (1-p)*tf.log(tf.div((1-p),(1-p_hat)))
    return tf.reduce_mean(a+b)
  
  def cross_entropy(v,r):
    e = 0.0000001
    return -tf.reduce_mean(tf.reduce_sum((v*tf.log(r+e) + (1-v)*tf.log(1-r-e)), reduction_indices=1))
  
  def weight_decay(W,b_v,b_h):
    return tf.reduce_sum(tf.pow(W,2)) + tf.reduce_sum(tf.pow(b_h,2))+tf.reduce_sum(tf.pow(b_v,2))


  def model(d, corr, train=False):
    """The Model definition."""
    data =corrupt_input(d,corr)
    h = v_h(data) 
    r = h_r(h) 
    L = cross_entropy(data,r)
    return L,data, h, r
    
  # Build computation graph
  loss, v, h, r = model(visible, True)
            
  # Learning rate scheduling
  batch = tf.Variable(0)

 
	
  # Optimization
  optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss,
                                                       var_list=[weights,bias_h, bias_v],
                                                       global_step=batch)


  # Create a local session to run this computation.
  with tf.Session() as s:
    tf.initialize_all_variables().run()
    print('Initialized!')
    costs = {"err":[],
             "sparsity": [],
             "weight": []}
    display_step = 1
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :]
      feed_dict = {visible: batch_data}
      _, l  = s.run(
          [optimizer, loss],
          feed_dict=feed_dict)

      if step % REPORT == 0:
        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
        print('Minibatch loss: %.3f' % (l))
      if step == display_step or (step+1)%(1*train_size // BATCH_SIZE) == 0:
        #display(s.run(visiblevar).reshape([BATCH_SIZE,28,28,1]))
        ht = s.run(h,feed_dict=feed_dict).reshape([1,BATCH_SIZE,NUM_HIDDEN,1])
        display(ht)
        if SPARSITY:
          bh = s.run(tf.reduce_mean(h,0),feed_dict=feed_dict).reshape([NUM_HIDDEN,1,1,1])
          display(bh)
        vt = s.run(v,feed_dict=feed_dict).reshape([BATCH_SIZE,28,28,1])
        rt = s.run(r,feed_dict=feed_dict).reshape([BATCH_SIZE,28,28,1])
        display(numpy.append(vt,rt,0))
        w = s.run(weights,feed_dict=feed_dict).reshape([NUM_HIDDEN,28,28,1])
        b_v = s.run(bias_v,feed_dict=feed_dict).reshape([1,28,28,1])
        #display(numpy.append(w, b_v, 0))
        display(w)
        plt.plot(costs["err"],'r', costs["sparsity"],'b', costs["weight"],'k')
        x0,x1,y0,y1 = plt.axis()
        plt.axis((x0,x1,y0,0.5*y1))
        plt.show()
        sys.stdout.flush()
        #display_step += int(REPORT*math.log(4*step+5))
        

if __name__ == '__main__':
  #tf.app.run()
  main()
