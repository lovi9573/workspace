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
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = config.batch_size
NUM_EPOCHS = config.epochs
NUM_HIDDEN = config.num_hidden
V_DIM = IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS
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


def extract_labels(filename, num_images):
  """Extract the labels into a 1-hot matrix [image index, label index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
  # Convert to dense 1-hot representation.
  return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images, NUM_LABELS), dtype=numpy.float32)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image, label] = 1.0
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 * 
      numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) / 
      predictions.shape[0])


def main(argv=None):  # pylint: disable=unused-argument
  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')

  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 60000)

  # Generate a validation set.
  train_data = train_data[:-VALIDATION_SIZE, :]
  num_epochs = NUM_EPOCHS
  train_size = train_data.shape[0]
  print("Training data samples: {}".format(train_size))

  #Setup persistent vars
  visible = tf.placeholder(tf.float32,shape=(BATCH_SIZE,V_DIM))
  weights = tf.Variable(
      tf.truncated_normal([NUM_HIDDEN, V_DIM],
                          stddev=0.1,
                          seed=SEED))
  bias_h = tf.Variable(tf.zeros([NUM_HIDDEN]))
  bias_v = tf.Variable(tf.zeros([V_DIM]))
  persistent_recon = tf.Variable(tf.zeros([BATCH_SIZE, V_DIM]))
  sparsity = tf.Variable(tf.fill([1],SPARSITY_TARGET))

  def v_h(v, sample=True):
    p_h = tf.sigmoid(tf.matmul(v,weights,transpose_b=True) +bias_h)
    if sample:
      thresh = tf.random_uniform([BATCH_SIZE, NUM_HIDDEN])
      h = tf.to_float(p_h > thresh)
    else:
      h = p_h
    return p_h,h
  
  def h_v(h, sample=True):
    p_r = tf.sigmoid(tf.matmul(h,weights) + bias_v)
    if sample:
      thresh = tf.random_uniform([BATCH_SIZE, V_DIM])
      r = tf.to_float(p_r > thresh)
    else:
      r = p_r
    return p_r,r

  def Energy(v,h):
    hwvr = tf.reduce_sum(h*tf.matmul(v,weights, transpose_b=True),reduction_indices=1)
    bh = tf.matmul(h,tf.expand_dims(bias_h,-1))
    bv = tf.matmul(v,tf.expand_dims(bias_v,-1))
    return -(hwvr + bh + bv)
 
  def FreeEnergy(v):
    e = tf.reduce_sum(tf.log(1 + tf.exp( tf.matmul(v,weights,transpose_b=True)+bias_h)),reduction_indices=1)
    bv = tf.matmul(v,tf.expand_dims(bias_v,-1)) 
    return -(bv + e)

  def model(data, train=False):
    """The Model definition."""
    # Positive Phase
    p_h_p,h_p = v_h(data, SAMPLE_HIDDEN)
     
    # Get positive v energy
    F_p = FreeEnergy(data)
    if PERSISTENT_CHAIN:
      _,h_t = v_h(persistent_recon, SAMPLE_HIDDEN)
    else:
      h_t = h_p
    # Gibbs chain  
    for i in range(N_GIBBS_STEPS-1):
      _,r = h_v(h_t,SAMPLE_VISIBLE)
      _,h_t = v_h(r, SAMPLE_HIDDEN)
    p_r,r = h_v(h_t,SAMPLE_VISIBLE)
    if PERSISTENT_CHAIN:
      r = persistent_recon.assign(r) 
    p_h_n,h_n = v_h(r, SAMPLE_HIDDEN)
   
    # Get positive phase energy
    F_n = FreeEnergy(r)   
    return F_p - F_n, p_h_p, p_h_n, r

  # Build computation graph
  F,h_p, h_n, recon = model(visible, True)
  if SPARSITY:
    print("Using sparsity target: {}".format(SPARSITY_TARGET))
    sparsity = (SPARSITY_DECAY*tf.reduce_mean(tf.abs(tf.sub(tf.reduce_mean(h_p,0), SPARSITY_TARGET)))) +\
               (1-SPARSITY_DECAY)*sparsity
    loss = tf.reduce_mean(F) + SPARSITY_LR*sparsity 
  else:
    loss = tf.reduce_mean(F)
  
  # Learning rate scheduling
  batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
      LR/BATCH_SIZE,  # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      0.95,  # Decay rate.
      staircase=True)
 
  # Optimization
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         MOMENTUM).minimize(loss,
                                                       var_list=[weights, bias_h,bias_v],
                                                       global_step=batch)

  display_progress = False
  # Create a local session to run this computation.
  with tf.Session() as s:
    tf.initialize_all_variables().run()
    print('Initialized!')
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :]
      feed_dict = {visible: batch_data}
      _, l, lr,_ = s.run(
          [optimizer, loss, learning_rate, recon],
          feed_dict=feed_dict)
      if numpy.max(numpy.abs(s.run(weights))) > 1000:
        display_progress = True
      if step % REPORT == 0:
        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
      if display_progress == True or (step+1)%(50*REPORT) == 0:
        #display(s.run(visiblevar).reshape([BATCH_SIZE,28,28,1]))
        h = s.run(tf.reduce_mean(h_p,0),feed_dict=feed_dict).reshape([NUM_HIDDEN,1,1,1])
        display(h)
        bh = s.run(bias_h).reshape([NUM_HIDDEN,1,1,1])
        display(bh)
        v = batch_data.reshape([BATCH_SIZE,28,28,1])
        r = s.run(recon,feed_dict=feed_dict).reshape([BATCH_SIZE,28,28,1])
        display(numpy.append(v,r,0))
        w = s.run(weights,feed_dict=feed_dict).reshape([NUM_HIDDEN,28,28,1])
        b_v = s.run(bias_v,feed_dict=feed_dict).reshape([1,28,28,1])
        display(numpy.append(w, b_v, 0))
        sys.stdout.flush()

if __name__ == '__main__':
  #tf.app.run()
  main()
