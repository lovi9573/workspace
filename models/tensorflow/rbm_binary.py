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
WORK_DIRECTORY = '/home/user/storage/mnist'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = config.batch_size
NUM_EPOCHS = 10
NUM_HIDDEN = 32
V_DIM = IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS
SAMPLE_HIDDEN = True
SAMPLE_VISIBLE = True
N_GIBBS_STEPS = 10
REPORT=100


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
  bias_v = tf.Variable(tf.constant(0.1, shape=[V_DIM]))
  recon = tf.Variable(tf.zeros([BATCH_SIZE, V_DIM]))
  hidden = tf.Variable(tf.zeros([BATCH_SIZE, NUM_HIDDEN],dtype=tf.float32))
  
  def v_h(v, sample=True):
    hidden.assign( tf.matmul(v,weights,transpose_b=True)+bias_h)
    if sample:
      thresh = tf.random_uniform([BATCH_SIZE, NUM_HIDDEN])
      hidden.assign( tf.to_float(hidden > thresh))
    return hidden
  
  def h_v(h, sample=True):
    recon.assign( tf.matmul(h,weights) + bias_v)
    if sample:
      thresh = tf.random_uniform([BATCH_SIZE, V_DIM])
      recon.assign( tf.to_float(recon > thresh))
    return recon

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
    hidden.assign( v_h(data, SAMPLE_HIDDEN))
    
    # Get positive phase energy
    F_p = FreeEnergy(data)

    h = hidden
    # Gibbs chain  
    for i in range(N_GIBBS_STEPS-1):
      r = h_v(h,SAMPLE_VISIBLE)
      h = v_h(r, SAMPLE_HIDDEN)
    recon.assign( h_v(h,SAMPLE_VISIBLE))
    hidden.assign( v_h(recon, SAMPLE_HIDDEN))
  
    # Get positive phase energy
    F_n = FreeEnergy(recon)   
    return F_p - F_n

  # Build computation graph
  F = model(visible, True)
  loss = tf.reduce_mean(F)
  
  # Learning rate scheduling
  batch = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
      0.001,  # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      0.95,  # Decay rate.
      staircase=True)
 
  # Optimization
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       var_list=[weights,bias_h,bias_v],
                                                       global_step=batch)


  # Create a local session to run this computation.
  with tf.Session() as s:
    tf.initialize_all_variables().run()
    print('Initialized!')
    for step in xrange(num_epochs * train_size // BATCH_SIZE):
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), :]
      feed_dict = {visible: batch_data}
      _, l, lr = s.run(
          [optimizer, loss, learning_rate],
          feed_dict=feed_dict)
      if step % REPORT == 0:
        print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        w = s.run(weights).reshape([NUM_HIDDEN,28,28,1])
        b_v = s.run(bias_v).reshape([1,28,28,1])
        display(numpy.append(w, b_v, 0))
        sys.stdout.flush()

if __name__ == '__main__':
  #tf.app.run()
  main()
