# Interpretable cnn for big five persoanlity traits using audio data #
# This script implemented cam cnn structure using summary spectrogram #



from __future__ import print_function

from random import shuffle
import numpy as np
import tensorflow as tf
import cnn_params as params
import random
import h5py
from scipy import stats

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_boolean(
    'train_cnn', False,
    'If True, allow model parameters to change during training'
    'If False, dont allow model parameters to change during training')

FLAGS = flags.FLAGS

def main(_):
   
  with tf.Graph().as_default():
       
       with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=params.INIT_STDDEV),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=FLAGS.train_cnn), \
         slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
         slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
         tf.variable_scope('cnn'):
           # Input: MFCC features consist of 2-D summary-spectrogram.
           features = tf.placeholder(tf.float32, shape=(params.NUM_FRAMES, params.NUM_BANDS),name='mfcc_features')
           # Reshape to 4-D tensor for the convolution.
           net = tf.reshape(features, [-1,params.NUM_FRAMES, params.NUM_BANDS, 1])
           # Network definition of alternating convolutions and max-pooling operations.
           net = slim.conv2d(net, 64, scope='conv1')
           net = slim.max_pool2d(net, scope='pool1')
           net = slim.conv2d(net, 128, scope='conv2')
           net = slim.max_pool2d(net, scope='pool2')
           net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
           net = slim.max_pool2d(net, scope='pool3')
           net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
           # Take output of the last convolution layer.
           conv_output = tf.identity(net, name='conv_output')
           # Global Average Pooling (GAP) layer.
           net = slim.avg_pool2d(net, [params.kernel_size_x,params.kernel_size_y],scope='GAP')
           # Flatten before entering fully-connected layers
           net = slim.flatten(net)
           # Fully-connected.
           logits = slim.fully_connected(net, params.NUM_CLASSES, activation_fn=None, trainable=True, scope='logits')
           
           with tf.variable_scope('model'):
            with tf.variable_scope('train'):
             global_step = tf.Variable(0, name='global_step', trainable=FLAGS.train_cnn,
                           collections = [tf.GraphKeys.GLOBAL_VARIABLES,
                           tf.GraphKeys.GLOBAL_STEP])
             labels = tf.placeholder(tf.float32, shape=(params.NUM_CLASSES), name='labels')
             xent = tf.squared_difference(logits, labels)
             loss = tf.reduce_mean(xent, name='loss_op')
             # Adam optimizer for training.
             optimizer = tf.train.AdamOptimizer(
                learning_rate=params.LEARNING_RATE,
                epsilon=params.ADAM_EPSILON)
             optimizer.minimize(loss, global_step=global_step,name='train_op')
       saver = tf.train.Saver()
       
       with tf.Session() as sess:
         # Locate tensors.
         features_tensor = sess.graph.get_tensor_by_name('cnn/mfcc_features')
         labels_tensor = sess.graph.get_tensor_by_name('cnn/model/train/labels:0')
         conv_output = sess.graph.get_tensor_by_name('cnn/conv_output:0')
         
         # Load trained model weights.
         weight_file = '.../path/to/load/model/weights/CAM_Summary_weights.npz'
         skip_layer = []
         sess.run(tf.global_variables_initializer())
         weights = np.load(weight_file)
         keys = sorted(weights.keys())
         parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
         param_names = {}
         for i, p in enumerate(parameters):
               param_names[p.name] = i
         for i, k in enumerate(keys):
                cond = [x not in k for x in skip_layer]
                if False not in cond:
                   print (i, k, np.shape(weights[k]))
                   try:
                        idx = param_names[k]
                        print (parameters[idx].name)
                        sess.run(parameters[idx].assign(weights[k]))
                   except:
                        print("Not found")

         # Data loaders.
         with h5py.File('.../Path/to/load/test_features.h5', 'r') as hf:
                data_features = hf['test_features'][:]
         with h5py.File('.../path/to/load/test_labels.h5', 'r') as hf:
                data_labels = hf['test_labels'][:]

         # Output layer weights.          
         all_vars= tf.global_variables()
         def get_var(name):
           for i in range(len(all_vars)):
             if all_vars[i].name.startswith(name):
                return all_vars[i]
           return None
         layer_wghts = get_var('cnn/logits')
         # Get predictions.
         no_of_samples = data_features.shape[0]
         input_mfcc = []
         mfcc_conv_features = []
         rand_array=[i for i in range(no_of_samples)]
         np.random.shuffle(rand_array)
         for y in range(no_of_samples):
	x = rand_array[y]
             features = data_features[x,:,:]
             labels = data_labels[x,:]
             [output_layer_wghts,conv_features,prediction] = sess.run([layer_wghts,conv_output,logits],
                                     feed_dict={features_tensor: features, labels_tensor: labels})
             model_predictions.append(prediction)
             input_mfcc.append(features)
             mfcc_conv_features.append(conv_features)

         np.save('.../path/to/save/input_mfcc', input_mfcc)
         np.save('.../path/to/save/output_layer_wghts', layer_wghts)
         np.save('.../path/to/save/model_predictions', model_predictions)
         
         
if __name__ == '__main__':
    tf.app.run()
