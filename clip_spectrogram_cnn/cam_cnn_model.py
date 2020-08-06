# Interpretable cnn for big five persoanlity traits using audio data #
# This script implemented cam cnn structure using clip spectrogram #


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
           # Input: MFCC features of a single video having '13' 2-D log-mel-spectrogram patches.
           features = tf.placeholder(tf.float32, shape=(None,params.NUM_FRAMES, params.NUM_BANDS),name='mfcc_features')
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
           last_conv = tf.identity(net, name='last_conv')
           # Global Average Pooling (GAP) layer.
           net = slim.avg_pool2d(net, [params.kernel_size_x,params.kernel_size_y],scope='GAP')
           # Flatten before entering fully-connected layers.
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
         # Initialize all variables in the model, and then load the fine-tunned cnn weights.
         sess.run(tf.global_variables_initializer())
         weight_file = '.../fine_tunned/cnn/weights/file/in/.npz'
         # skip fully connected layer
         skip_layer = ['cnn/fc1']  
         sess.run(tf.global_variables_initializer())
         weights = np.load(weight_file)
         keys = sorted(weights.keys())
         parameters = tf.get_collection(tf.GraphKeys.VARIABLES)
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

         all_vars= tf.global_variables()
         def get_var(name):
           for i in range(len(all_vars)):
             if all_vars[i].name.startswith(name):
                return all_vars[i]
           return None

         # Locate all the tensors and ops we need for the training loop.
         features_tensor = sess.graph.get_tensor_by_name('cnn/mfcc_features')
         labels_tensor = sess.graph.get_tensor_by_name('cnn/model/train/labels:0')
         global_step_tensor = sess.graph.get_tensor_by_name('cnn/model/train/global_step:0')
         loss_tensor = sess.graph.get_tensor_by_name('cnn/model/train/loss_op:0')
         train_op = sess.graph.get_operation_by_name('cnn/model/train/train_op')
         last_conv = sess.graph.get_tensor_by_name('cnn/last_conv:0')
         
         # Data loaders.   
         with h5py.File('.../path/to/load/train_features.h5', 'r') as hf:
              train_features = hf['train_features'][:]
         with h5py.File('.../path/to/load/train_labels.h5', 'r') as hf:
              train_labels = hf['train_labels'][:]
         with h5py.File('.../path/to/load/test_features.h5', 'r') as hf:
              test_features = hf['test_features'][:]
         with h5py.File('.../path/to/load/test_labels.h5', 'r') as hf:
              test_labels = hf['test_labels'][:]
 
         m_train = train_features.shape[0]
         m_test = test_features.shape[0]

         graph_train = []
         graph_test = []
         train_loss = []
         train_loss_mean = []
         test_loss = []
         test_loss_meam = []
         accuracy = []
         
         for epoch in range(params.No_of_Epochs):
           print("Epoch:", '%04d' % (epoch+1))
           rand_array=[i for i in range(m_train)]
           np.random.shuffle(rand_array)
           # Array initialization. 
           MAE_clip_Acc = []
           
           # Training. 
           for x in range(m_training):
             y = rand_array[x]
             Tfeatures = train_features[y,:,:,:]
             # Data standardization. 
             Tfeatures = stats.zscore(Tfeatures)
             Tlabels = data_labels[y,:]
             [loss_batch,_,num_steps] = sess.run([loss_tensor,train_op,global_step_tensor],
                     feed_dict={features_tensor:Tfeatures, labels_tensor:Tlabels})
             train_loss.append(loss_batch)

           training_loss_mean = np.mean(train_loss)
           graph_train.append(training_loss_mean)

           if epoch == No_of_Epochs-1:
           # Save model in npz.
            parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            dictionary = {}
            dir_path = '.../path/to/save/model/weights/'
            for j, param in enumerate(parameters):
                param_values = param.eval(session=sess)
                new_key = param.name
                dictionary[new_key] = param_values
            npz_name = dir_path + 'weights.npz'
            np.savez(npz_name, **dictionary)

           # Testing.
           rand_array_1=[i for i in range(m_test)]
           np.random.shuffle(rand_array_1)
           Predicted_video_mean=np.zeros([m_test,1])
           for y in range(m_test):
             x = rand_array_1[y]
             features = test_features[x,:,:,:]
             # Data standardization
             features = stats.zscore(features)
             labels = data_labels_val[x,:]
             [loss_batch_1,prediction,last_conv_output] = sess.run([loss_tensor,logits,last_conv],
                                   feed_dict={features_tensor: features, labels_tensor: labels})
             test_loss.append(loss_batch_1)
             
             # Clip level accuracy. 
             labels = labels.astype('float64')
             difference = labels - prediction
             abs_difference = np.absolute(difference)
             one_diff = 1 - abs_difference
             sum_1 = np.sum(one_diff)
             mean_accuracy = mean_sum_1/params.NUM_CLASSES
             Predicted_video_mean[y] = mean_accuracy
           MAE_Clip_Mean = np.average(Predicted_video_mean)
           # Accuracy of the model.
           accuracy.append(MAE_Clip_Mean)
           # Mean testing loss.
           test_loss_mean = np.mean(test_loss)
           graph_test.append(test_loss_mean)

           # Save mean_train_loss, mean_test_loss, and accuracy.                     
           np.save('.../path/to/save/train_loss', graph_train)
           np.save('.../path/to/save/test_loss', graph_test)
           np.save('.../path/to/save/model_accuracy',accuracy)
           

         print('Completed')

if __name__ == '__main__':
    tf.app.run()
