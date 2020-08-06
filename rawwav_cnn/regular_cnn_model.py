# Interpretable cnn for big five personality traits using audio data #
# cnn model using raw audio as an input #


import numpy as np
import tensorflow as tf
from random import shuffle
import random
import cnn_params as param


def main(_):
    with tf.Graph().as_default():

        x, y = tf.placeholder(tf.float32, shape=[None,param.raw_audio_dim,1]), tf.placeholder(tf.float32, shape=[1,param.NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)
        # Store layers weight & bias
        weights = {
        # conv, 1 input, 64 outputs
        'wc1': tf.Variable(tf.random_normal([8, 1, 64])),
        # conv, 64 inputs, 128 outputs
        'wc2': tf.Variable(tf.random_normal([6, 64, 128])),
        # conv, 128 inputs, 256 outputs
        'wc3': tf.Variable(tf.random_normal([6, 128, 256])),}
        biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bc3': tf.Variable(tf.random_normal([256])),}
        # Convolutional layer.
        net = tf.nn.conv1d(x, weights['wc1'], stride=1, padding="SAME", name='conv_1') 
        net = tf.nn.bias_add(net, biases['bc1'])
        net = tf.nn.sigmoid(net)
        # Max pooling.
        net = tf.layers.max_pooling1d(net,10,10,padding='valid',name='max_pooling_1')
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Convolutional layer.
        net = tf.nn.conv1d(net, weights['wc2'], stride=1, padding="SAME", name='conv_2') 
        net = tf.nn.bias_add(net, biases['bc2'])
        net = tf.nn.sigmoid(net)
        # Max pooling.
        net = tf.layers.max_pooling1d(net,8,8,padding='valid',name='max_pooling_2')
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Convolutional layer.
        net = tf.nn.conv1d(net, weights['wc3'], stride=1, padding="SAME", name='conv_3') 
        net = tf.nn.bias_add(net, biases['bc3'])
        net = tf.nn.sigmoid(net)
        # Max pooling.
        net = tf.layers.max_pooling1d(net,8,8,padding='valid',name='max_pooling_3')
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Output layer.
        logits = tf.contrib.layers.fully_connected(net,param.NUM_CLASSES,activation_fn=None,    
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),trainable=True,scope='output_layer')
        
        with tf.variable_scope('train'):
            loss_tensor = tf.losses.huber_loss(y,logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=param.LEARNING_RATE )
            train = optimizer.minimize(loss_tensor,name='train_op')
        saver = tf.train.Saver()

        # data loaders. 
        train_features = np.load('.../path/to/load/fft_features/.npy')
        train_labels = np.load('.../path/to/load/feature_labels/.npy')
        test_features = np.load('.../path/to/load/fft_features/.npy')
        test_labels = np.load('.../path/to/load/feature_labels/.npy')
        
        # Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Locate train tensor for the training loop.
            train_op = sess.graph.get_operation_by_name('train/train_op')
                        
            train_loss = []
            test_loss = []
            mean_train_loss = []
            mean_test_loss = []
            model_accuracy = []
            pred_accuracy = []
            
            for epoch in range(param.No_of_Epochs):
                print("Epoch:", '%04d' % (epoch+1))
                print('Training...')
                rand_array = [i for i in range(len(train_features))]
                np.random.shuffle(rand_array)
                for i in range(len(train_features)):
                    index = rand_array[i]
                    t_input = train_features[index]
                    t_label = train_labels[index]
                    t_label = np.reshape(t_label,[1,param.NUM_CLASSES])
                    pred,_,loss = sess.run([logits,train_op,loss_tensor],feed_dict={ x: t_input, y: t_label,keep_prob:0.5})
                    train_loss.append(loss)
                mean_loss = np.mean(train_loss)
                mean_train_loss.append(mean_loss)
                if epoch == param.No_of_Epochs - 1 :
                        save_path = saver.save(sess, "...path/to/save/model/weights/.ckpt")
                
                print('Testing...')
                rand_array1 = [i for i in range(len(test_features))]
                np.random.shuffle(rand_array1)
                for j in range(len(test_features)):
                    index = rand_array1[j]
                    t_input = test_features[index]
                    t_label = test_labels[index]
                    t_label = np.reshape(t_label,[1,param.NUM_CLASSES])
                    pred,loss = sess.run([logits,loss_tensor],feed_dict={ x: t_input, y: t_label,keep_prob:1.0})
                    test_loss.append(loss)
                    difference = t_label - pred
                    abs_difference = np.absolute(difference)
                    one_diff = 1 - abs_difference
                    sum_1 = np.sum(one_diff)
                    accuracy = sum_1/5
                    pred_accuracy.append(accuracy)
                mean_loss = np.mean(test_loss)
                mean_test_loss.append(mean_loss)
                accuracy = np.mean(pred_accuracy)
                model_accuracy.append(accuracy)
                        
        
            np.save('...path/to/save/train_loss',mean_train_loss)
            np.save('...path/to/save/test_loss',mean_test_loss)
            np.save('.../path/to/save/model_accuracy',model_accuracy)
            print('completed')


if __name__ == '__main__':
    tf.app.run()       
        
