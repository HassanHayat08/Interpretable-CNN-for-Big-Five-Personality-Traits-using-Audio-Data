# Interpretable cnn for big five personality traits using audio data #
# Get conv_features with corresponding input and output layer weights #


import numpy as np
import tensorflow as tf
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
        # Max Pooling.
        net = tf.layers.max_pooling1d(net,10,10,padding='valid',name='max_pooling_1')
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Convolutional layer.
        net = tf.nn.conv1d(net, weights['wc2'], stride=1, padding="SAME", name='conv_2')
        net = tf.nn.bias_add(net, biases['bc2'])
        net = tf.nn.sigmoid(net)
        # Max Pooling.
        net = tf.layers.max_pooling1d(net,8,8,padding='valid',name='max_pooling_2')
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Convolutional layer.
        net = tf.nn.conv1d(net, weights['wc3'], stride=1, padding="SAME", name='conv_3')
        net = tf.nn.bias_add(net, biases['bc3'])
        net = tf.nn.sigmoid(net)
        # Max Pooling.
        net = tf.layers.max_pooling1d(net,8,8,padding='valid',name='max_pooling_3')
        # Last convolution output.        
        conv_output = tf.identity(net, name='conv_output')
        # Global Average Pooling GAP layer.
        net = tf.reduce_mean(net, axis=[1])
        # Dropout.
        net = tf.nn.dropout(net,keep_prob)
        # Output layer.
        logits = tf.contrib.layers.fully_connected(net,param.NUM_CLASSES,activation_fn=None,    
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer(),trainable=False,scope='output_layer')
        
        with tf.variable_scope('train'):
            loss_tensor = tf.losses.huber_loss(y,logits)
            optimizer = tf.train.AdamOptimizer(learning_rate=param.LEARNING_RATE)
            train = optimizer.minimize(loss_tensor,name='train_op')
        saver = tf.train.Saver()
        
        # Data loaders. 
        testfeatures = np.load('.../path/to/load/fft_features/.npy')
        test_labels = np.load('.../path/to/load/feature_labels/.npy')

        with tf.Session() as sess:
            # Restore cmm model weights.
            saver.restore(sess,".../path/to/load/model/weights.ckpt")
            print("Model restored.")
            # Locate conv_ouput tensor.
            conv_output = sess.graph.get_tensor_by_name('conv_output:0')
            model_pred = []
            model_inputs = []
            model_conv_features = []
            last_layer_wght = []
            model_labels = []
            rand_array = [i for i in range(len(test_features))]
            np.random.shuffle(rand_array1)
            all_vars= tf.all_variables()
            var = [v for v in tf.trainable_variables() if v.name == "output_layer/weights:0"][0]
            for j in range(len(test_features)):
                index = rand_array[j]
                t_input = test_features[index]
                t_input = np.reshape(t_input,(1,t_input.shape[0],1))
                t_label = test_labels[index]
                t_label = np.reshape(t_label,[1,param.NUM_CLASSES])
                pred,loss,conv_features,layer_wght = sess.run([logits,loss_tensor,convr_output,var],
                                                              feed_dict={ x: v_input, y: v_label, keep_prob : 1.0})
                model_pred.append(pred)
                input_labels.append(t_label)
                model_conv_features.append(conv_features)
                model_inputs.append(t_input)
                last_layer_wght.append(layer_wght)
                    
        
            np.save('.../path/to/save/model_pred',model_pred)
            np.save('.../path/to/save/model_labels',model_labels)
            np.save('.../path/to/save/model_conv_features',model_conv_features)
            np.save('.../path/to/save/model_inputs',model_inputs)
            np.save('.../path/to/save/last_layer_wght',last_layer_wght)
            print("Finished!")

if __name__ == '__main__':
    tf.app.run() 



