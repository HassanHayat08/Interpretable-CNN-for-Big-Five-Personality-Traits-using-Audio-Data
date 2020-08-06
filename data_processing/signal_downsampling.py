### Interpretable cnn for big five personality traits using audio data ###
### This script downsamples 41000 kz signal into 4000 kz signal ###


from __future__ import absolute_import, division, print_function
import pathlib
import random
import csv
import numpy as np
from scipy.io import wavfile
import tensorflow as tf 
import itertools
from scipy import stats


### functions for mapping ###
def normalize_with_moments(data, axes=[0], epsilon=1e-8):
    mean, variance = tf.nn.moments(data, axes=axes)
    data_normed = (data - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return data_normed

def get_wav(path, label):
    wav_file = tf.read_file(path)
    data = tf.contrib.ffmpeg.decode_audio(tf.read_file(path), file_format="wav",samples_per_second=4000, channel_count=1)
    data = tf.cast(data,tf.complex64)
    data = tf.fft(data,name='FFT')
    return normalize_with_moments(data), label


### down sample the data ###
data = []
labels = []

folder_path = '/...path/to/wav/data/folder/'
folder_path = pathlib.Path(folder_path)
files_path = list(folder_path.glob('*.wav'))
files_path = [str(path) for path in files_path]
no_of_samples = len(files_path)


### load data labels ###
with open('/...path/to/.csv/labels/file', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data.append(row)
        
for i in range(len(files_path)):
    file_1 = files_path[i]
    file_1 = file_1.split("/")[5]
    file_name_1 = file_1[:-4]
    new_filename_1 = file_name_1 + '.mp4'
    label_1 = []
    label_2 = []
    matching = [s for s in data if new_filename_1 in s]
    label_1= np.delete(matching,[0],axis=1)
    label_2 = label_1[0,:]
    label_2 = [float(i) for i in label_2]
    labels.append(label_2)
    
### dataset pipeline ###
ds = tf.data.Dataset.from_tensor_slices((files_path, labels))
data_ds = ds.map(get_wav)         
ds = data_ds.shuffle(buffer_size=wavfiles_count)
ds = ds.repeat()
ds = ds.batch(1)
### prefetch the data batches in the background ###
ds = ds.prefetch(buffer_size=1)
iterator = ds.make_one_shot_iterator()
next_ele = iterator.get_next()

features_4k = []
labels_4k = []

with tf.Session() as sess:
    for _ in range(len(files_path)):        
        t_features, t_labels = sess.run(next_ele)
        features_4k.append(t_features)
        labels_4k.append(t_labels)
        
    np.save('.../save/path/',features_4k)
    np.save('.../save/path/',labels_4k)
    print('Completed')
            
            
            
            
