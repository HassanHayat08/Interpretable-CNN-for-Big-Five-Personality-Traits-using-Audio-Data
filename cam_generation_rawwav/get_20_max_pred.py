# Interpretable cnn for big five personality traits using audio data #
# Get 20 max predictions of each traits #

import scipy.io
import numpy as np
import pandas as pd
import tensorflow as tf
import heapq

# Load files.
model_preds = np.load('.../path/to/load/model_pred.npy')
model_conv_features = np.load('.../path/to/load/model_conv_features.npy')
model_inputs = np.load('.../path/to/load/model_inputs.npy')

model_preds = model_preds[:,0,:]
model_conv_features = model_conv_features[:,0,:,:]
model_inputs = model_inputs[:,0:,:]

# Extraversion predictions.
extra_pred = model_preds[:,0]
# Agreeableness predictions.
agree_pred = model_preds[:,1]
# Conscientiousness predictions.
consc_pred = model_preds[:,2]
# Neurotisicm predictions.
neuro_pred = model_preds[:,3]
# Openness predictions.
open_pred = model_preds[:,4]

# Take 20 highest prediction of the extraversion.
idx_extra_max = heapq.nlargest(20,range(len(extra_pred)),extra_pred.take)
# Take 20 highest prediction of the Agreeableness.
idx_agree_max = heapq.nlargest(20,range(len(agree_pred)),agree_pred.take)
# Take 20 highest prediction of the Conscientiousness.
idx_consc_max = heapq.nlargest(20,range(len(consc_pred)),consc_pred.take)
# Take 20 highest prediction of the Neurotisicm.
idx_neuro_max = heapq.nlargest(20,range(len(neuro_pred)),neuro_pred.take)
# Take 20 highest prediction of the Openness.
idx_open_max = heapq.nlargest(20,range(len(open_pred)),open_pred.take)

input_video_extra_max = []
conv_output_extra_max = []

input_video_agree_max = []
conv_output_agree_max = []

input_video_consc_max = []
conv_output_consc_max = []

input_video_neuro_max = []
conv_output_neuro_max = []

input_video_open_max = []
conv_output_open_max = []

for i in range (20):
    # Extraversion. 
    # Max.
    video_index_max =  idx_extra_max[i]
    # take corresponding video fft and conv_output. 
    input_video = model_inputs[video_index_max][:][:]
    conv_output = model_conv_features[video_index_max][:][:]
    input_video_extra_max.append(input_video)
    conv_output_extra_max.append(conv_output )
    # Agreeableness. 
    # Max.
    video_index_max =  idx_agree_max[i]
    # take corresponding video fft and conv_output.
    input_video = model_inputs[video_index_max][:][:]
    conv_output = model_conv_features[video_index_max][:][:]
    input_video_agree_max.append(input_video)
    conv_output_agree_max.append(conv_output )
    # Conscientiousness. 
    # Max.
    video_index_max =  idx_consc_max[i]
    # take corresponding video fft and conv_output.
    input_video = model_inputs[video_index_max][:][:]
    conv_output = model_conv_features[video_index_max][:][:]
    input_video_consc_max.append(input_video)
    conv_output_consc_max.append(conv_output )
    # Neurotisicm. 
    # Max.
    video_index_max =  idx_neuro_max[i]
    # take corresponding video fft and conv_output.
    input_video = model_inputs[video_index_max][:][:]
    conv_output = model_conv_features[video_index_max][:][:]
    input_video_neuro_max.append(input_video)
    conv_output_neuro_max.append(conv_output )
    # Openness. 
    # Max.
    video_index_max =  idx_open_max[i]
    # take corresponding video fft and conv_output.
    input_video = model_inputs[video_index_max][:][:]
    conv_output = model_conv_features[video_index_max][:][:]
    input_video_open_max.append(input_video)
    conv_output_open_max.append(conv_output)
    

 
np.save('.../path/to/save/input_feature_extra_max_fft',input_video_extra_max)
np.save('.../path/to/save/conv_output_extra_max_fft',conv_output_extra_max)
np.save('.../path/to/save/input_feature_agree_max_fft',input_video_agree_max)
np.save('.../path/to/save/conv_output_agree_max_fft',conv_output_agree_max)
np.save('.../path/to/save/input_feature_consc_max_fft',input_video_consc_max)
np.save('.../path/to/save/conv_output_consc_max_fft',conv_output_consc_max)
np.save('.../path/to/save/input_feature_neuro_max_fft',input_video_neuro_max)
np.save('.../path/to/save/conv_output_neuro_max_fft', conv_output_neuro_max)
np.save('.../path/to/save/input_feature_open_max_fft',input_video_open_max)
np.save('.../path/to/save/conv_output_open_max_fft', conv_output_open_max)
 
print('completed')


