# Interpretable cnn for big five personality traits using audio data #
# Generate cam for 20 highest predictions of each prosnality traits #

import numpy as np
import pandas as pd
import scipy.io

def extraversionCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir):
    for i in range (20):
        df = pd.DataFrame(predictions,columns=['A','B','C','D','E'])
        # EXTRAVERSION, 20 highest prediction 
        max_value_extra = df['A'].idxmax()
        # Take corresponding input mfcc 
        input_clip_max = input_mfcc_features[max_value_extra][:][:]
        scipy.io.savemat(save_dir+str(i)+str(max_value_extra),mdict={'arr':input_clip_max})
        # CAM generation.
        weights = tf.reshape(output_layer_wghts,[5,512])
        extraversion_weights = weights[0,:]
        result = conv_features[max_value_extra][:][:][:] * extraversion_weights
        result = result.eval()
        cam = np.sum(result,axis=2)
        scipy.io.savemat(save_dir+str(i)+str(max_value_extra),mdict={'arr':cam})


def agreeablenessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir):
    for i in range(20):
        df = pd.DataFrame(predictions,columns=['A','B','C','D','E'])
        # AGREEABLENESS, 20 highest prediction.
        max_value_agree = df['B'].idxmax()
        #  Take corresponding input mfcc.
        input_clip_max = input_mfcc_features[max_value_agree][:][:]
        scipy.io.savemat(save_dir+str(i)+str(max_value_agree),mdict={'arr':input_clip_max})
        #  CAM generation. 
        weights = tf.reshape(output_layer_wghts,[5,512])
        agreeableness_weights = weights[1,:]
        result = conv_features[max_value_agree][:][:][:] * agreeableness_weights
        result = result.eval()
        cam = np.sum(result,axis=2)
        scipy.io.savemat(save_dir+str(i)+str(max_value_agree),mdict={'arr':cam})
        
        
def conscientiousnessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir):
    for i in range(20):
        df = pd.DataFrame(predictions,columns=['A','B','C','D','E'])
        # CONSCIENTIOUSNESS, 20 highest prediction.
        max_value_consc = df['C'].idxmax()
        # Take corresponding input mfcc.
        input_clip_max = input_mfcc_features[max_value_consc][:][:]
        scipy.io.savemat(save_dir+str(i)+str(max_value_consc),mdict={'arr':input_clip_max})
        # CAM generation. 
        weights = tf.reshape(output_layer_wghts,[5,512])
        Conscientiousness_weights = weights[2,:]
        result = conv_features[max_value_consc][:][:][:] * Conscientiousness_weights
        result = result.eval()
        cam = np.sum(result,axis=2)
        scipy.io.savemat(save_dir+str(i)+str(max_value_consc),mdict={'arr':cam})
        

def NeurotisicmCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir):
    for i in range(20):
        df = pd.DataFrame(predictions,columns=['A','B','C','D','E'])
        # NEUROTISICM,20 highest trait.
        max_value_neuro = df['D'].idxmax()
        # Take corresponding input mfcc. 
        input_clip_max = input_mfcc_features[max_value_neuro][:][:]
        scipy.io.savemat(save_dir+str(i)+str(max_value_neuro),mdict={'arr':input_clip_max})
        # CAM generation. 
        weights = tf.reshape(output_layer_wghts,[5,512])
        Neurotisicm_weights = weights[3,:]
        result = conv_features[max_value_neuro][:][:][:] * Neurotisicm_weights
        result = result.eval()
        cam = np.sum(result,axis=2)
        scipy.io.savemat(save_dir+str(i)+str(max_value_neuro),mdict={'arr':cam})
        

def OpennessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir):
    for i in range(20):
        df = pd.DataFrame(predictions,columns=['A','B','C','D','E'])
        # Openness,20 highest trait. 
        max_value_open = df['E'].idxmax()
        # Take corresponding input mfcc. 
        input_clip_max = input_mfcc_features[max_value_open][:][:]
        scipy.io.savemat(save_dir+str(i)+str(max_value_open),mdict={'arr':input_clip_max})
        # CAM generation. 
        weights = tf.reshape(output_layer_wghts,[5,512])
        Openness_weights = weights[4,:]
        result = conv_features[max_value_open][:][:][:] * Openness_weights
        result = multiply_result.eval()
        cam = np.sum(result,axis=2)
        scipy.io.savemat(save_dir+str(i)+str(max_value_open),mdict={'arr':cam})
        

# data loaders.
prediction = '.../path/to/load/model/predictions'
output_layer_wghts = '.../path/to/load/output_layer/wghts'
input_mfcc_features = '.../path/to/load/input_features'
conv_features = '.../path/to/load/conv_features'
save_dir = '.../path/to/save'

extraversionCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir)
agreeablenessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir)
conscientiousnessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir)
NeurotisicmCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir)
OpennessCAM(predictions,input_mfcc_features,conv_features,output_layer_wghts,save_dir)
