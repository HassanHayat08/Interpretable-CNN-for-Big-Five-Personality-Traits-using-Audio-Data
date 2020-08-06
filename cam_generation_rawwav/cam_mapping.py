# Interpretable cnn for big five persoanlity traits using audio data #
# generate cam mapping on raw wav data #


import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import wave
from scipy import signal
from scipy.fftpack import fft,fftfreq

# Load files.
conv_features = np.load('.../Path/to/load/conv_features.npy')
model_inputs =  np.load('.../path/to/load/model_inputs.npy')
layer_wghts =   np.load('.../path/to/load/layer_wghts.npy')

model_inputs = model_inputs[:,0,:,0]
layer_wghts = layer_wghts[1,:,:]

all_wghts = []
# Weights for each personality traits. 
extra_wghts = layer_wghts[:,0]
extra_wghts = np.reshape(extra_wghts,[1,256])
all_wghts.append.append(extra_wghts)
agree_wghts = layer_wghts[:,1]
agree_wghts = np.reshape(agree_wghts,[1,256])
all_wghts.append(agree_wghts)
consc_wghts = layer_wghts[:,2]
consc_wghts = np.reshape(consc_wghts,[1,256])
all_wghts.append(consc_wghts)
neuro_wghts = layer_wghts[:,3]
neuro_wghts = np.reshape(neuro_wghts,[1,256])
all_wghts.append(neuro_wghts)
open_wghts = layer_wghts[:,4]
open_wghts = np.reshape(open_wghts,[1,256])
all_wghts.append(open_eghts)

for j in range(len(all_wghts)):
    if j==0:
        trait_name = 'Extraversion'
    else if j==1:
        trait_name = 'Agreeableness'
    else if j==2:
        trait_name = 'Conscientiousness'
    else if j==3:
        trait_name = 'Openness'
    else if j==4:
        trait_name = 'Neuroticism'
    for i in range(5)
        fea = conv_features[i]
        weights = all_wghts[j]
        freq_map = np.multiply(fea,weights)
        freq_map = np.sum(freq_map,axis=0)
        freq_map_cmb.append(freq_map)
        # audio signal.
        signal = model_inputs[i]
        n = np.size(signal)
        freqs = fftfreq(n)
        mask = freqs > 0
        fft_theo = 2.0 * np.abs(signal/n)
        plt.figure(1)
        plt.figure(figsize=(10,6))
        plt.title('Raw Wav')
        plt.xlabel('Time/Distance',fontsize=20)
        plt.ylabel('Amplitude',fontsize=20)
        plt.plot(signal)
        plt.show()
        plt.savefig('...path/to/save/Raw_Wav' + str(i)+'.eps')
        # FFT 
        plt.figure(figsize=(10,6))
        plt.figure(2)
        plt.figure(figsize=(10,6))
        plt.title('Raw Wav Frequency Domain')
        plt.xlabel('Frequency',fontsize=20)
        plt.ylabel('Amplitude',fontsize=20)
        plt.plot(freqs[mask],fft_theo[mask])
        plt.show()
        plt.savefig('...path/to/save/freq' + str(i)+'.eps')
        # Cam fft
        m = np.size(freq_map)
        freqs1 = fftfreq(m)
        mask1 = freqs1 > 0
        fft_theo1 = 2.0 * np.abs(freq_map/m)
        plt.figure(3)
        plt.figure(figsize=(10,6))
        plt.title('%Trait name'%trait_name)
        plt.xlabel('Frequency',fontsize=20)
        plt.ylabel('Amplitude',fontsize=20)
        plt.plot(freqs1[mask1],fft_theo1[mask1])
        plt.show()
        plt.savefig('.../path/to/save/cam_freq' + str(i)+'.eps')
    


