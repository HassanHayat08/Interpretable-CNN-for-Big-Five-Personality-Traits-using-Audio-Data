### Interpretable cnn for big five personality traits using audio data ###
### This script computes the FFT of an audio signal ###

import numpy as np
import soundfile as sf
import os
import csv
import h5py
from scipy.io import wavfile
from scipy.fftpack import fft,fftfreq

fft_features = []

data = np.load('/...path/to/downsampled_data/')

for i in range(len(data)):
    data = data[i]
    data = fft(data)
    fft_features.append(data)
np.save('../save/path/',fft_features)
