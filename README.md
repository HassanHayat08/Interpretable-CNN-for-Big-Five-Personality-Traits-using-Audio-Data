# Source code of Interpretable CNN for Big Five Personality Traits using Audio Data. (The Code Follows MIT License)

This package was developed by Mr.Hassan Hayat (hhassan0@uoc.edu). Please feel free to contact in case of any query regarding the package. You can run this package at your own risk. This package is free for academic use.

To read the paper: Hassan Hayat, Carles Ventura, Agata Lapedriza, “On the Use of Interpretable CNN for Personality Trait Recognition from Audio”, Artificial Intelligence Research and Development, Vol: 69, Page: 135 - 144, 2019. DOI: 10.3233/FAIA190116

**Operating System** 

- Ubuntu Linux

**Requirements**

- Python3.x.x

- GPU with CUDA support

- Audio features (https://github.com/tensorflow/models/tree/master/research/audioset/vggish)

- Tensorflow1.14

- Matlab 2014b

- libffmpeg

**Dataset**

- Dataset [ChaLearn](http://chalearnlap.cvc.uab.es/dataset/24/description/)

## Setup

*Extract wav files from mp4 video files in the same directory*

./data_processing/mp4_to_wav.sh	

**Using Spectrogram**

*Use vggish_input.py and mel_features.py from Audio feature to generate 2D Mel_Spectrogram*
	
***From here, we divided Mel_Spectrogram into two categories:*** 

*Clip_Spectrogram: is a 3D matrix. In which, 1st dimension represents the length (in second) of the wav file, 2nd dimension represents the number of 
frames in a second, and 3rd dimension represents frequency bands.*

**Fine-Tune CNN**

- ./clip_spectrogram_cnn/regular_cnn_model.py

**Fine-Tune CAM_CNN**

- ./clip_spectrogram_cnn/cam_cnn_model.py  

**CAM Generation**

*Store inputs with corresponding conv_features and the last layer weights*

- ./cam_generation_clip_spectrogram/get_features_input_weights.py

*Get 20 maximum predictions of all big five personality traits with corresponding inputs, conv_features, and the last layer weights* 

- ./cam_generation_clip_spectrogram/get_20_max_pred.py

*Generate cam mapping of all big five personality traits* 

- ./cam_generation_clip_spectrogram/cam_mapping.m

*Summary_Spectrogram: It holds the whole audio information of a video and concatenates the clip-based 2D log-Mel spectrograms along with the temporal domain.
This way, we obtain a 1248x64 spectrogram. Then, an average pool operation is performed to reduce the size of the spectrogram. We take an average of 60 ms 
frame across all 64 Mel-spaced frequency bins, obtaining, as a result, a 208x64 spectrogram, which we refer to as Summary-Spectrogram.*

**Fine-Tuned CNN**

- ./summary_spectrogram_cnn/regular_cnn_model.py 

**Fine-Tuned CAM_CNN**

- ./summary_spectrogram_cnn/cam_cnn_model.py 

**CAM Generation**

*Store inputs with corresponding conv_features and the last layer weights*

- ./cam_generation_summary_spectrogram/get_features_input_weights.py  

*Get 20 maximum predictions of all big five personality traits with corresponding inputs, conv_features, and the last layer weights*

- ./cam_generation_summary_spectrogram/get_20_max_pred.py  

*Generate cam mapping of all big five personality traits*
  
- ./cam_generation_summary_spectrogram/cam_mapping.m

 
**Using Raw Audio Wav**

*Down sample the audio signal*

- ./data_processing/signal_downsamplng.py 

*Convert raw wav into different frequency bins*
  
- ./data_processing/get_fft_features.py

**Train CAM_CNN**

*Train the cam_cnn model*

- ./rawwav_cnn/cam_cnn_model.py 	
 
**CAM Generation**

*Store inputs with corresponding conv_features and the last layer weights*

- ./cam_generation_rawwav/get_features_input_weights.py  

*Get 20 maximum predictions of all big five personality traits with corresponding inputs, conv_features, and the last layer weights*  
  
- ./cam_generation_rawwav/get_20_max_pred.py 

*Generate cam mapping of all big five personality traits*
  
- ./cam_generation_rawwav/cam_mapping.py

