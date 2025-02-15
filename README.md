# Interpretable CNN for Big Five Personality Traits using Audio Data

This repository contains the source code for the paper:

**Hassan Hayat, Carles Ventura, Agata Lapedriza, “On the Use of Interpretable CNN for Personality Trait Recognition from Audio”, Artificial Intelligence Research and Development, Vol: 69, Pages: 135-144, 2019. DOI: [10.3233/FAIA190116](https://doi.org/10.3233/FAIA190116)**

This package was developed by Mr. Hassan Hayat (hhassan0@uoc.edu). Please feel free to contact him if you have any queries regarding the package. Use this package at your own risk; it is free for academic use.

---

## Operating System

- Ubuntu Linux

---

## Requirements

- **Python:** 3.x.x
- **GPU:** With CUDA support
- **Audio Features:** [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
- **TensorFlow:** 1.14
- **MATLAB:** 2014b
- **libffmpeg**

---

## Dataset

- **ChaLearn Dataset:** [ChaLearn LAP 2016](http://chalearnlap.cvc.uab.es/dataset/24/description/)

---

## Setup and Usage

### 1. Preprocessing: Extract WAV Files from MP4 Videos

Extract WAV files from MP4 video files in the same directory:

```bash
./data_processing/mp4_to_wav.sh
```

---

### 2. Using Spectrogram

#### Generate 2D Mel-Spectrogram

Use `vggish_input.py` and `mel_features.py` from the Audio features to generate a 2D Mel-Spectrogram.  
The Mel-Spectrogram is divided into two categories:

- **Clip_Spectrogram:** A 3D matrix where:
  - 1st dimension represents the length (in seconds) of the WAV file,
  - 2nd dimension represents the number of frames per second,
  - 3rd dimension represents frequency bands.

#### Fine-Tune CNN Models on Clip-Based Spectrograms

- **Fine-Tune Regular CNN:**

  ```bash
  ./clip_spectrogram_cnn/regular_cnn_model.py
  ```

- **Fine-Tune CAM-CNN:**

  ```bash
  ./clip_spectrogram_cnn/cam_cnn_model.py
  ```

#### CAM Generation for Clip-Based Spectrograms

- **Store Inputs with Corresponding Convolutional Features and Last Layer Weights:**

  ```bash
  ./cam_generation_clip_spectrogram/get_features_input_weights.py
  ```

- **Get 20 Maximum Predictions of All Big Five Personality Traits:**

  ```bash
  ./cam_generation_clip_spectrogram/get_20_max_pred.py
  ```

- **Generate CAM Mapping:**

  ```bash
  ./cam_generation_clip_spectrogram/cam_mapping.m
  ```

---

### 3. Using Summary Spectrogram

**Summary_Spectrogram Description:**  
This holds the complete audio information of a video by concatenating clip-based 2D log-Mel spectrograms along the temporal domain, resulting in a 1248x64 spectrogram. An average pooling operation (averaging 60 ms frames across all 64 Mel-spaced frequency bins) is then applied, producing a 208x64 Summary-Spectrogram.

#### Fine-Tune CNN Models on Summary Spectrogram

- **Fine-Tune Regular CNN:**

  ```bash
  ./summary_spectrogram_cnn/regular_cnn_model.py
  ```

- **Fine-Tune CAM-CNN:**

  ```bash
  ./summary_spectrogram_cnn/cam_cnn_model.py
  ```

#### CAM Generation for Summary Spectrogram

- **Store Inputs with Corresponding Convolutional Features and Last Layer Weights:**

  ```bash
  ./cam_generation_summary_spectrogram/get_features_input_weights.py
  ```

- **Get 20 Maximum Predictions of All Big Five Personality Traits:**

  ```bash
  ./cam_generation_summary_spectrogram/get_20_max_pred.py
  ```

- **Generate CAM Mapping:**

  ```bash
  ./cam_generation_summary_spectrogram/cam_mapping.m
  ```

---

### 4. Using Raw Audio WAV

#### Preprocessing Raw Audio

- **Downsample the Audio Signal:**

  ```bash
  ./data_processing/signal_downsamplng.py
  ```

- **Convert Raw WAV into Different Frequency Bins:**

  ```bash
  ./data_processing/get_fft_features.py
  ```

#### Train CAM-CNN on Raw Audio

- **Train the CAM-CNN Model:**

  ```bash
  ./rawwav_cnn/cam_cnn_model.py
  ```

#### CAM Generation for Raw Audio

- **Store Inputs with Corresponding Convolutional Features and Last Layer Weights:**

  ```bash
  ./cam_generation_rawwav/get_features_input_weights.py
  ```

- **Get 20 Maximum Predictions of All Big Five Personality Traits:**

  ```bash
  ./cam_generation_rawwav/get_20_max_pred.py
  ```

- **Generate CAM Mapping:**

  ```bash
  ./cam_generation_rawwav/cam_mapping.py
  ```

---

## License

This project is licensed under the MIT License.

---

## Contact

For any questions or further information regarding this package, please contact:

**Mr. Hassan Hayat**  
Email: [hhassan0@uoc.edu](mailto:hhassan0@uoc.edu)
```
