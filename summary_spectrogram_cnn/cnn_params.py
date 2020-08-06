# Interpretable cnn for big five personality traits using audio data #
# Parameters initalization #

NUM_FRAMES = 208 # No of Frames for summary spectrogram.
NUM_BANDS = 64  # No of frequency bands for mel-spectrogram.

# Model hyperparameters.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

No_of_Epochs = 50 # No of epochs for training the model.
NUM_CLASSES = 5   # No of output classess.

kernel_size_x = 26 # x-axis dimension for GAP layer.
kernel_size_y = 8  # y-axis dimension for GAP layer.


