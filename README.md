# EMOTION_CLASSIFICATION_ON_SPEECH_DATA
A robust Speech Emotion Recognition system using MFCC, Chroma, Mel-Spectrogram, Spectral Contrast, Tonnetz, and VGGish features. Includes data augmentation, hyperparameter tuning, and a 3D CNN for accurate multi-class emotion classification from audio.

Accuracy metrics:
Final Accuracy on Validation Data: 59.02%
80/80 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step
              precision    recall  f1-score   support

       angry       0.91      0.51      0.65       378
        calm       0.76      0.60      0.67       378
     disgust       0.78      0.62      0.69       230
     fearful       0.90      0.50      0.64       378
       happy       0.30      0.93      0.46       377
     neutral       0.80      0.51      0.62       189
         sad       0.84      0.44      0.58       378
   surprised       0.85      0.59      0.70       230

    accuracy                           0.59      2538
   macro avg       0.77      0.59      0.63      2538
weighted avg       0.76      0.59      0.62      2538

✅ 1. Project Description
This project aims to detect human emotions from speech audio using deep learning. It uses audio recordings (in .wav format) to classify emotions like happy, sad, angry, neutral, etc. The system extracts acoustic features from audio, processes them through a trained model, and outputs the detected emotion. A Streamlit-based web app makes it easy for users to upload audio files and get real-time predictions.

✅ 2. Pre-processing Methodology
Raw .wav audio files cannot be directly fed to a machine learning model. Preprocessing is required to extract meaningful patterns. Steps include:

Resampling Audio: All files are resampled to a consistent sampling rate (22050 Hz) for uniformity.

Audio Trimming: Clips are trimmed/padded to a fixed duration (e.g., 3 seconds).

Feature Extraction:

MFCCs (Mel-frequency cepstral coefficients) are extracted using librosa. These represent short-term power spectrum of sound, capturing tone and pitch.

The extracted MFCCs are scaled and averaged into fixed-size vectors to serve as model input.

✅ 3. Model Pipeline
The system uses a deep learning pipeline, combining Convolutional Neural Networks (CNN) and LSTM layers for best performance.

📦 Pipeline Steps:
Input Layer: Receives a 1D MFCC feature vector (flattened).

Convolutional Layers:

Detect local temporal patterns in the features.

Use ReLU activation + dropout for regularization.

Bidirectional LSTM Layer:

Learns sequential emotional patterns in both forward and backward directions.

Dense Layer:

Final layer with softmax activation that outputs probability for each emotion.

Output:

Emotion label with the highest probability (e.g., HAPPY, SAD, ANGRY etc.)

