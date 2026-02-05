
import numpy as np
import librosa
import tensorflow as tf

# Load the trained model
MODEL = tf.keras.models.load_model("model.h5")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(file_path, sr=22050, duration=3, offset=0.5, n_mfcc=40):
    y, sr = librosa.load(file_path, duration=duration, offset=offset, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = MODEL.predict(features)[0]
    return EMOTIONS[np.argmax(prediction)]
