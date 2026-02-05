# main.py
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tensorflow as tf
from utils import preprocess_audio  # You must define this in utils.py

model = tf.keras.models.load_model("model.h5")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Speech Emotion API is live!"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    audio_data = await file.read()
    features = preprocess_audio(audio_data)  # Convert to model-ready features
    features = np.expand_dims(features, axis=0)  # batch dimension
    prediction = model.predict(features)
    label = np.argmax(prediction, axis=1)[0]
    return {"prediction": int(label)}
