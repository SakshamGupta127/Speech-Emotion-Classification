import streamlit as st
from utils import predict_emotion
import tempfile

# Set the app title and icon
st.set_page_config(page_title="🎙️ Speech Emotion Classifier", page_icon="🎧")

# Title section
st.title("🎙️ Speech Emotion Recognition App")
st.markdown("Upload a `.wav` file and I’ll predict the emotion from your voice!")

# File uploader
uploaded_file = st.file_uploader("🎧 Choose an audio file", type=["wav"])

# If a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # Play the audio
    st.audio(uploaded_file, format="audio/wav")

    # Predict button
    if st.button("🔍 Predict Emotion"):
        try:
            prediction = predict_emotion(temp_path)
            st.success(f"🎉 Emotion Detected: **{prediction.upper()}**")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
