import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Image preprocessing function (you must match this with your training pipeline)
def preprocess_image(image):
    image = image.resize((224, 224))  # Change to your input size
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.set_page_config(page_title="Fake Image Detector", layout="centered")
st.title("Deepfake Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)[0][0]  # Assuming binary output
    label = "FAKE" if prediction > 0.5 else "REAL"

    # Show result
    st.subheader("Prediction:")
    st.markdown(f"**{label}** (Confidence: {prediction:.2f})")
