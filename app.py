import streamlit as st
from PIL import Image

st.set_page_config(page_title="Fake Image Detector", layout="centered")

st.title("Deepfake Image Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Placeholder for prediction
    st.subheader("Prediction:")
    st.write("Model output will appear here (real/fake)")
