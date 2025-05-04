import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
# Ensure this path is correct for your environment
model_path = './xception_model.h5'
try:
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load MTCNN detector
detector = MTCNN()

# Title
st.title("Deepfake Detection Web App")
st.write("Upload a facial image and detect whether it's *FAKE* or *REAL*")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Prediction function
def detect_face_and_predict(image):
    """
    Detects faces in the input image, preprocesses the face, and predicts
    whether it's real or fake using the loaded Xception model.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if len(faces) == 0:
        return None, "No face detected"

    # Process only the first detected face
    x, y, w, h = faces[0]['box']
    face_img = image_rgb[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (160, 160))  # Consistent with Xception training
    face_array = img_to_array(face_img) / 255.0
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

    try:
        pred = model.predict(face)[0][0]
        label = "FAKE" if pred > 0.5 else "REAL"
        confidence = f"Prediction Confidence: {pred:.4f}"
        return label, confidence
    except Exception as e:
        return None, f"Error during prediction: {e}"

# Display result
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, channels="BGR", caption="Uploaded Image")

    label, confidence = detect_face_and_predict(image)
    if label:
        st.subheader(f"🕵 Prediction: {label}")
        st.text(confidence)
    else:
        st.warning(confidence)
