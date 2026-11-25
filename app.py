import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("🏏 DeepGuard - Cricketer Recognition")

model = load_model("cricketer_cnn_model.keras")
classes = np.load("cricketer_classes.npy")

uploaded = st.file_uploader(
    "Upload Cricketer Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    img = image.load_img(uploaded, target_size=(64,64))
    arr = image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)
    idx = np.argmax(pred)

    st.success(f"Predicted Cricketer: {classes[idx]}")
    st.write(f"Confidence: {pred[0][idx]*100:.2f}%")
