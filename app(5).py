import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import joblib

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Cricketer Image Classification",
    page_icon="🏏",
    layout="centered"
)

# ---------------------------------------------------
# Load trained model and label encoder
# ---------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model("cricketer_model.keras")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_resources()

# ---------------------------------------------------
# App title
# ---------------------------------------------------
st.title("🏏 Cricketer Image Classification")
st.write(
    "Upload an image of one of the trained cricketers and the CNN model "
    "will predict the cricketer's name."
)

st.write("Model classes:")
st.write(", ".join(encoder.classes_))

# ---------------------------------------------------
# Upload image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a cricketer image",
    type=["jpg", "jpeg", "jfif", "png", "webp"]
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if uploaded_file is not None:

    # Open image in RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # IMPORTANT:
    # This preprocessing matches the training notebook:
    # 128 x 128, RGB, values scaled to 0-1.
    resized_image = image.resize((128, 128))

    img_array = np.array(resized_image, dtype=np.float32)
    img_array = img_array / 255.0

    # Add batch dimension:
    # (128, 128, 3) -> (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    probabilities = model.predict(img_array, verbose=0)

    predicted_class = int(np.argmax(probabilities[0]))
    predicted_name = encoder.inverse_transform([predicted_class])[0]
    confidence = float(np.max(probabilities[0]) * 100)

    # ---------------------------------------------------
    # Show result
    # ---------------------------------------------------
    st.success(f"Prediction: {predicted_name}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Show all class probabilities
    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(encoder.classes_):
        probability = float(probabilities[0][i] * 100)
        st.write(f"{class_name}: {probability:.2f}%")

    # Small warning for low-confidence output
    if confidence < 50:
        st.warning(
            "The model confidence is low. Try a clearer, front-facing image "
            "with the cricketer's face visible."
        )
