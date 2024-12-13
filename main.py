import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

new_model = tf.keras.models.load_model('./model/traffic_classifier.keras')

def load_and_process_image(image):
    img = Image.open(image)
    img = img.resize((30, 30))
    img = np.expand_dims(img, axis=0)
    img = np.array(img)
    return img

def predict(img):
    pred = new_model.predict(img)
    predicted_class = np.argmax(pred, axis=-1)
    confidence = np.max(pred) * 100  # Convert to percentage
    return (pred, predicted_class[0], confidence)

st.title("Traffic Sign Classifier")
st.write("Upload an image of a traffic sign to make a prediction.")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

    img = load_and_process_image(uploaded_image)
    (predictions, predicted_class, confidence) = predict(img)

    with col2:
        st.subheader(f"Predicted Sign (Class {predicted_class})")
        meta_image_path = f"./data/Meta/{predicted_class}.png"
        if os.path.exists(meta_image_path):
            meta_image = Image.open(meta_image_path)
            st.image(meta_image, use_container_width=True)
        else:
            st.warning(f"Meta image for class {predicted_class} not found.")

    st.markdown(
        f"""
        <div style="
            background-color: #10141b;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            margin-top: 20px;
        ">
            <h3 style="margin-bottom: 10px;">Confidence</h3>
            <div style="
                background: linear-gradient(to right,
                    hsl({min(120, confidence * 1.2)}, 80%, 45%) 0%,
                    hsl({min(120, confidence * 1.2)}, 80%, 45%) {confidence}%);
                color: white;
                padding: 5px;
                border-radius: 5px;
                font-size: 24px;
                font-weight: bold;
            ">
                {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.info("Upload an image to start the prediction.")