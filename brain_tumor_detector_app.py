import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Image dimensions and class names as per configuration
IMG_WIDTH, IMG_HEIGHT = 150, 150
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']


# Cache the model loading so it doesn't reload on every interaction
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model("improved_brain_tumor_model.keras")
    return model


# Load the model
model = load_trained_model()

# Streamlit app title and instructions
st.title("Brain Tumor Classification App")
st.write(
    """
    Upload a brain MRI image (jpg, jpeg, or png) to determine whether it shows signs of a brain tumor and,
    if so, which type: glioma, meningioma, notumor, or pituitary.
    """
)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image: resize and normalize
    image_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(image_resized)

    # If the image has an alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # Normalize pixel values to [0,1]
    img_array = img_array / 255.0

    # Expand dimensions to match the model's input shape (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction using the loaded model
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    # Display the prediction results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
