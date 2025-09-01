import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image


# Function to preprocess an uploaded image
def preprocess_image(image):
    # Convert the PIL Image to a NumPy array
    image = np.array(image)

    # Check if the image is already grayscale (1 channel)
    if len(image.shape) == 3 and image.shape[2] == 3:  # If it's RGB
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:  # If it's RGBA
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # Resize to 64x64
    image = cv2.resize(image, (64, 64))

    # Normalize pixel values
    image = image / 255.0

    # Reshape to include batch and channel dimensions
    image = image.reshape((1, 64, 64, 1))

    return image

# Load the trained model
model = models.load_model("saved_model\saved_model.pb")

# Streamlit App
st.title("Signature Forgery Checker")

# Upload image inputs
# Create two columns for side-by-side display
col1, col2 = st.columns(2)

with col1:
    image_file_A = st.file_uploader("Upload Original Image", type=["jpg", "png", "jpeg"])

with col2:
    image_file_B = st.file_uploader("Upload Forged Image", type=["jpg", "png", "jpeg"])

# Confidence threshold slider
confidence_threshold = st.slider(
    "Set Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)


# Process and predict when both images are uploaded
if image_file_A and image_file_B:
    # Read and preprocess images
    img_A = Image.open(image_file_A)
    img_B = Image.open(image_file_B)

    preprocessed_img_A = preprocess_image(img_A)
    preprocessed_img_B = preprocess_image(img_B)

    st.subheader("Preprocessed Images")
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_A, caption="Original Image", use_container_width=True)

    with col2:
        st.image(img_B, caption="Forged Image", use_container_width=True)
    # Predict similarity score
    similarity_score = model.predict([preprocessed_img_A, preprocessed_img_B]).flatten()[0]

    # Display results
    st.subheader("Results")
    st.write(f"Similarity Score: {similarity_score:.4f}")
    if similarity_score > confidence_threshold:
        st.success("The images are similar!")
    else:
        st.error("The images are not similar.")

    # Display the confidence threshold used
    st.write(f"Confidence Threshold: {confidence_threshold}")
