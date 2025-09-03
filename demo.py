import streamlit as st
import cv2
import numpy as np
from tensorflow.keras import models
from PIL import Image
import tempfile
import os

# Function to preprocess an uploaded image with enhanced preprocessing
def preprocess_image(image):
    # Convert the PIL Image to a NumPy array
    image = np.array(image)
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding to enhance signature features
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Resize to 64x64
    image = cv2.resize(image, (64, 64))
    
    # Normalize pixel values
    image = image / 255.0
    
    # Reshape to include batch and channel dimensions
    image = image.reshape((1, 64, 64, 1))
    
    return image

# Function to check if image contains a signature (basic check)
def is_likely_signature(image):
    """Basic check to see if image might contain a signature"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Check if image has reasonable contrast (signatures should have some contrast)
    if np.std(image) < 25:  # Low contrast
        return False
    
    # Check if image has reasonable non-white area
    non_white_pixels = np.sum(image < 240)  # Adjust threshold as needed
    if non_white_pixels < 50:  # Too few signature pixels
        return False
    
    return True

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = models.load_model("model.h5")
        return model
    except:
        st.error("Model file not found. Please make sure 'model.h5' is in the current directory.")
        return None

model = load_model()

# Streamlit App
st.title("Signature Verification System")

st.sidebar.header("About")
st.sidebar.info("""
This system verifies whether two signatures are from the same person.
Upload an original signature and a test signature to check for similarity.
""")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Set Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,  # Higher default threshold
    step=0.05,
    help="Higher values require more similarity to declare signatures as matching"
)

# Upload image inputs
st.header("Upload Signatures")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Signature")
    image_file_A = st.file_uploader("Upload Original Image", type=["jpg", "png", "jpeg"], key="original")

with col2:
    st.subheader("Test Signature")
    image_file_B = st.file_uploader("Upload Test Image", type=["jpg", "png", "jpeg"], key="test")

# Process and predict when both images are uploaded
if image_file_A and image_file_B and model is not None:
    # Read images
    img_A = Image.open(image_file_A)
    img_B = Image.open(image_file_B)
    
    # Check if images are likely signatures
    if not is_likely_signature(np.array(img_A)):
        st.warning("The first image may not contain a clear signature. Please upload a valid signature image.")
    
    if not is_likely_signature(np.array(img_B)):
        st.warning("The second image may not contain a clear signature. Please upload a valid signature image.")
    
    # Display original images
    st.subheader("Uploaded Images")
    display_col1, display_col2 = st.columns(2)
    
    with display_col1:
        st.image(img_A, caption="Original Signature", use_container_width=True)
    
    with display_col2:
        st.image(img_B, caption="Test Signature", use_container_width=True)
    
    # Preprocess images
    try:
        preprocessed_img_A = preprocess_image(img_A)
        preprocessed_img_B = preprocess_image(img_B)
        
        # Display preprocessed images
        st.subheader("Preprocessed Images")
        prep_col1, prep_col2 = st.columns(2)
        
        with prep_col1:
            st.image(preprocessed_img_A[0, :, :, 0], caption="Preprocessed Original", use_container_width=True, clamp=True)
        
        with prep_col2:
            st.image(preprocessed_img_B[0, :, :, 0], caption="Preprocessed Test", use_container_width=True, clamp=True)
        
        # Predict similarity score with progress bar
        with st.spinner("Analyzing signatures..."):
            similarity_score = model.predict([preprocessed_img_A, preprocessed_img_B], verbose=0).flatten()[0]
        
        # Display results
        st.subheader("Verification Results")
        
        # Create a confidence meter
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Display confidence score with color
            if similarity_score > confidence_threshold + 0.1:
                score_color = "green"
            elif similarity_score > confidence_threshold:
                score_color = "orange"
            else:
                score_color = "red"
                
            st.metric("Similarity Score", f"{similarity_score:.4f}", 
                     delta="MATCH" if similarity_score > confidence_threshold else "NO MATCH",
                     delta_color="normal")
            
            # Confidence bar
            st.progress(float(similarity_score))
            
            # Result message
            if similarity_score > confidence_threshold:
                st.success("✅ Signatures appear to be from the same person!")
            else:
                st.error("❌ Signatures do not appear to match.")
                
            st.info(f"Threshold: {confidence_threshold:.2f}")
            
            # Additional info based on score
            if similarity_score > 0.8:
                st.info("High confidence in match")
            elif similarity_score > confidence_threshold:
                st.info("Moderate confidence in match")
            elif similarity_score > 0.3:
                st.warning("Low confidence - possible forgery or poor quality")
            else:
                st.warning("Very low confidence - likely different persons")
                
    except Exception as e:
        st.error(f"Error processing images: {str(e)}")

elif model is None:
    st.error("Model could not be loaded. Please check if the model file exists.")

# Add instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload a clear image of a known original signature
2. Upload the signature you want to verify
3. Adjust the confidence threshold if needed
4. View the similarity score and verification result

**Tips for best results:**
- Use images with white backgrounds
- Ensure signatures are clearly visible
- Avoid blurry or low-contrast images
- For best accuracy, use similar image quality for both signatures
""")