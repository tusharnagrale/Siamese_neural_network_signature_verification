import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image

import torch
from network import snn  # Make sure network.py is in the same directory or PYTHONPATH

# =====================
# Helper Functions
# =====================
def compare_images_ssim(imageA, imageB):
    # Both images should be single-channel (grayscale or binary)
    if len(imageA.shape) == 3:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    if len(imageB.shape) == 3:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    height = min(imageA.shape[0], imageB.shape[0])
    width = min(imageA.shape[1], imageB.shape[1])
    imageA = cv2.resize(imageA, (width, height))
    imageB = cv2.resize(imageB, (width, height))
    score, _ = ssim(imageA, imageB, full=True)
    return score

def hash_difference(imageA, imageB):
    # Both images should be single-channel (grayscale or binary)
    if len(imageA.shape) == 2:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2BGR)
    if len(imageB.shape) == 2:
        imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
    hashA = imagehash.average_hash(Image.fromarray(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)))
    hashB = imagehash.average_hash(Image.fromarray(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)))
    return abs(hashA - hashB)

def crop_detected_signature(results, image):
    cropped_sigs = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_sig = image[y1:y2, x1:x2]
            cropped_sigs.append(cropped_sig)
    return cropped_sigs

def preprocess_for_snn(img):
    # img should be single-channel (binary)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # channel
    img = np.expand_dims(img, axis=0)  # batch
    return torch.tensor(img)

# =====================
# Load Models
# =====================
st.set_page_config(page_title="YOLO Signature Authentication", layout="wide")
st.title("🖊 YOLO-based Signature Detection & Authentication")

# Load YOLO model
model_path = "detection_model.pt"  # Change to your YOLO model path

model = YOLO(model_path)

# Load Siamese model
snn_model = snn()
snn_model.load_state_dict(torch.load("model_last.pth", map_location=torch.device('cpu')))
snn_model.eval()

# =====================
# Streamlit UI
# =====================
uploaded_doc = st.file_uploader("Upload Document with Signature", type=["jpg", "jpeg", "png"])
uploaded_auth = st.file_uploader("Upload Authorised Signature", type=["jpg", "jpeg", "png"])

if uploaded_doc and uploaded_auth:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_doc:
        tmp_doc.write(uploaded_doc.read())
        tmp_doc_path = tmp_doc.name

    # Read authorised signature
    authorised_sig = np.array(Image.open(uploaded_auth).convert("RGB"))
    authorised_sig_cv = cv2.cvtColor(authorised_sig, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(tmp_doc_path)

    # Read document image
    doc_img = cv2.imread(tmp_doc_path)

    # Crop signatures from document
    cropped_sigs = crop_detected_signature(results, doc_img)

    if cropped_sigs:
        cropped_sig = cropped_sigs[0]  # Taking first detected signature

        # Convert both signatures to grayscale and then to black & white (binary)
        cropped_gray = cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2GRAY)
        _, cropped_bw = cv2.threshold(cropped_gray, 127, 255, cv2.THRESH_BINARY)

        auth_gray = cv2.cvtColor(authorised_sig_cv, cv2.COLOR_BGR2GRAY)
        _, auth_bw = cv2.threshold(auth_gray, 127, 255, cv2.THRESH_BINARY)

        # Compare SSIM & Hash using black & white images
        ssim_score = compare_images_ssim(cropped_bw, auth_bw)
        hash_diff = hash_difference(cropped_bw, auth_bw)

        # SNN Model Prediction (use black & white images)
        sig1 = preprocess_for_snn(cropped_bw)
        sig2 = preprocess_for_snn(auth_bw)
        with torch.no_grad():
            output = snn_model(sig1, sig2)
            print(output)
            similarity = torch.sigmoid(output).item()  # Output between 0 and 1

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("YOLO Detected Signature")
            annotated_img = results[0].plot()
            st.image(annotated_img, channels="BGR")
        with col2:
            st.subheader("Authorised Signature (B&W)")
            st.image(auth_bw, channels="GRAY")
        with col3:
            st.subheader("Cropped Signature (B&W)")
            st.image(cropped_bw, channels="GRAY")

        st.markdown("---")
        st.subheader("🔍 Authentication Results")
        st.write(f"**SSIM Similarity:** {ssim_score:.4f}")
        st.write(f"**Hash Difference:** {hash_diff}")
        st.write(f"**SNN Model Similarity:** {similarity:.4f}")

        if similarity > 0.5:  # Adjust threshold as needed
            st.success("✅ Signature Matched (SNN)!")
        else:
            st.error("❌ Signature Mismatch (SNN)!")
    else:
        st.error("No signature detected in the document.")

    os.remove(tmp_doc_path)