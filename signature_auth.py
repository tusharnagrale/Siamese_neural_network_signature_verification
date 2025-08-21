import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from network import snn  # Import your Siamese network

# =====================
# Helper Functions
# =====================
def preprocess_image(img):
    """Convert OpenCV/Numpy image to 32x32 grayscale tensor."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

def compare_signatures_snn(model, imgA, imgB, device):
    """Run Siamese model and return similarity score."""
    model.eval()
    with torch.no_grad():
        tensorA = preprocess_image(imgA).to(device)
        tensorB = preprocess_image(imgB).to(device)
        output = model(tensorA, tensorB)
        similarity = torch.sigmoid(output).item()
    return similarity

def crop_detected_signature(results, image):
    cropped_sigs = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_sig = image[y1:y2, x1:x2]
            cropped_sigs.append(cropped_sig)
    return cropped_sigs

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="YOLO Signature Authentication", layout="wide")
st.title("🖊 YOLO-based Signature Detection & Authentication (Siamese NN)")

# Load YOLO model
model_path = "runs/detect/train/weights/detection_model.pt"  # Your trained YOLO model
yolo_model = YOLO(model_path)

# Load Siamese model
device = "cuda" if torch.cuda.is_available() else "cpu"
snn_model = snn().to(device)
snn_model.load_state_dict(torch.load("runs/detect/train/weights/comparison_model.pth", map_location=device))

# Upload files
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
    results = yolo_model(tmp_doc_path)

    # Read document image
    doc_img = cv2.imread(tmp_doc_path)

    # Crop signatures from document
    cropped_sigs = crop_detected_signature(results, doc_img)

    if cropped_sigs:
        cropped_sig = cropped_sigs[0]  # Taking first detected signature

        # Compare using Siamese NN
        similarity_score = compare_signatures_snn(snn_model, cropped_sig, authorised_sig_cv, device)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("YOLO Detected Signature")
            annotated_img = results[0].plot()  # YOLO annotated output
            st.image(annotated_img, channels="BGR")
        with col2:
            st.subheader("Authorised Signature")
            st.image(authorised_sig, channels="RGB")
        with col3:
            st.subheader("Cropped Signature")
            st.image(cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2RGB), channels="RGB")

        st.markdown("---")
        st.subheader("🔍 Authentication Results")
        st.write(f"**Siamese Similarity Score:** {similarity_score:.4f}")

        if similarity_score > 0.5:
            st.success("✅ Signature Matched!")
        else:
            st.error("❌ Signature Mismatch!")
    else:
        st.error("No signature detected in the document.")

    os.remove(tmp_doc_path)
