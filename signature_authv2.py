import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from network_signature_authv2 import SigNet, load_model  # function, not a method

# =====================
# Helper Functions
# =====================
def preprocess_image(img):
    """Convert OpenCV/Numpy image to 155x220 grayscale tensor."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((155, 220)),  # match training size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(img).unsqueeze(0)  # [1,1,155,220]

def compare_signatures_snn(model, imgA, imgB, device, method="cosine"):
    """Run Siamese model and return similarity score [0-1]."""
    model.eval()
    with torch.no_grad():
        tensorA = preprocess_image(imgA).to(device)
        tensorB = preprocess_image(imgB).to(device)
        # Forward pass → embeddings (SigNet.forward returns two embeddings)
        embA, embB = model(tensorA, tensorB)

        if method == "euclidean":
            dist = torch.norm(embA - embB, p=2).item()
            similarity = 1.0 / (1.0 + dist)  # map to (0,1]
        else:
            cos = torch.nn.CosineSimilarity(dim=1)
            similarity = cos(embA, embB).item()
            similarity = (similarity + 1.0) / 2.0  # [-1,1] -> [0,1]
    return similarity

def crop_detected_signature(results, image):
    """Crop all signatures detected by YOLO."""
    cropped_sigs = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # clamp to image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                cropped_sigs.append(image[y1:y2, x1:x2])
    return cropped_sigs

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="YOLO Signature Authentication", layout="wide")
st.title("🖊 YOLO-based Signature Detection & Authentication (Siamese NN)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"Device: **{device}**")

@st.cache_resource(show_spinner=False)
def load_yolo(path):
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_signet(weights_path, device_):
    # Use the loader function from your network file (handles CPU/GPU + keys)
    return load_model(weights_path, device_)

# Paths to weights
DETECTION_WEIGHTS = "runs/detect/train/weights/detection_model.pt"
SNN_WEIGHTS = "runs/detect/train/weights/convnet_best_loss.pt"

# Load models once
yolo_model = load_yolo(DETECTION_WEIGHTS)
snn_model = load_signet(SNN_WEIGHTS, device)

# Upload files
uploaded_doc = st.file_uploader("Upload Document with Signature", type=["jpg", "jpeg", "png"])
uploaded_auth = st.file_uploader("Upload Authorised Signature", type=["jpg", "jpeg", "png"])

if uploaded_doc and uploaded_auth:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_doc:
        tmp_doc.write(uploaded_doc.read())
        tmp_doc_path = tmp_doc.name

    try:
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
            st.subheader("Detected Signatures from Document")

            # Similarity method selection
            similarity_method = st.sidebar.radio(
                "Select similarity method",
                ["cosine", "euclidean"],
                index=0
            )

            # Slider for threshold
            threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.5, 0.01)

            # Show YOLO annotated output (BGR)
            try:
                st.image(results[0].plot(), caption="YOLO Detection Results", channels="BGR")
            except Exception:
                st.info("Could not render annotated YOLO image preview.")

            # Compare all detected signatures
            for i, cropped_sig in enumerate(cropped_sigs):
                similarity_score = compare_signatures_snn(
                    snn_model, cropped_sig, authorised_sig_cv, device, method=similarity_method
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader(f"Cropped Signature {i+1}")
                    st.image(cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2RGB), channels="RGB")
                with col2:
                    st.subheader("Authorised Signature")
                    st.image(authorised_sig, channels="RGB")
                with col3:
                    st.subheader("Result")
                    st.write(f"**Similarity Score:** {similarity_score:.4f}")
                    if similarity_score >= threshold:
                        st.success("✅ Signature Matched!")
                    else:
                        st.error("❌ Signature Mismatch!")

                st.markdown("---")
        else:
            st.error("No signature detected in the document.")
    finally:
        # Always clean up the temp file
        try:
            os.remove(tmp_doc_path)
        except Exception:
            pass
