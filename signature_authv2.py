import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from network_signature_authv2 import SigNet, load_model # Import your Siamese network

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
    return transform(img).unsqueeze(0)  # add batch dimension

def compare_signatures_snn(model, imgA, imgB, device ,method="cosine"):
    """Run Siamese model and return similarity score [0-1]."""
    model.eval()

    with torch.no_grad():
        tensorA = preprocess_image(imgA).to(device)
        tensorB = preprocess_image(imgB).to(device)

        # Forward pass → embeddings
        embA, embB = model(tensorA, tensorB)

        if method == "euclidean":
            # Euclidean distance → similarity [0,1]
            dist = torch.norm(embA - embB, p=2).item()
            similarity = 1 / (1 + dist)
        else:
            # Cosine similarity → [-1,1], map to [0,1]
            cos = torch.nn.CosineSimilarity(dim=1)
            similarity = cos(embA, embB).item()
            similarity = (similarity + 1) / 2  

    return similarity

def crop_detected_signature(results, image):
    """Crop all signatures detected by YOLO."""
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
snn_model_path = "runs/detect/train/weights/convnet_best_loss.pt"
snn_model = SigNet()
snn_model.load_model(snn_model_path)

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
        st.subheader("Detected Signatures from Document")

                # Similarity method selection
        similarity_method = st.sidebar.radio(
            "Select similarity method", 
            ["cosine", "euclidean"], 
            index=0
        )

        # Slider for threshold
        threshold = st.slider("Set similarity threshold", 0.0, 1.0, 0.5, 0.01)

        # Show YOLO annotated output
        st.image(results[0].plot(), caption="YOLO Detection Results", channels="BGR")

        # Compare all detected signatures
        for i, cropped_sig in enumerate(cropped_sigs):
            similarity_score = compare_signatures_snn(
                                    snn_model, cropped_sig, authorised_sig_cv, device, method=similarity_method)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader(f"Signature {i+1} (Cropped)")
                st.image(cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2RGB), channels="RGB")
            with col2:
                st.subheader("Authorised Signature")
                st.image(authorised_sig, channels="RGB")
            with col3:
                st.subheader("Result")
                st.write(f"**Similarity Score:** {similarity_score:.4f}")
                if similarity_score > threshold:
                    st.success("✅ Signature Matched!")
                else:
                    st.error("❌ Signature Mismatch!")

            st.markdown("---")

    else:
        st.error("No signature detected in the document.")

    os.remove(tmp_doc_path)