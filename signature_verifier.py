# signature_authv3.py

import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
from keras.saving import register_keras_serializable
import json

# =====================
# Custom Layers (needed for loading)
# =====================
@register_keras_serializable(package="Custom", name="L2Normalization")
class L2Normalization(keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1)

@register_keras_serializable(package="Custom", name="euclidean_distance")
def euclidean_distance(tensors):
    a, b = tensors
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True) + 1e-12)

# =====================
# Preprocessing Function
# =====================
def preprocess_signature(img):
    """Preprocess signature (cropped BGR/Gray) to 128x128 for SNN model."""
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    if np.mean(thr) > 127:
        thr = 255 - thr
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    thr = cv2.resize(thr, (128, 128))
    thr = thr.astype(np.float32) / 255.0
    thr = np.expand_dims(thr, axis=-1)
    return np.expand_dims(thr, axis=0)  # shape: (1,128,128,1)

# =====================
# YOLO Helper
# =====================
def crop_detected_signature(results, image):
    """Crop all signatures detected by YOLO."""
    cropped_sigs = []
    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                cropped_sigs.append(image[y1:y2, x1:x2])
    return cropped_sigs

# =====================
# Load Models
# =====================
@st.cache_resource(show_spinner=False)
def load_yolo(path):
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_snn():
    try:
        model = keras.models.load_model("artifacts/siamese_best.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"SNN model not loaded: {e}")
        return None

def load_threshold():
    try:
        with open("artifacts/optimal_threshold.json", "r") as f:
            return json.load(f)["threshold"]
    except:
        return 0.5

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="YOLO + SNN Signature Authentication", layout="wide")
st.title("🖊 Signature Detection & Verification (YOLO + Siamese NN)")

# Load models
DETECTION_WEIGHTS = "my_model/my_model.pt"
yolo_model = load_yolo(DETECTION_WEIGHTS)
snn_model = load_snn()
confidence_threshold = load_threshold()

# Uploads
uploaded_doc = st.file_uploader("Upload Document with Signature", type=["jpg", "jpeg", "png"])
uploaded_auth = st.file_uploader("Upload Authorised Signature", type=["jpg", "jpeg", "png"])

if uploaded_doc and uploaded_auth and snn_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_doc:
        tmp_doc.write(uploaded_doc.read())
        tmp_doc_path = tmp_doc.name

    try:
        # Read authorised signature
        authorised_sig = np.array(Image.open(uploaded_auth).convert("RGB"))
        authorised_sig_cv = cv2.cvtColor(authorised_sig, cv2.COLOR_RGB2BGR)
        proc_auth = preprocess_signature(authorised_sig_cv)

        # Run YOLO detection
        results = yolo_model(tmp_doc_path)

        # Read document image
        doc_img = cv2.imread(tmp_doc_path)

        # Crop signatures from document
        cropped_sigs = crop_detected_signature(results, doc_img)

        if cropped_sigs:
            st.subheader("Detected Signatures from Document")

            # Show YOLO annotated output
            try:
                st.image(results[0].plot(), caption="YOLO Detection Results", channels="BGR")
            except Exception:
                st.info("Could not render annotated YOLO image preview.")

            # Threshold slider
            user_threshold = st.slider(
                "Adjust Distance Threshold",
                min_value=0.1,
                max_value=2.0,
                value=float(confidence_threshold),
                step=0.05
            )

            # Compare detected signatures with authorized
            for i, cropped_sig in enumerate(cropped_sigs):
                proc_doc = preprocess_signature(cropped_sig)

                if proc_doc is None or proc_auth is None:
                    st.error("Error preprocessing one of the signatures.")
                    continue

                pred = snn_model.predict([proc_doc, proc_auth])[0][0]
                is_match = pred <= user_threshold

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader(f"Cropped Signature {i+1}")
                    st.image(cv2.cvtColor(cropped_sig, cv2.COLOR_BGR2RGB), channels="RGB")
                with col2:
                    st.subheader("Authorised Signature")
                    st.image(authorised_sig, channels="RGB")
                with col3:
                    st.subheader("Result")
                    st.write(f"**Distance Score:** {pred:.4f}")
                    if is_match:
                        st.success("✅ Signature Matched!")
                    else:
                        st.error("❌ Signature Mismatch!")

                # Show preprocessed grayscale inputs
                with st.expander("🔍 See Preprocessed Inputs to SNN"):
                    st.image(
                        [proc_doc.reshape(128, 128), proc_auth.reshape(128, 128)],
                        caption=[f"Cropped Sig {i+1} (Processed)", "Authorised Sig (Processed)"],
                        width=200,
                        clamp=True,
                        channels="GRAY"
                    )

                st.markdown("---")
        else:
            st.error("No signature detected in the document.")
    finally:
        try:
            os.remove(tmp_doc_path)
        except Exception:
            pass
