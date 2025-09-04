import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow import keras

# ================== IMAGE PREPROCESSING ==================
def preprocess_image(image):
    image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    image = cv2.resize(image, (64, 64)) / 255.0
    return image.reshape((1, 64, 64))


def is_likely_signature(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if np.std(image) < 25:
        return False
    if np.sum(image < 240) < 50:
        return False
    return True


def l1_distance(tensors):
    return tf.abs(tensors[0] - tensors[1])


# ================== LOAD TRAINED MODEL ==================
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("best_model.h5", custom_objects={"l1_distance": l1_distance})
        return model
    except Exception as e:
        st.error(f"Model file not found or invalid: {e}")
        return None

model = load_model()


# ================== LOAD OPTIMAL THRESHOLD ==================
def load_threshold():
    try:
        with open("optimal_threshold.json", "r") as f:
            return json.load(f)["threshold"]
    except:
        return 0.7

confidence_threshold = load_threshold()
