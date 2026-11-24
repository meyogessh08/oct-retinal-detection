"""Streamlit app: upload an OCT image -> predicted class, confidence, Grad-CAM heatmap.

Run locally (CPU is fine for inference):
    streamlit run app.py
"""
import glob
import os

import numpy as np
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src import config
from src.gradcam import GradCAM, overlay_heatmap
from src.models import get_gradcam_target_layer, load_checkpoint
from src.preprocessing import preprocess_pil_image

MAX_FILE_MB = 10
MIN_SIDE_PX = 100
ALLOWED_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(page_title="Retinal OCT Classifier", page_icon="👁️", layout="wide")


@st.cache_resource
def load_model(path: str):
    model, model_name, class_names = load_checkpoint(path, device="cpu")
    return model, model_name, class_names


def validate_image(uploaded) -> Image.Image:
    """Basic input checks. Raises ValueError with a user-friendly message."""
    if uploaded.size > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_FILE_MB} MB.")
    try:
        image = Image.open(uploaded)
        image.load()
    except (UnidentifiedImageError, OSError):
        raise ValueError("This file could not be read as an image.")
    w, h = image.size
    if min(w, h) < MIN_SIDE_PX:
        raise ValueError(f"Image is too small ({w}x{h}px).")
    if not 0.5 <= w / h <= 4:
        raise ValueError("Unusual aspect ratio - OCT B-scans are roughly square or wide.")
    # OCT B-scans are grayscale: colour photos are almost certainly not OCT scans.
    rgb = np.asarray(image.convert("RGB")).astype(int)
    if np.abs(rgb[..., 0] - rgb[..., 1]).mean() > 12 or np.abs(rgb[..., 1] - rgb[..., 2]).mean() > 12:
        raise ValueError("This looks like a colour image, not a grayscale OCT scan.")
    return image.convert("RGB")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
checkpoints = sorted(glob.glob(str(config.MODEL_DIR / "*.pt")))
if not checkpoints:
    st.error("No trained model found in outputs/models/. Train one first (see README).")
    st.stop()
default_idx = next((i for i, p in enumerate(checkpoints) if "best" in os.path.basename(p)), 0)
ckpt_path = st.sidebar.selectbox("Model checkpoint", checkpoints, index=default_idx,
                                 format_func=os.path.basename)
show_cam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)

model, model_name, class_names = load_model(ckpt_path)

# ---------------------------------------------------------------- main page
st.title("Retinal Disease Detection from OCT Images")
st.warning("**Educational prototype - not a clinical diagnostic tool.** "
           "Never use these predictions for medical decisions.")

uploaded = st.file_uploader("Upload an OCT image", type=ALLOWED_TYPES)

if uploaded is not None:
    try:
        image = validate_image(uploaded)
    except ValueError as err:
        st.error(f"Invalid input: {err}")
        st.stop()

    tensor = preprocess_pil_image(image, pretrained=model_name != "custom_cnn")
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred = int(probs.argmax())

    col1, col2, col3 = st.columns(3)
    col1.subheader("Uploaded image")
    col1.image(image, use_container_width=True)

    col2.subheader("Prediction")
    col2.metric("Predicted class", class_names[pred])
    col2.metric("Confidence", f"{probs[pred] * 100:.1f}%")
    col2.bar_chart({n: float(p) for n, p in zip(class_names, probs)})
    col2.caption("Confidence is a softmax score, not a guarantee of correctness.")

    if show_cam:
        cam_engine = GradCAM(model, get_gradcam_target_layer(model, model_name))
        cam, _, _ = cam_engine(tensor, class_idx=pred)
        col3.subheader("Grad-CAM")
        col3.image(overlay_heatmap(image, cam), use_container_width=True)
        col3.caption("Red = regions that most influenced the prediction.")
