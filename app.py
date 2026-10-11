"""
app.py - Streamlit UI for EfficientDet-Lite0 object detection.
Run:  streamlit run app.py
"""
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from object_detection import ObjectDetector, draw_detections
from mediapipe_detector import MEDIAPIPE_MODELS, MediaPipeDetector

st.set_page_config(page_title="Object Detection - EfficientDet-Lite0", page_icon="🎯", layout="wide")


LOCAL_MODEL = "EfficientDet-Lite0 (local .tflite file)"


@st.cache_resource(show_spinner="Loading model (first MediaPipe run downloads it)...")
def load_detector(model_name: str):
    if model_name == LOCAL_MODEL:
        return ObjectDetector()
    return MediaPipeDetector(model_name)


def run_and_show(image_rgb: np.ndarray, score_thr: float, iou_thr: float, max_det: int):
    detector = load_detector(model_name)
    t0 = time.perf_counter()
    detections = detector.detect(image_rgb, score_thr, iou_thr, max_det)
    latency_ms = (time.perf_counter() - t0) * 1000
    annotated = draw_detections(image_rgb, detections)
    return detections, annotated, latency_ms


# ------------------------------- sidebar -------------------------------- #
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model", list(MEDIAPIPE_MODELS) + [LOCAL_MODEL])
score_thr = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
iou_thr = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, 0.50, 0.05)
max_det = st.sidebar.slider("Max detections", 1, 100, 25)
mode = st.sidebar.radio("Input source", ["Upload image", "Webcam snapshot"])
with st.sidebar.expander("Tips for better detection"):
    st.markdown(
        "- All models here know only the **COCO classes** (e.g. cell phone, bottle, cup, "
        "laptop, book, scissors, mouse). Pens, wallets, etc. are not included. "
        "Pick **EfficientDet-Lite2** for the best accuracy.\n"
        "- Hold the object **at arm's length**, fully visible, not covering your face.\n"
        "- Use good lighting and a plain background.\n"
        "- Lower the confidence threshold to ~0.25 to see weaker detections."
    )

st.title("🎯 Object Detection")
st.caption(f"{model_name} - COCO-trained, runs on CPU")

# -------------------------------- input --------------------------------- #
image_rgb = None
if mode == "Upload image":
    file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if file is not None:
        image_rgb = np.array(Image.open(file).convert("RGB"))
else:
    snap = st.camera_input("Take a picture")
    if snap is not None:
        image_rgb = np.array(Image.open(snap).convert("RGB"))

# -------------------------------- output -------------------------------- #
if image_rgb is None:
    st.info("Provide an image to start detecting.")
else:
    detections, annotated, latency_ms = run_and_show(image_rgb, score_thr, iou_thr, max_det)

    col1, col2 = st.columns(2)
    col1.subheader("Original")
    col1.image(image_rgb, use_container_width=True)
    col2.subheader("Detections")
    col2.image(annotated, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Objects found", len(detections))
    m2.metric("Inference time", f"{latency_ms:.0f} ms")
    m3.metric("Image size", f"{image_rgb.shape[1]}x{image_rgb.shape[0]}")

    if detections:
        counts = Counter(d.label for d in detections)
        st.subheader("Summary")
        st.bar_chart(pd.Series(counts, name="count"))

        st.subheader("Details")
        st.dataframe(
            pd.DataFrame(
                [{"label": d.label, "confidence": round(d.score, 3),
                  "x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3]}
                 for d in detections]
            ),
            use_container_width=True,
            hide_index=True,
        )

        ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button("Download annotated image", buf.tobytes(),
                               file_name="detections.png", mime="image/png")
    else:
        st.warning("No objects above the confidence threshold. Try lowering it in the sidebar.")