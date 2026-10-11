"""
mediapipe_detector.py
MediaPipe Tasks ObjectDetector wrapper with the same interface as
object_detection.ObjectDetector, so the Streamlit app can switch between them.

Models are downloaded automatically on first use into ./models/
(all are COCO-trained, ~80 classes - MediaPipe has no "detect everything" model).
"""
from __future__ import annotations

import os
import urllib.request
import zlib
from typing import List

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from object_detection import Detection

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

_BASE = "https://storage.googleapis.com/mediapipe-models/object_detector"
MEDIAPIPE_MODELS = {
    "EfficientDet-Lite2 (best accuracy, 448px)": (
        "efficientdet_lite2.tflite", f"{_BASE}/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite"),
    "EfficientDet-Lite0 (balanced, 320px)": (
        "mp_efficientdet_lite0.tflite", f"{_BASE}/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"),
    "SSD MobileNetV2 (fastest, 256px)": (
        "ssd_mobilenet_v2.tflite", f"{_BASE}/ssd_mobilenet_v2/float32/latest/ssd_mobilenet_v2.tflite"),
}


def _ensure_model(filename: str, url: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # network / URL problem
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(
                f"Could not download the model.\n  URL: {url}\n  Error: {e}\n"
                f"Download it manually (see https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector) "
                f"and save it as: {path}"
            ) from e
    return path


class MediaPipeDetector:
    def __init__(self, model_name: str = "EfficientDet-Lite2 (best accuracy, 448px)"):
        filename, url = MEDIAPIPE_MODELS[model_name]
        model_path = _ensure_model(filename, url)
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=0.05,   # filter later, so the UI slider works without reloading
            max_results=100,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, image_rgb: np.ndarray, score_threshold: float = 0.3,
               iou_threshold: float = 0.5, max_detections: int = 25) -> List[Detection]:
        """iou_threshold is ignored: MediaPipe applies its own NMS internally."""
        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(image_rgb))
        result = self.detector.detect(mp_image)

        out: List[Detection] = []
        for det in result.detections:
            cat = det.categories[0]
            if cat.score < score_threshold:
                continue
            bb = det.bounding_box
            x1, y1 = max(int(bb.origin_x), 0), max(int(bb.origin_y), 0)
            x2, y2 = min(int(bb.origin_x + bb.width), w - 1), min(int(bb.origin_y + bb.height), h - 1)
            label = cat.category_name or (str(cat.index) if cat.index is not None else "object")
            # cat.index can be None depending on the model's metadata; use a stable
            # value derived from the label so box colors stay consistent per class.
            class_id = int(cat.index) if cat.index is not None else zlib.crc32(label.encode()) % 1000
            out.append(Detection(label=label, class_id=class_id, score=float(cat.score),
                                 box=(x1, y1, x2, y2)))
        out.sort(key=lambda d: d.score, reverse=True)
        return out[:max_detections]