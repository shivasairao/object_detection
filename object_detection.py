"""
object_detection.py
EfficientDet-Lite0 (TFLite) detector with raw-output decoding.

This model variant has NO built-in post-processing. It returns:
  - class scores  [1, 19206, 90]  (already sigmoid-activated)
  - box offsets   [1, 19206, 4]   (ty, tx, th, tw relative to anchors)
so we generate the anchors, decode the boxes and run NMS ourselves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

try:  # lightweight runtime (preferred)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # fallback: full TensorFlow
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "efficientdet_lite0.tflite")
DEFAULT_LABELS = os.path.join(HERE, "labels.txt")


@dataclass
class Detection:
    label: str
    class_id: int
    score: float
    box: tuple  # (x1, y1, x2, y2) in original-image pixels


def _generate_anchors(image_size: int = 320, min_level: int = 3, max_level: int = 7,
                      num_scales: int = 3, anchor_scale: float = 3.0,
                      aspect_ratios=((1.0, 1.0), (1.4, 0.7), (0.7, 1.4))) -> np.ndarray:
    """Return anchors as [N, 4] = (y_center, x_center, height, width), in pixels."""
    all_anchors = []
    for level in range(min_level, max_level + 1):
        stride = 2 ** level
        feat = int(np.ceil(image_size / stride))  # feature-map size at this level
        centers = stride / 2.0 + stride * np.arange(feat)
        xv, yv = np.meshgrid(centers, centers)
        xv, yv = xv.reshape(-1, 1), yv.reshape(-1, 1)
        per_loc = []
        for octave in range(num_scales):
            scale = 2 ** (octave / num_scales)
            for ar_x, ar_y in aspect_ratios:
                size = anchor_scale * stride * scale
                w = np.full_like(xv, size * ar_x)
                h = np.full_like(yv, size * ar_y)
                per_loc.append(np.concatenate([yv, xv, h, w], axis=1))
        # [locations, anchors_per_location, 4] -> [locations * anchors_per_location, 4]
        all_anchors.append(np.stack(per_loc, axis=1).reshape(-1, 4))
    return np.concatenate(all_anchors, axis=0).astype(np.float32)


class ObjectDetector:
    def __init__(self, model_path: str = DEFAULT_MODEL, labels_path: str = DEFAULT_LABELS,
                 num_threads: int = 4):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.outputs = self.interpreter.get_output_details()
        _, self.in_h, self.in_w, _ = self.input["shape"]

        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [l.strip() for l in f if l.strip()]

        self.anchors = _generate_anchors(int(self.in_h))

        # identify outputs by their last dimension (90 classes vs 4 box coords)
        self._scores_idx = next(o["index"] for o in self.outputs if o["shape"][-1] != 4)
        self._boxes_idx = next(o["index"] for o in self.outputs if o["shape"][-1] == 4)

    # ------------------------------------------------------------------ #
    def detect(self, image_rgb: np.ndarray, score_threshold: float = 0.4,
               iou_threshold: float = 0.5, max_detections: int = 25) -> List[Detection]:
        """image_rgb: HxWx3 uint8 RGB array. Returns detections in image pixel coords."""
        orig_h, orig_w = image_rgb.shape[:2]

        # Aspect-preserving resize + gray padding (letterbox). Stretching a wide
        # webcam frame into a square distorts objects and hurts accuracy.
        scale = min(self.in_w / orig_w, self.in_h / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        canvas = np.full((int(self.in_h), int(self.in_w), 3), 127, dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(image_rgb, (new_w, new_h))
        input_tensor = np.expand_dims(canvas, 0).astype(self.input["dtype"])

        self.interpreter.set_tensor(self.input["index"], input_tensor)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self._scores_idx)[0]  # [N, C]
        deltas = self.interpreter.get_tensor(self._boxes_idx)[0]   # [N, 4]

        # best class per anchor, then threshold
        class_ids = scores.argmax(axis=1)
        best_scores = scores.max(axis=1)
        keep = best_scores >= score_threshold
        if not keep.any():
            return []

        deltas, anchors = deltas[keep], self.anchors[keep]
        class_ids, best_scores = class_ids[keep], best_scores[keep]

        # decode: (ty, tx, th, tw)
        ya, xa, ha, wa = anchors.T
        ty, tx, th, tw = deltas.T
        yc = ty * ha + ya
        xc = tx * wa + xa
        h = np.exp(th) * ha
        w = np.exp(tw) * wa

        # corners in model-input pixels -> original image pixels (undo letterbox)
        x1 = np.clip((xc - w / 2) / scale, 0, orig_w - 1)
        y1 = np.clip((yc - h / 2) / scale, 0, orig_h - 1)
        x2 = np.clip((xc + w / 2) / scale, 0, orig_w - 1)
        y2 = np.clip((yc + h / 2) / scale, 0, orig_h - 1)

        # class-aware NMS
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, best_scores.tolist(), score_threshold, iou_threshold)
        idxs = np.array(idxs).reshape(-1)[:max_detections] if len(idxs) else []

        results = []
        for i in idxs:
            cid = int(class_ids[i])
            results.append(Detection(
                label=self.labels[cid] if cid < len(self.labels) else str(cid),
                class_id=cid,
                score=float(best_scores[i]),
                box=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
            ))
        return results


# ---------------------------------------------------------------------- #
def draw_detections(image_rgb: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Return a copy of the image with boxes and labels drawn."""
    out = image_rgb.copy()
    thickness = max(2, int(round(min(out.shape[:2]) / 250)))
    font_scale = max(0.5, min(out.shape[:2]) / 900)
    for d in detections:
        rng = np.random.RandomState(d.class_id * 7 + 3)
        color = tuple(int(c) for c in rng.randint(60, 255, 3))
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{d.label} {d.score:.0%}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(out, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python object_detection.py <image_path>\n"
                 "Or launch the web UI with: streamlit run app.py")
    path = sys.argv[1]
    bgr = cv2.imread(path)
    if bgr is None:
        sys.exit(f"Could not read image: {path!r} (check the path and file name)")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    det = ObjectDetector()
    res = det.detect(img)
    for r in res:
        print(f"{r.label:15s} {r.score:.2f} {r.box}")
    if not res:
        print("No objects detected.")
    cv2.imwrite("output.jpg", cv2.cvtColor(draw_detections(img, res), cv2.COLOR_RGB2BGR))
    print("Saved output.jpg")