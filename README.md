# 🎯 Real-Time Object Detection App

A Streamlit web app that detects objects in uploaded images or live webcam snapshots using **EfficientDet-Lite** models on plain CPU, with no GPU needed.

It supports two backends you can switch between from the sidebar:

| Backend | Model | Notes |
|---|---|---|
| **Custom TFLite pipeline** | EfficientDet-Lite0 (raw export) | Anchor generation, box decoding and NMS written from scratch |
| **MediaPipe Tasks** | EfficientDet-Lite2 / Lite0, SSD MobileNetV2 | Official models with built-in post-processing |

## ✨ Features

- Upload an image **or** take a webcam snapshot
- Adjustable confidence threshold, NMS IoU threshold and max detections
- Side-by-side original vs. annotated image
- Inference time, object count and per-class summary chart
- Detections table (label, confidence, bounding box)
- Download the annotated image
- Aspect-ratio-preserving (letterbox) preprocessing for better accuracy on widescreen webcam frames

## 🧠 How the custom pipeline works

The bundled `efficientdet_lite0.tflite` is a **raw export**: it returns class scores `[1, 19206, 90]` and box offsets `[1, 19206, 4]`, with no built-in post-processing. `object_detection.py` therefore does the following itself:

1. **Letterbox** the input to 320×320 (keeps aspect ratio, gray padding)
2. **Generate anchors** for pyramid levels 3–7 (3 scales × 3 aspect ratios = 19,206 anchors)
3. **Decode** the `(ty, tx, th, tw)` offsets against the anchors into boxes
4. **Filter** by confidence and run **NMS** (OpenCV)
5. **Map** boxes back to the original image resolution

## 📁 Project structure

```
object_detection/
├── app.py                    # Streamlit UI
├── object_detection.py       # Custom TFLite detector + drawing helpers
├── mediapipe_detector.py     # MediaPipe Tasks wrapper (same interface)
├── requirements.txt
├── efficientdet_lite0.tflite # raw EfficientDet-Lite0 model
├── labels.txt                # 90 COCO labels (extracted from the model)
└── models/                   # MediaPipe models (auto-downloaded on first use)
```

## 🚀 Getting started

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

You can also run the custom detector from the command line:

```bash
python object_detection.py path/to/image.jpg   # saves output.jpg
```

### MediaPipe models

The first time you select a MediaPipe model, it is downloaded into `models/`. If the automatic download fails, get the model from the
[MediaPipe Object Detector page](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector)
and save it with the filename shown in the error message.

## ⚠️ Limitations

- All models are trained on **COCO** and only recognise those classes (person, bottle, cup, spoon, backpack, handbag, cell phone, laptop, book, scissors, …). Objects outside that list, such as **pens**, are not detected, and the model may label them as the closest class it knows.
- Adding a name to `labels.txt` does **not** teach the model a new object. The file only maps output indices to names.
- Small, partly hidden or very close objects can be missed, especially with the Lite0 models.

## 🔭 Roadmap

- [ ] Open-vocabulary detection (type a prompt like "pen" or "metal bottle") with OWLv2 / YOLO-World
- [ ] Fine-tune EfficientDet-Lite0 on custom objects with MediaPipe Model Maker
- [ ] Live video stream instead of snapshots
- [ ] Docker image and Streamlit Cloud deployment

## 🛠️ Tech stack

Python · Streamlit · TensorFlow Lite (ai-edge-litert) · MediaPipe · OpenCV · NumPy · Pandas

## 📄 License

MIT. Model licenses: EfficientDet and MediaPipe models are released under Apache 2.0 by Google.
