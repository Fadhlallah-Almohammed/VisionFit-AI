# 🏋️‍♂️ VisionFit — AI Fitness Trainer

**VisionFit** is an AI-powered fitness dashboard that analyzes your exercise form in real time using computer vision.

It tracks body movements via your webcam (or a YouTube video), counts reps, and gives instant corrective feedback such as:
- *"Lower Hips"* — body is not aligned
- *"Tuck Elbows"* — elbows flaring out
- *"Head Too Low"* — neck out of neutral

---

## 📋 Prerequisites

Before you start, make sure you have:
- **Python 3.10 or 3.11** — [Download here](https://www.python.org/downloads/) *(3.12+ is not yet supported by MediaPipe)*
- **Git** — [Download here](https://git-scm.com/downloads)
- A **webcam** connected to your computer

---

## 🚀 Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Fadilullah-Almohammed/VisionFit-AI.git
cd VisionFit-AI
```

---

### 2. Create & Activate a Virtual Environment

A virtual environment keeps all dependencies isolated from the rest of your system. **Do this before installing anything.**

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

✅ You'll know it's working when you see `(.venv)` at the start of your terminal line.

---

### 3. Install Dependencies

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```

> ⚠️ **Do not upgrade MediaPipe.** The `requirements.txt` pins `mediapipe==0.10.9` intentionally — newer versions (0.10.15+) removed the pose detection API used in this project and will cause an `AttributeError` crash on startup.

---

### 4. Download the Model Files

These model files are too large for Git and must be downloaded separately. Place both files in the **root of the project folder** (same folder as `app.py`).

#### MediaPipe Pose Model — `pose_landmarker.task`

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "pose_landmarker.task"
```

**macOS / Linux:**
```bash
curl -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

#### YOLO Pose Model — `yolo11n-pose.pt`

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt" -OutFile "yolo11n-pose.pt"
```

**macOS / Linux:**
```bash
curl -L -o yolo11n-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
```

---

### 5. Verify Your Setup

Your project folder should look like this before running:

```
VisionFit-AI/
├── app.py
├── pose_module.py
├── pose_landmarker.task    ← downloaded in step 4
├── yolo11n-pose.pt         ← downloaded in step 4
├── requirements.txt
└── templates/
    └── index.html
```

---

## ▶️ Running the App

Make sure your virtual environment is **active** (`(.venv)` visible in terminal), then run:

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Live Webcam** | Real-time pose tracking via your webcam |
| **YouTube Mode** | Analyze form from a YouTube video URL |
| **Rep Counter** | Counts valid push-up reps automatically |
| **Form Feedback** | Flags posture errors in real time |
| **MediaPipe Backend** | Lightweight, CPU-friendly pose detection |

---

## 🛠️ Troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
→ Wrong MediaPipe version installed. Fix it with:
```bash
pip install mediapipe==0.10.9
```

**Webcam not opening / black screen**
→ Make sure no other app (e.g. Zoom, Teams) is using your webcam. If you have multiple cameras, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py`.

**`ModuleNotFoundError`**
→ Your virtual environment is not active. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux) first.

**`pip install` fails or installs wrong versions**
→ Make sure you're inside the virtual environment before running `pip install -r requirements.txt`. Check with `where python` (Windows) — it should point to your `.venv` folder.
