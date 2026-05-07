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

### 2. Create a Virtual Environment

Using a virtual environment keeps dependencies isolated from the rest of your system.

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

You'll know it's active when you see `(.venv)` at the start of your terminal line.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** The `requirements.txt` pins specific versions (especially `mediapipe==0.10.9`). Do **not** upgrade MediaPipe — newer versions (0.10.15+) break the pose detection API used in this project.

---

### 4. Download the Model Files

These files are too large for Git and must be downloaded separately.

#### 📥 MediaPipe Pose Landmarker (`pose_landmarker.task`)

Download from the official MediaPipe page:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Rename the downloaded file to **`pose_landmarker.task`** and place it in the **root of the project folder** (same folder as `app.py`).

You can also download it via terminal:

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" -OutFile "pose_landmarker.task"
```

**macOS / Linux:**
```bash
curl -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
```

#### 📥 YOLO Pose Model (`yolo11n-pose.pt`)

Download from Ultralytics:

```
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

Place `yolo11n-pose.pt` in the **root of the project folder**.

**Windows (PowerShell):**
```powershell
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt" -OutFile "yolo11n-pose.pt"
```

**macOS / Linux:**
```bash
curl -L -o yolo11n-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
```

---

### ✅ Verify Your Setup

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

Make sure your virtual environment is active, then run:

```bash
python app.py
```

Open your browser and go to:

```
http://127.0.0.1:5000
```

The dashboard will open and your webcam will start automatically.

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
→ You have the wrong MediaPipe version. Run: `pip install mediapipe==0.10.9`

**Webcam not opening / black screen**
→ Make sure no other app is using your webcam. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` if you have multiple cameras.

**`ModuleNotFoundError`**
→ Make sure your virtual environment is activated before running `python app.py`.