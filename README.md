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
git clone [https://github.com/YOUR_USERNAME/VisionFit.git](https://github.com/YOUR_USERNAME/VisionFit.git)
cd VisionFit
```
### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

For Windows:

```Bash
# Create the environment
python -m venv .venv
.venv\Scripts\activate
```
For macOS / Linux:

```Bash
# Create the environment
python3 -m venv .venv
source .venv/bin/activate
```
You will know it worked if you see (.venv) appear at the start of your terminal line.

### 3. Install Dependencies
Install all required libraries (Flask, OpenCV, MediaPipe, etc.) using pip:

```Bash
pip install -r requirements.txt
```
### 4. How to Run

Start the Application:
Make sure your virtual environment is activated, then run:

```Bash
python app.py
```
Open the Dashboard:
You will see a message saying Running on http://127.0.0.1:5000.
Open your web browser and navigate to:
http://127.0.0.1:5000
