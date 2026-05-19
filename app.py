import cv2
import yt_dlp
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from pose_module import PoseDetector

app = Flask(__name__)

current_config = {
    "mode": "youtube",
    "url": None,
    "start": 0,
    "end": 0,
    "is_running": False,
    "exercise": "pushup"
}

detector = PoseDetector()

def get_youtube_stream(url):
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except:
        return None

def generate_frames():
    cap = None
    try:
        # 1. SETUP SOURCE
        if current_config["mode"] == "youtube":
            if not current_config["url"]:
                # YouTube mode but no URL: show black frame
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "PASTE YOUTUBE URL TO BEGIN", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                return 

            src = get_youtube_stream(current_config["url"])
            cap = cv2.VideoCapture(src if src else 0)
        else:
            # Native Backend Webcam mode!
            # Try camera 0 first. If it fails, you can change this 0 to a 1 or 2.
            cap = cv2.VideoCapture(0)
            time.sleep(0.5) # Give the camera a half second to warm up

        if not cap or not cap.isOpened():
            print("🚨 ERROR: Python OpenCV could not open the camera! Check OS permissions or change the camera index.")
            return

        last_frame = None
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_config["mode"] == "webcam":
                frame = cv2.flip(frame, 1) # Mirror the webcam naturally

            last_frame = frame.copy()

            if current_config.get("is_running", False):
                frame = detector.process_frame(frame)
            else:
                # Dim the screen if paused
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
                cv2.putText(frame, "READY TO START", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        if cap:
            cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    title, tip, severity = detector.feedback_detail
    return jsonify({
        'reps':     detector.count,
        'state':    detector.feedback,
        'form':     title,
        'tip':      tip,
        'severity': severity,
        'exercise': current_config['exercise']
    })

@app.route('/set_exercise', methods=['POST'])
def set_exercise():
    data = request.json
    exercise = data.get('exercise', 'pushup')
    current_config['exercise'] = exercise
    detector.set_exercise(exercise)
    return jsonify({"status": "ok", "exercise": exercise})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    current_config["mode"] = data.get("mode")
    if "url" in data:
        current_config["url"] = data.get("url")
    current_config["is_running"] = False
    return jsonify({"status": "ok"})

@app.route('/toggle_exercise', methods=['POST'])
def toggle_exercise():
    data = request.json
    action = data.get("action")
    if action == "start":
        current_config["is_running"] = True
        detector.reset()
    elif action == "stop":
        current_config["is_running"] = False
    return jsonify({"status": "ok", "is_running": current_config["is_running"]})

if __name__ == "__main__":
    app.run(debug=True)