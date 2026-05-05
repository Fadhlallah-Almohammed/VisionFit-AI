import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import subprocess
import time


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Stats
        self.count = 0
        self.direction = 0
        self.max_low_angle = 180
        self.feedback = "Ready"
        self.form_feedback = "Good Form"

        # This becomes True only when the warning actually changes
        self.feedback_changed = False
        self.feedback_to_display = "Good Form"

        # Stable feedback control
        self.pending_feedback = "Good Form"
        self.pending_feedback_frames = 0
        self.feedback_stability_frames = 8

        # Buffers
        self.arm_buffer = deque(maxlen=5)
        self.body_buffer = deque(maxlen=5)
        self.flare_buffer = deque(maxlen=5)

        # Voice feedback settings
        self.voice_messages = {
            "Head Too Low": "Raise your head",
            "Tuck Elbows": "Keep your elbows close",
            "Lower Hips": "Lower your hips",
            "Hips Sagging": "Lift your hips",
            "Hands Under Shoulders": "Put your hands under your shoulders"
        }

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle

    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update_stable_feedback(self, new_feedback):
        # Reset change flag every frame
        self.feedback_changed = False

        # If the new feedback is already the current feedback, do nothing
        if new_feedback == self.form_feedback:
            self.pending_feedback = new_feedback
            self.pending_feedback_frames = 0
            return self.form_feedback

        # If the new feedback is different from the pending one, start counting again
        if new_feedback != self.pending_feedback:
            self.pending_feedback = new_feedback
            self.pending_feedback_frames = 1
            return self.form_feedback

        # If the same new feedback keeps appearing, increase the counter
        self.pending_feedback_frames += 1

        # Apply the new feedback only when it is stable for enough frames
        if self.pending_feedback_frames >= self.feedback_stability_frames:
            self.form_feedback = new_feedback
            self.feedback_to_display = new_feedback
            self.feedback_changed = True
            self.pending_feedback_frames = 0

        return self.form_feedback

    def speak_feedback(self, text):
        # Do not speak when the form is correct
        if text == "Good Form" or text == "Perfect Form":
            return

        # Convert feedback to a clearer voice message
        voice_text = self.voice_messages.get(text, text)

        # Use Windows built-in speech engine through PowerShell
        command = f'''
        Add-Type -AssemblyName System.Speech;
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        $speak.Rate = 0;
        $speak.Volume = 100;
        $speak.Speak("{voice_text}");
        '''

        subprocess.Popen(
            ["powershell", "-Command", command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            def get_xy(idx):
                return [landmarks[idx].x * w, landmarks[idx].y * h]

            # --- 1. GET KEYPOINTS ---
            l_shldr = get_xy(11)
            l_elbow = get_xy(13)
            l_wrist = get_xy(15)
            l_hip = get_xy(23)
            l_knee = get_xy(25)
            l_ear = get_xy(7)  # Using Ear instead of Nose

            # --- 2. CALCULATE ANGLES ---
            arm_angle = self.calculate_angle(l_shldr, l_elbow, l_wrist)
            body_angle = self.calculate_angle(l_shldr, l_hip, l_knee)
            flare_angle = self.calculate_angle(l_hip, l_shldr, l_elbow)

            # Neck Angle: Ear - Shoulder - Hip
            neck_angle = self.calculate_angle(l_ear, l_shldr, l_hip)

            # Fix Arm Angle
            if arm_angle > 180:
                arm_angle = 360 - arm_angle

            # Adaptive Metrics
            torso_length = self.get_distance(l_shldr, l_hip)
            wrist_shoulder_diff = abs(l_wrist[0] - l_shldr[0])

            # --- 3. SMOOTHING ---
            self.arm_buffer.append(arm_angle)
            self.body_buffer.append(body_angle)
            self.flare_buffer.append(flare_angle)

            s_arm = np.mean(self.arm_buffer)
            s_body = np.mean(self.body_buffer)
            s_flare = np.mean(self.flare_buffer)

            # --- 4. RELAXED LOGIC ---
            current_errors = []

            if s_body < 145:
                current_errors.append("Lower Hips")
            elif s_body > 210:
                current_errors.append("Hips Sagging")

            if s_flare > 85:
                current_errors.append("Tuck Elbows")

            if wrist_shoulder_diff > (torso_length * 0.6):
                current_errors.append("Hands Under Shoulders")

            # Threshold relaxed to 120
            if neck_angle < 120:
                current_errors.append("Head Too Low")

            # --- 5. SET STABLE FEEDBACK ---
            if not current_errors:
                raw_feedback = "Good Form"
            else:
                raw_feedback = current_errors[0]

            stable_feedback = self.update_stable_feedback(raw_feedback)

            # --- VOICE FEEDBACK ---
            # Speak only when the warning actually changes
            if self.feedback_changed:
                self.speak_feedback(stable_feedback)

            # --- 6. REP COUNTING ---
            if s_arm > 160:
                self.feedback = "Up"
                if self.direction == 1:
                    if self.max_low_angle <= 100:
                        if self.form_feedback == "Good Form":
                            self.count += 1
                    self.direction = 0
                    self.max_low_angle = 180

            if s_arm < 100:
                self.feedback = "Down"
                self.direction = 1
                if s_arm < self.max_low_angle:
                    self.max_low_angle = s_arm

            # --- 7. DRAWING ---
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

        return frame