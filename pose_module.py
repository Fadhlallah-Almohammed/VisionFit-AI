import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
import subprocess

# ── FEEDBACK CATALOGUE ───────────────────────────────────────────────────────
FEEDBACK_CATALOGUE = {
    # Push-up
    "PU_HIPS_HIGH":   ("Hips Too High",       "Lower your hips — keep your body in a straight plank line.", "warn"),
    "PU_HIPS_SAG":    ("Hips Sagging",         "Engage your core and lift your hips to maintain a straight line.", "error"),
    "PU_ELBOW_FLARE": ("Elbows Flaring Out",   "Draw your elbows closer to your body at ~45° from your torso.", "warn"),
    "PU_HEAD_DROP":   ("Head Dropping",        "Keep your neck neutral — look slightly ahead of your hands.", "warn"),
    "PU_HAND_WIDE":   ("Hands Too Wide",       "Place hands roughly shoulder-width apart directly under shoulders.", "warn"),
    "PU_DEPTH":       ("Not Going Low Enough", "Lower your chest closer to the ground for a full rep.", "error"),
    
    # Squat
    "SQ_KNEE_CAVE":   ("Knees Caving In",      "Push your knees out over your pinky toes as you lower.", "error"),
    "SQ_FORWARD_LEAN":("Excessive Forward Lean","Keep your chest up and back straighter — sit back into your heels.", "warn"),
    "SQ_DEPTH":       ("Not Deep Enough",      "Aim for thighs parallel to the floor or deeper.", "error"),
    "SQ_HEEL_RISE":   ("Heels Rising",         "Keep feet flat on the floor — work on ankle mobility.", "warn"),
    
    # Bicep Curl
    "BC_ELBOW_DRIFT": ("Upper Arm Moving",     "Pin your elbows to your ribs. Your upper arm should stay perfectly still.", "error"),
    "BC_SHRUG":       ("Shoulders Shrugging",  "Keep your shoulders depressed and relaxed. Don't use your traps.", "warn"),
    "BC_SWING":       ("Body Swinging",        "Control the weight — avoid leaning back to use momentum.", "warn"),
    "BC_PARTIAL":     ("Partial Range",        "Fully extend your arms at the bottom for complete muscle activation.", "warn"),
    
    # High Knees (Presentation)
    "HK_LOW_KNEE":    ("Knee Too Low",         "Drive your knee up until your thigh is parallel to the floor.", "error"),
    "HK_LEANING":     ("Torso Leaning",        "Keep your chest up and back straight. Don't lean backward to lift your leg.", "warn"),
    "HK_BENT_LEG":    ("Standing Leg Bent",    "Keep your planted leg completely straight for stability.", "warn"),

    # Arm Crossovers / Cross Hands (NEW EXERCISE)
    "CH_ARMS_LOW":    ("Arms Dropping",        "Keep your arms elevated at shoulder height, parallel to the floor.", "warn"),
    "CH_NOT_CROSSED": ("Cross Arms Fully",     "Ensure your wrists overlap or get very close in front of your chest.", "error"),
    "CH_NOT_OPEN":    ("Open Arms Wider",      "Stretch your arms completely out to your sides to open your chest.", "warn"),

    # Good
    "GOOD_FORM":      ("Perfect Form ✓",       "Great technique! Keep it up.", "good"),
}

VOICE_LINES = {
    "PU_HIPS_HIGH":   "Lower your hips",
    "PU_HIPS_SAG":    "Lift your hips",
    "PU_ELBOW_FLARE": "Tuck your elbows",
    "PU_HEAD_DROP":   "Raise your head",
    "PU_HAND_WIDE":   "Move hands under shoulders",
    "PU_DEPTH":       "Go lower",
    "SQ_KNEE_CAVE":   "Push knees out",
    "SQ_FORWARD_LEAN":"Keep chest up",
    "SQ_DEPTH":       "Squat deeper",
    "SQ_HEEL_RISE":   "Keep heels down",
    "BC_ELBOW_DRIFT": "Pin your elbows",
    "BC_SHRUG":       "Shoulders down",
    "BC_SWING":       "Control the weight",
    "BC_PARTIAL":     "Full extension",
    "HK_LOW_KNEE":    "Drive knees higher",
    "HK_LEANING":     "Don't lean back",
    "HK_BENT_LEG":    "Straighten planted leg",
    "CH_ARMS_LOW":    "Keep arms up",
    "CH_NOT_CROSSED": "Cross arms fully",
    "CH_NOT_OPEN":    "Open wider",
}

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec_lm = mp.solutions.drawing_utils.DrawingSpec(
            color=(180, 120, 255), thickness=2, circle_radius=3)
        self.draw_spec_con = mp.solutions.drawing_utils.DrawingSpec(
            color=(100, 80, 200), thickness=2)

        self.exercise = "pushup"

        self.count = 0
        self.direction = 0          
        self.feedback = "Ready"     
        self.form_key = "GOOD_FORM" 
        self.angles = {}            

        self._pending_key = "GOOD_FORM"
        self._pending_frames = 0
        self._stability = 10        

        self._buf = {k: deque(maxlen=6) for k in
                     ["arm", "body", "flare", "neck",
                      "knee_l", "knee_r", "hip", "curl"]}

        self._min_angle = 180       
        self._max_angle = 0         # Added to track maximum extension/openness

    @property
    def form_feedback(self):
        return FEEDBACK_CATALOGUE[self.form_key][0]

    @property
    def feedback_detail(self):
        return FEEDBACK_CATALOGUE[self.form_key]

    def set_exercise(self, exercise):
        self.exercise = exercise
        self.reset()

    def reset(self):
        self.count = 0
        self.direction = 0
        self.feedback = "Ready"
        self.form_key = "GOOD_FORM"
        self._pending_key = "GOOD_FORM"
        self._pending_frames = 0
        self._min_angle = 180
        self._max_angle = 0
        for buf in self._buf.values():
            buf.clear()

    def process_frame(self, frame):
        dispatch = {
            "pushup": self._process_pushup,
            "squat":  self._process_squat,
            "curl":   self._process_curl,
            "highknees": self._process_highknees,
            "crosshands": self._process_crosshands, # Routing for new exercise
        }
        return dispatch.get(self.exercise, lambda f: f)(frame)

    def _get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return None, None
        lm = res.pose_landmarks.landmark
        h, w = frame.shape[:2]
        self.mp_draw.draw_landmarks(
            frame, res.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            self.draw_spec_lm, self.draw_spec_con)
        def xy(i):
            return [lm[i].x * w, lm[i].y * h]
        def vis(i):
            return lm[i].visibility
        return xy, vis

    @staticmethod
    def _angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    @staticmethod
    def _dist(p, q):
        return math.hypot(p[0]-q[0], p[1]-q[1])

    def _smooth(self, key, val):
        self._buf[key].append(val)
        return float(np.mean(self._buf[key]))

    def _gate_feedback(self, key):
        if key == self._pending_key:
            self._pending_frames += 1
        else:
            self._pending_key = key
            self._pending_frames = 1
        if self._pending_frames >= self._stability:
            if self.form_key != key:
                self.form_key = key
                self._pending_frames = 0
                self._speak(key)

    def _speak(self, key):
        if key == "GOOD_FORM":
            return
        txt = VOICE_LINES.get(key, "")
        if not txt:
            return
        cmd = (
            f'Add-Type -AssemblyName System.Speech;'
            f'$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f'$s.Rate=1;$s.Volume=100;$s.Speak("{txt}");'
        )
        subprocess.Popen(["powershell", "-Command", cmd],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _draw_angle(self, frame, pt, angle, label=""):
        x, y = int(pt[0]), int(pt[1])
        cv2.putText(frame, f"{label}{angle:.0f}",
                    (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (220, 200, 255), 1, cv2.LINE_AA)

    # ── EXISTING EXERCISES ──────────────────────────────────────────────────
    def _process_pushup(self, frame):
        xy, vis = self._get_landmarks(frame)
        if xy is None: return frame
        l_vis = min(vis(11), vis(13), vis(15))
        r_vis = min(vis(12), vis(14), vis(16))
        if r_vis > l_vis:
            shldr, elbow, wrist = xy(12), xy(14), xy(16)
            hip, knee, ankle = xy(24), xy(26), xy(28)
            ear = xy(8)
        else:
            shldr, elbow, wrist = xy(11), xy(13), xy(15)
            hip, knee, ankle = xy(23), xy(25), xy(27)
            ear = xy(7)
        arm_raw   = self._angle(shldr, elbow, wrist)
        body_raw  = self._angle(shldr, hip, ankle)
        flare_raw = self._angle(hip, shldr, elbow)
        neck_raw  = self._angle(ear, shldr, hip)
        torso     = self._dist(shldr, hip)
        hand_off  = self._dist(wrist, shldr)
        s_arm   = self._smooth("arm",   arm_raw)
        s_body  = self._smooth("body",  body_raw)
        s_flare = self._smooth("flare", flare_raw)
        s_neck  = self._smooth("neck",  neck_raw)
        self._draw_angle(frame, elbow, s_arm, "arm:")
        self._draw_angle(frame, hip,   s_body,"body:")
        errors = []
        if s_body < 155: errors.append("PU_HIPS_HIGH")
        elif s_body > 205: errors.append("PU_HIPS_SAG")
        if s_flare > 75: errors.append("PU_ELBOW_FLARE")
        if s_neck < 130: errors.append("PU_HEAD_DROP")
        if hand_off > torso * 0.95: errors.append("PU_HAND_WIDE")
        self._gate_feedback(errors[0] if errors else "GOOD_FORM")
        if s_arm > 155:
            self.feedback = "Up"
            if self.direction == 1:
                if self._min_angle < 90: self.count += 1
                self.direction = 0
                self._min_angle = 180
        if s_arm < 90:
            self.feedback = "Down"
            self.direction = 1
            if s_arm < self._min_angle: self._min_angle = s_arm
        return frame

    def _process_squat(self, frame):
        xy, vis = self._get_landmarks(frame)
        if xy is None: return frame
        l_hip, l_knee, l_ankle = xy(23), xy(25), xy(27)
        r_hip, r_knee, r_ankle = xy(24), xy(26), xy(28)
        l_shldr, r_shldr = xy(11), xy(12)
        knee_l = self._angle(l_hip, l_knee, l_ankle)
        knee_r = self._angle(r_hip, r_knee, r_ankle)
        s_knee = self._smooth("knee_l", (knee_l + knee_r) / 2)
        hip_raw = (self._angle(l_shldr, l_hip, l_knee) + self._angle(r_shldr, r_hip, r_knee)) / 2
        s_hip = self._smooth("hip", hip_raw)
        self._draw_angle(frame, l_knee, s_knee, "knee:")
        self._draw_angle(frame, l_hip,  s_hip,  "hip:")
        l_cave = (l_ankle[0] - l_knee[0])   
        r_cave = (r_knee[0] - r_ankle[0])   
        knee_cave = max(l_cave, r_cave)
        hip_w = self._dist(l_hip, r_hip)
        errors = []
        if s_knee < 160 and knee_cave > hip_w * 0.18: errors.append("SQ_KNEE_CAVE")
        if s_hip < 30 and s_knee < 130: errors.append("SQ_FORWARD_LEAN")
        if s_knee > 145 and self.direction == 1: pass  
        self._gate_feedback(errors[0] if errors else "GOOD_FORM")
        if s_knee > 160:
            self.feedback = "Up"
            if self.direction == 1:
                if self._min_angle < 105: self.count += 1
                else: self._gate_feedback("SQ_DEPTH")
                self.direction = 0
                self._min_angle = 180
        if s_knee < 105:
            self.feedback = "Down"
            self.direction = 1
            if s_knee < self._min_angle: self._min_angle = s_knee
        return frame

    def _process_curl(self, frame):
        xy, vis = self._get_landmarks(frame)
        if xy is None: return frame
        l_shldr, l_elbow, l_wrist = xy(11), xy(13), xy(15)
        r_shldr, r_elbow, r_wrist = xy(12), xy(14), xy(16)
        l_hip, r_hip = xy(23), xy(24)
        l_ear, r_ear = xy(7), xy(8)
        curl_l = self._angle(l_shldr, l_elbow, l_wrist)
        curl_r = self._angle(r_shldr, r_elbow, r_wrist)
        s_curl = self._smooth("curl", (curl_l + curl_r) / 2)
        self._draw_angle(frame, l_elbow, curl_l, "L:")
        self._draw_angle(frame, r_elbow, curl_r, "R:")
        torso_w = self._dist(l_shldr, r_shldr)
        torso_h = self._dist(l_shldr, l_hip)
        l_upper_arm_angle = self._angle(l_hip, l_shldr, l_elbow)
        r_upper_arm_angle = self._angle(r_hip, r_shldr, r_elbow)
        avg_upper_arm_drift = (l_upper_arm_angle + r_upper_arm_angle) / 2
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        shldr_mid_x = (l_shldr[0] + r_shldr[0]) / 2
        sway = abs(hip_mid_x - shldr_mid_x)
        l_neck_len = self._dist(l_ear, l_shldr)
        r_neck_len = self._dist(r_ear, r_shldr)
        avg_neck = (l_neck_len + r_neck_len) / 2
        errors = []
        if avg_upper_arm_drift > 22: errors.append("BC_ELBOW_DRIFT")
        elif avg_neck < (torso_h * 0.22): errors.append("BC_SHRUG")
        elif sway > (torso_w * 0.35): errors.append("BC_SWING")
        self._gate_feedback(errors[0] if errors else "GOOD_FORM")
        if s_curl > 150:
            self.feedback = "Down"
            if self.direction == 1:
                if self._min_angle < 65: self.count += 1
                else: self._gate_feedback("BC_PARTIAL")
                self.direction = 0
                self._min_angle = 180
        if s_curl < 55:
            self.feedback = "Up"
            self.direction = 1
            if s_curl < self._min_angle: self._min_angle = s_curl
        return frame

    def _process_highknees(self, frame):
        xy, vis = self._get_landmarks(frame)
        if xy is None: return frame
        l_shldr, r_shldr = xy(11), xy(12)
        l_hip, r_hip = xy(23), xy(24)
        l_knee, r_knee = xy(25), xy(26)
        body_angle_l = self._angle(l_shldr, l_hip, l_knee)
        body_angle_r = self._angle(r_shldr, r_hip, r_knee)
        active_leg_angle = min(body_angle_l, body_angle_r)
        standing_leg_angle = max(body_angle_l, body_angle_r)
        s_active = self._smooth("knee_l", active_leg_angle)
        if body_angle_l < body_angle_r:
            self._draw_angle(frame, l_hip, s_active, "Hip:")
        else:
            self._draw_angle(frame, r_hip, s_active, "Hip:")
        torso_h = self._dist(l_shldr, l_hip)
        shldr_mid_x = (l_shldr[0] + r_shldr[0]) / 2
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        lean_dist = abs(shldr_mid_x - hip_mid_x)
        errors = []
        if lean_dist > (torso_h * 0.25): errors.append("HK_LEANING")
        elif standing_leg_angle < 155: errors.append("HK_BENT_LEG")
        self._gate_feedback(errors[0] if errors else "GOOD_FORM")
        if body_angle_l > 155 and body_angle_r > 155:
            self.feedback = "Down"
            if self.direction == 1:
                if self._min_angle > 110 and self._min_angle != 180:
                    self._gate_feedback("HK_LOW_KNEE")
                else: self.count += 1
                self.direction = 0
                self._min_angle = 180
        if s_active < 140:
            self.feedback = "Up"
            self.direction = 1
            if s_active < self._min_angle: self._min_angle = s_active
        return frame

    # ── ARM CROSSOVERS (NEW EXERCISE) ───────────────────────────────────────
    def _process_crosshands(self, frame):
        xy, vis = self._get_landmarks(frame)
        if xy is None: return frame

        l_shldr, r_shldr = xy(11), xy(12)
        l_wrist, r_wrist = xy(15), xy(16)
        
        # Calculate Distances
        shldr_dist = self._dist(l_shldr, r_shldr)
        wrist_dist = self._dist(l_wrist, r_wrist)
        
        # We use a Ratio to determine how "open" the arms are regardless of camera distance
        # Fully open = Ratio > 2.0. Fully crossed = Ratio < 0.6
        open_ratio = wrist_dist / (shldr_dist + 1e-6)
        s_ratio = self._smooth("arm", open_ratio) 

        self._draw_angle(frame, l_shldr, s_ratio * 10, "R:") # Debug visualizer

        # Form Check 1: Are the arms dropping too low?
        avg_shldr_y = (l_shldr[1] + r_shldr[1]) / 2
        avg_wrist_y = (l_wrist[1] + r_wrist[1]) / 2
        torso_h = self._dist(l_shldr, xy(23))
        drop_dist = avg_wrist_y - avg_shldr_y # Positive means wrists are below shoulders

        errors = []
        if drop_dist > (torso_h * 0.35):
            errors.append("CH_ARMS_LOW")
            
        self._gate_feedback(errors[0] if errors else "GOOD_FORM")

        # Rep Counting Engine
        # When fully open (Ratio > 2.0)
        if s_ratio > 2.0:
            self.feedback = "Open"
            if self.direction == 1: # Was crossing, now opening
                # Form Check 2: Evaluate the cross that just finished
                if self._min_angle > 0.8: 
                    self._gate_feedback("CH_NOT_CROSSED")
                else:
                    self.count += 1
                
                self.direction = 0
                self._min_angle = 999 # Reset min tracker
                
            # Track how wide they open
            if s_ratio > self._max_angle:
                self._max_angle = s_ratio

        # When crossing inward (Ratio < 1.0)
        if s_ratio < 1.0:
            self.feedback = "Cross"
            if self.direction == 0: # Was opening, now crossing
                # Form Check 3: Evaluate the opening that just finished
                if self._max_angle < 1.8:
                    self._gate_feedback("CH_NOT_OPEN")
                    
                self.direction = 1
                self._max_angle = 0 # Reset max tracker

            # Track how tightly they cross
            if s_ratio < self._min_angle:
                self._min_angle = s_ratio
                
        return frame