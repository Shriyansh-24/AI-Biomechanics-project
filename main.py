#!/usr/bin/env python3
# ============================================================
# Markerless Motion Capture — Injury Prevention System
# WITH Auto High-Risk Screenshot Capture
# For NSRI Sports Biomechanics Research
# ============================================================

import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
from collections import deque
from mediapipe.framework.formats import landmark_pb2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

MODEL_PATH          = "pose_landmarker_heavy.task"
# OLD — replace these two lines
# NEW — each session gets its own timestamped folder
_SESSION_TS      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SCREENSHOTS_DIR  = os.path.join("risk_screenshots", f"session_{_SESSION_TS}")
REPORT_FILE      = os.path.join(SCREENSHOTS_DIR, "risk_report.txt")
CSV_FILENAME = f"session_{_SESSION_TS}_data.csv"

AUTO_SAVE_INTERVAL  = 30      # save to CSV every N frames
COUNTDOWN_SECONDS   = 5       # seconds before manual save
SCREENSHOT_COOLDOWN = 2.0     # minimum seconds between screenshots
HIGH_RISK_THRESHOLD = 40      # risk score that triggers a screenshot

# Injury detection thresholds
VALGUS_THRESHOLD    = 0.04
ASYMMETRY_THRESHOLD = 15.0
SHALLOW_THRESHOLD   = 130.0

# Landmark indices
RIGHT_HIP = 24; RIGHT_KNEE = 26; RIGHT_ANKLE = 28
LEFT_HIP  = 23; LEFT_KNEE  = 25; LEFT_ANKLE  = 27

mp_drawing          = mp.solutions.drawing_utils
mp_drawing_styles   = mp.solutions.drawing_styles
mp_pose_connections = mp.solutions.pose.POSE_CONNECTIONS

# ============================================================
# SECTION 2: SETUP FOLDERS
# ============================================================

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ============================================================
# SECTION 3: BIOMECHANICS MATH
# ============================================================

def calculate_3d_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2  = a - b, c - b
    m1, m2  = np.linalg.norm(v1), np.linalg.norm(v2)
    if m1 == 0 or m2 == 0:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
    return round(np.degrees(np.arccos(cos_a)), 2)

def get_coords(landmarks, idx):
    lm = landmarks[idx]
    return [lm.x, lm.y, lm.z]

def to_pixel(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

# ============================================================
# SECTION 3b: ORIENTATION DETECTOR
# Figures out if the athlete is facing the camera (FRONT)
# or standing sideways (SIDE) by measuring shoulder width.
#
# Front view: both shoulders visible → large horizontal gap
# Side view:  one shoulder hides behind the other → small gap
# ============================================================

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

def detect_orientation(landmarks, front_threshold=0.15):
    """
    Returns 'FRONT', 'SIDE', or 'UNKNOWN'.

    front_threshold: minimum x-gap between shoulders to be
                     considered front-facing (tweak if needed).
                     Default 0.15 works well for most webcams.
    """
    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]

    # Horizontal distance between shoulders (normalized 0-1)
    shoulder_width = abs(ls.x - rs.x)

    if shoulder_width > front_threshold:
        return "FRONT", round(shoulder_width, 3)
    else:
        return "SIDE",  round(shoulder_width, 3)


# ============================================================
# SECTION 4: INJURY RISK DETECTION
# ============================================================

def detect_all_risks(landmarks, right_angle, left_angle,
                     orientation="UNKNOWN", is_squatting=False):
    """
    orientation controls which checks are active:
      FRONT → valgus + asymmetry ON,  shallow/hyperextension ON
      SIDE  → valgus + asymmetry OFF, shallow/hyperextension ON
      UNKNOWN → same as SIDE (safe default)

    is_squatting: True only when RepCounter state is SQUATTING or RISING.
      Depth and hyperextension checks only make sense mid-squat.
      When standing still, a 170 degree angle is normal — not a problem.
    """
    problems = []

    r_hip   = get_coords(landmarks, RIGHT_HIP)
    r_knee  = get_coords(landmarks, RIGHT_KNEE)
    r_ankle = get_coords(landmarks, RIGHT_ANKLE)
    l_hip   = get_coords(landmarks, LEFT_HIP)
    l_knee  = get_coords(landmarks, LEFT_KNEE)
    l_ankle = get_coords(landmarks, LEFT_ANKLE)

    is_front = (orientation == "FRONT")

    # Check 1: Knee Valgus — FRONT VIEW ONLY
    # Side view makes the knee appear to "cave" even when perfectly
    # aligned, so we skip this check entirely when sideways.
    if is_front:
        for side, hip, knee, ankle in [
            ("Right", r_hip, r_knee, r_ankle),
            ("Left",  l_hip, l_knee, l_ankle)
        ]:
            ideal_x   = (hip[0] + ankle[0]) / 2.0
            deviation = knee[0] - ideal_x
            if deviation < -VALGUS_THRESHOLD:
                severity = "HIGH" if deviation < -0.07 else "MEDIUM"
                problems.append({
                    "type":        f"{side} Knee Valgus",
                    "severity":    severity,
                    "explanation": (
                        f"The {side.lower()} knee is collapsing inward. "
                        f"This twists the ACL ligament under load — "
                        f"the #1 cause of ACL tears in sport."
                    ),
                    "fix": (
                        f"Push the {side.lower()} knee out over your little toe. "
                        f"Strengthen glute medius with clamshells and lateral band walks."
                    ),
                    "color_bgr": (0, 0, 220),
                    "knee_side":  side
                })

    # Check 2: Left-Right Asymmetry — FRONT VIEW ONLY
    # From the side, only one leg is visible so comparison is meaningless.
    if is_front and right_angle is not None and left_angle is not None:
        diff = abs(right_angle - left_angle)
        if diff > ASYMMETRY_THRESHOLD:
            weaker   = "Left" if right_angle < left_angle else "Right"
            severity = "HIGH" if diff > 25 else "MEDIUM"
            problems.append({
                "type":        "Left-Right Asymmetry",
                "severity":    severity,
                "explanation": (
                    f"Knees differ by {round(diff,1)}° "
                    f"(R:{right_angle}  L:{left_angle}). "
                    f"The {weaker} side is weaker, causing overuse injuries."
                ),
                "fix": (
                    f"Train the {weaker} side with Bulgarian split squats "
                    f"and single-leg press exercises."
                ),
                "color_bgr": (0, 140, 255),
                "knee_side":  "Both"
            })

    # Check 3: Shallow Squat — ONLY DURING SQUAT MOVEMENT
    # A standing person always has a high knee angle (~170°).
    # Flagging this as "too shallow" when they haven't started
    # squatting yet is a false positive. Only check depth when
    # the rep counter confirms a squat is actually in progress.
    if is_squatting:
        for side, angle in [("Right", right_angle), ("Left", left_angle)]:
            if angle is not None and angle > SHALLOW_THRESHOLD:
                problems.append({
                    "type":        f"Shallow Squat ({side})",
                    "severity":    "MEDIUM",
                    "explanation": (
                        f"{side} knee at {angle} degrees — too shallow. "
                        f"Overloads the knee cap and reduces glute activation."
                    ),
                    "fix": (
                        f"Work on ankle and hip mobility. "
                        f"Try heel-elevated squats to build depth."
                    ),
                    "color_bgr": (0, 200, 255),
                    "knee_side":  side
                })

    # Check 4: Hyperextension — ONLY DURING SQUAT MOVEMENT
    # Hyperextension is only a risk when the knee is under load
    # during a movement. Standing relaxed with straight legs is
    # completely normal and should not be flagged.
    if is_squatting:
        for side, angle in [("Right", right_angle), ("Left", left_angle)]:
            if angle is not None and angle > 175:
                problems.append({
                    "type":        f"Hyperextension Risk ({side})",
                    "severity":    "HIGH",
                    "explanation": (
                        f"{side} knee at {angle} degrees — near full lock-out under load. "
                        f"Stresses the PCL ligament and posterior capsule."
                    ),
                    "fix": (
                        f"Keep a soft 5-10 degree bend at the top of every rep."
                    ),
                    "color_bgr": (0, 0, 220),
                    "knee_side":  side
                })

    return problems


def calculate_risk_score(problems):
    score = sum({"HIGH": 40, "MEDIUM": 20}.get(p["severity"], 10) for p in problems)
    return min(score, 100)


def detect_knee_valgus(hip, knee, ankle):
    ideal_x   = (hip[0] + ankle[0]) / 2.0
    deviation = knee[0] - ideal_x
    return deviation < -VALGUS_THRESHOLD, round(deviation, 4)


def detect_asymmetry(right_angle, left_angle):
    if right_angle is None or left_angle is None:
        return False, None
    diff = abs(right_angle - left_angle)
    return diff > ASYMMETRY_THRESHOLD, round(diff, 2)


def detect_fatigue(rep_depths, drop=20.0):
    if len(rep_depths) < 2:
        return False, 0.0
    return (rep_depths[-1] - rep_depths[0]) > drop, round(rep_depths[-1] - rep_depths[0], 2)

# ============================================================
# SECTION 5: BUILD ANNOTATED SCREENSHOT
# Works on a COPY of the frame — live feed is never affected
# ============================================================

def build_annotated_screenshot(frame, landmarks, problems,
                                right_angle, left_angle,
                                risk_score, frame_number, timestamp):
    h, w = frame.shape[:2]
    img  = frame.copy()

    # Skeleton
    lp = landmark_pb2.NormalizedLandmarkList()
    for lm in landmarks:
        lp.landmark.add(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
    mp_drawing.draw_landmarks(
        img, lp, mp_pose_connections,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )

    # Red circles on bad knees
    for prob in problems:
        col = prob["color_bgr"]
        if prob["knee_side"] in ["Right", "Both"]:
            cv2.circle(img, to_pixel(landmarks[RIGHT_KNEE], w, h), 28, col, 3)
            cv2.circle(img, to_pixel(landmarks[RIGHT_KNEE], w, h), 38, col, 1)
        if prob["knee_side"] in ["Left", "Both"]:
            cv2.circle(img, to_pixel(landmarks[LEFT_KNEE], w, h), 28, col, 3)
            cv2.circle(img, to_pixel(landmarks[LEFT_KNEE], w, h), 38, col, 1)

    # Top banner
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, 55), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
    cv2.putText(img, "HIGH RISK POSITION",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2, cv2.LINE_AA)
    cv2.putText(img, f"Frame {frame_number}  |  {timestamp}",
                (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Risk badge
    badge_col = (0, 0, 180) if risk_score > 50 else (0, 110, 200)
    cv2.rectangle(img, (w - 135, 4), (w - 4, 52), badge_col, -1)
    cv2.putText(img, "RISK SCORE",
                (w - 128, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, f"{risk_score} / 100",
                (w - 128, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Knee angle labels
    for s, idx, ang in [("R", RIGHT_KNEE, right_angle), ("L", LEFT_KNEE, left_angle)]:
        if ang is not None:
            px = to_pixel(landmarks[idx], w, h)
            cv2.putText(img, f"{s}:{ang}",
                        (px[0]+12, px[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Problem labels
    y = 72
    for prob in problems:
        label = f"[{prob['severity']}] {prob['type']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ov2 = img.copy()
        cv2.rectangle(ov2, (6, y-3), (tw+16, y+th+4), (0,0,0), -1)
        cv2.addWeighted(ov2, 0.55, img, 0.45, 0, img)
        cv2.putText(img, label, (10, y+th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, prob["color_bgr"], 1, cv2.LINE_AA)
        y += th + 16

    # Bottom explanation + fix strip
    if problems:
        p   = problems[0]
        exp = p['explanation'][:85] + "..." if len(p['explanation']) > 85 else p['explanation']
        fix = "FIX: " + (p['fix'][:80] + "..." if len(p['fix']) > 80 else p['fix'])
        ov3 = img.copy()
        cv2.rectangle(ov3, (0, h-58), (w, h), (0,0,0), -1)
        cv2.addWeighted(ov3, 0.75, img, 0.25, 0, img)
        cv2.putText(img, exp,  (10, h-38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(img, fix,  (10, h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150,255,150), 1, cv2.LINE_AA)

    return img

# ============================================================
# SECTION 6: SAVE SCREENSHOT + REPORT
# ============================================================

screenshot_log = []

def save_risk_screenshot(annotated_img, problems, risk_score,
                          frame_number, timestamp, right_angle, left_angle):
    filename = f"risk_{timestamp}_f{frame_number}_score{risk_score}.jpg"
    filepath = os.path.join(SCREENSHOTS_DIR, filename)
    cv2.imwrite(filepath, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    screenshot_log.append(filename)

    with open(REPORT_FILE, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"FILE   : {filename}\n")
        f.write(f"TIME   : {timestamp}  |  Frame: {frame_number}\n")
        f.write(f"RISK   : {risk_score}/100\n")
        f.write(f"ANGLES : Right={right_angle}  Left={left_angle}\n")
        f.write("-" * 60 + "\n")
        for prob in problems:
            f.write(f"\n  [{prob['severity']}] {prob['type']}\n")
            f.write(f"  WHAT IS WRONG : {prob['explanation']}\n")
            f.write(f"  HOW TO FIX    : {prob['fix']}\n")

    print(f"[SCREENSHOT] {filename} | Risk:{risk_score}/100")
    for p in problems:
        print(f"             -> [{p['severity']}] {p['type']}")

# ============================================================
# SECTION 7: LIVE FEEDBACK
# ============================================================

def generate_live_feedback(angle, valgus, asym, asym_val, fatigued, risk):
    fb = []
    if angle is not None:
        if   angle < 70:  fb.append(("DEPTH: Excellent",     (0,255,0)))
        elif angle < 100: fb.append(("DEPTH: Good",          (0,255,0)))
        elif angle < 130: fb.append(("DEPTH: Shallow",       (0,165,255)))
        else:             fb.append(("DEPTH: Too shallow",   (0,0,255)))
    if valgus:
        fb.append(("! KNEE CAVING INWARD",                   (0,0,255)))
    if asym and asym_val:
        fb.append((f"! ASYMMETRY: {asym_val} deg",           (0,0,255)))
    if fatigued:
        fb.append(("! FATIGUE — rest recommended",           (0,165,255)))
    sc = (0,255,0) if risk < 20 else (0,165,255) if risk < 50 else (0,0,255)
    fb.append((f"RISK: {risk}/100",                           sc))
    return fb

# ============================================================
# SECTION 8: CSV EXPORT
# ============================================================

def save_to_csv(frame_number, timestamp, side,
                hip_c, knee_c, ankle_c, angle,
                valgus, asym_val, fatigued, risk, rep):
    file_exists = os.path.isfile(CSV_FILENAME)
    with open(CSV_FILENAME, mode='a', newline='') as f:
        fields = ['frame','timestamp_s','rep_number','side',
                  'hip_x','hip_y','hip_z','knee_x','knee_y','knee_z',
                  'ankle_x','ankle_y','ankle_z','knee_angle_deg',
                  'knee_valgus','asymmetry_deg','fatigue_detected',
                  'risk_score_0_to_100']
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'frame': frame_number, 'timestamp_s': round(timestamp,3),
            'rep_number': rep, 'side': side,
            'hip_x': hip_c[0],    'hip_y': hip_c[1],    'hip_z': hip_c[2],
            'knee_x': knee_c[0],  'knee_y': knee_c[1],  'knee_z': knee_c[2],
            'ankle_x': ankle_c[0],'ankle_y': ankle_c[1],'ankle_z': ankle_c[2],
            'knee_angle_deg': angle, 'knee_valgus': valgus,
            'asymmetry_deg': asym_val if asym_val else 'N/A',
            'fatigue_detected': fatigued, 'risk_score_0_to_100': risk
        })

# ============================================================
# SECTION 9: REP COUNTER
# ============================================================

class RepCounter:
    def __init__(self):
        self.state       = "STANDING"
        self.rep_count   = 0
        self.current_min = 180.0
        self.rep_depths  = []

    def update(self, angle):
        if angle is None:
            return False
        done = False
        if self.state == "STANDING":
            if angle < 130:
                self.state = "SQUATTING"; self.current_min = angle
        elif self.state == "SQUATTING":
            if angle < self.current_min: self.current_min = angle
            elif angle > self.current_min + 15: self.state = "RISING"
        elif self.state == "RISING":
            if angle > 150:
                self.rep_count += 1
                self.rep_depths.append(self.current_min)
                self.state = "STANDING"; self.current_min = 180.0
                done = True
                print(f"[REP] #{self.rep_count} depth:{self.rep_depths[-1]}")
        return done

# ============================================================
# SECTION 10: SESSION SUMMARY
# ============================================================

def write_session_summary(total_frames, duration):
    if not screenshot_log:
        print("[INFO] No high-risk positions detected this session.")
        return

    summary = [
        "=" * 60,
        "   HIGH RISK POSITION REPORT — SESSION SUMMARY",
        "=" * 60,
        f"Duration        : {round(duration,1)} seconds",
        f"Total Frames    : {total_frames}",
        f"Screenshots     : {len(screenshot_log)}",
        f"Saved to        : {SCREENSHOTS_DIR}/",
        "", "SCREENSHOTS TAKEN:"
    ]
    for s in screenshot_log:
        summary.append(f"  • {s}")
    summary += ["", "FULL DETAILS BELOW:", ""]

    existing = open(REPORT_FILE).read() if os.path.exists(REPORT_FILE) else ""
    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(summary))
        f.write(existing)

    print(f"\n[REPORT] Saved: '{REPORT_FILE}'")
    print(f"[DONE]   {len(screenshot_log)} screenshots in '{SCREENSHOTS_DIR}/'")

# ============================================================
# SECTION 11: MAIN
# ============================================================

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: '{MODEL_PATH}' not found.")
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    print("[SYSTEM] Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return

    frame_count          = 0
    start_time           = time.time()
    last_screenshot_time = 0
    countdown_active     = False
    countdown_start      = None

    right_ctr = RepCounter()
    left_ctr  = RepCounter()
    r_hist    = deque(maxlen=10)
    l_hist    = deque(maxlen=10)

    print(f"[SYSTEM] Running. Risk screenshots -> '{SCREENSHOTS_DIR}/'")
    print("[SYSTEM] S=countdown save  Q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        elapsed      = time.time() - start_time
        h, w, _      = frame.shape

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(elapsed * 1000))

        # Reset
        right_angle = left_angle  = None
        problems    = []
        risk_score  = 0
        valgus_r = valgus_l = asym_flag = False
        asym_val = None
        r_fatigue = l_fatigue = False
        r_hip_c = r_knee_c = r_ankle_c = None
        l_hip_c = l_knee_c = l_ankle_c = None

        if result.pose_landmarks:
            lms = result.pose_landmarks[0]

            r_hip_c   = get_coords(lms, RIGHT_HIP)
            r_knee_c  = get_coords(lms, RIGHT_KNEE)
            r_ankle_c = get_coords(lms, RIGHT_ANKLE)
            l_hip_c   = get_coords(lms, LEFT_HIP)
            l_knee_c  = get_coords(lms, LEFT_KNEE)
            l_ankle_c = get_coords(lms, LEFT_ANKLE)

            right_angle = calculate_3d_angle(r_hip_c, r_knee_c, r_ankle_c)
            left_angle  = calculate_3d_angle(l_hip_c, l_knee_c, l_ankle_c)

            if right_angle: r_hist.append(right_angle)
            if left_angle:  l_hist.append(left_angle)
            smooth_r = round(np.mean(r_hist), 1) if r_hist else None
            smooth_l = round(np.mean(l_hist), 1) if l_hist else None

            right_ctr.update(right_angle)
            left_ctr.update(left_angle)

            # Detect which way athlete is facing THIS frame
            orientation, shoulder_gap = detect_orientation(lms)

            # True when either leg is mid-squat (SQUATTING or RISING state)
            is_squatting = (right_ctr.state in ("SQUATTING", "RISING") or
                            left_ctr.state  in ("SQUATTING", "RISING"))

            problems    = detect_all_risks(lms, right_angle, left_angle,
                                           orientation, is_squatting)
            risk_score  = calculate_risk_score(problems)

            # Valgus only meaningful from front view
            if orientation == "FRONT":
                valgus_r, _ = detect_knee_valgus(r_hip_c, r_knee_c, r_ankle_c)
                valgus_l, _ = detect_knee_valgus(l_hip_c, l_knee_c, l_ankle_c)
            else:
                valgus_r = valgus_l = False

            # Asymmetry only meaningful from front view
            if orientation == "FRONT":
                asym_flag, asym_val = detect_asymmetry(right_angle, left_angle)
            else:
                asym_flag, asym_val = False, None
            r_fatigue, _ = detect_fatigue(right_ctr.rep_depths)
            l_fatigue, _ = detect_fatigue(left_ctr.rep_depths)

            r_risk = calculate_risk_score(detect_all_risks(lms, right_angle, None,
                                                            orientation, is_squatting))
            l_risk = calculate_risk_score(detect_all_risks(lms, None, left_angle,
                                                            orientation, is_squatting))

            # Draw skeleton on live frame
            lp = landmark_pb2.NormalizedLandmarkList()
            for lm in lms:
                lp.landmark.add(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            mp_drawing.draw_landmarks(
                frame, lp, mp_pose_connections,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Angle text
            for s, idx, ang, rv in [("R", RIGHT_KNEE, smooth_r, r_risk),
                                     ("L", LEFT_KNEE,  smooth_l, l_risk)]:
                if ang is not None:
                    col = (0,255,0) if rv<20 else (0,165,255) if rv<50 else (0,0,255)
                    px  = to_pixel(lms[idx], w, h)
                    cv2.putText(frame, f"{s}:{ang}",
                                (px[0]+10, px[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

            # Valgus circles
            if valgus_r: cv2.circle(frame, to_pixel(lms[RIGHT_KNEE],w,h), 28,(0,0,220),3)
            if valgus_l: cv2.circle(frame, to_pixel(lms[LEFT_KNEE], w,h), 28,(0,0,220),3)

            # Orientation badge — shows what mode is active
            orient_color = (0, 255, 255) if orientation == "FRONT" else (255, 200, 0)
            orient_label = (f"MODE: {orientation} VIEW"
                            f"  ({'valgus ON' if orientation == 'FRONT' else 'valgus OFF'})")
            cv2.putText(frame, orient_label,
                        (10, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, orient_color, 1, cv2.LINE_AA)

            # Feedback panel
            dom_ang = right_angle if r_risk>=l_risk else left_angle
            dom_v   = valgus_r    if r_risk>=l_risk else valgus_l
            dom_fat = r_fatigue   if r_risk>=l_risk else l_fatigue
            fb = generate_live_feedback(dom_ang, dom_v, asym_flag, asym_val,
                                         dom_fat, risk_score)
            ov = frame.copy()
            cv2.rectangle(ov, (4,50), (300, 56+len(fb)*32), (0,0,0), -1)
            cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
            for i,(msg,col) in enumerate(fb):
                cv2.putText(frame, msg, (10, 74+i*32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)

            # ── AUTO SCREENSHOT on high risk ─────────────────
            cooldown_ok = (elapsed - last_screenshot_time) >= SCREENSHOT_COOLDOWN
            if risk_score >= HIGH_RISK_THRESHOLD and cooldown_ok and problems:
                ts_str = datetime.now().strftime("%H-%M-%S")
                ann    = build_annotated_screenshot(
                    frame.copy(), lms, problems,
                    right_angle, left_angle,
                    risk_score, frame_count, ts_str
                )
                save_risk_screenshot(ann, problems, risk_score,
                                     frame_count, ts_str,
                                     right_angle, left_angle)
                last_screenshot_time = elapsed

            # Auto CSV save
            if frame_count % AUTO_SAVE_INTERVAL == 0:
                if right_angle and r_hip_c:
                    save_to_csv(frame_count, elapsed, "RIGHT",
                                r_hip_c, r_knee_c, r_ankle_c,
                                right_angle, valgus_r, asym_val,
                                r_fatigue, r_risk, right_ctr.rep_count)
                if left_angle and l_hip_c:
                    save_to_csv(frame_count, elapsed, "LEFT",
                                l_hip_c, l_knee_c, l_ankle_c,
                                left_angle, valgus_l, asym_val,
                                l_fatigue, l_risk, left_ctr.rep_count)

        # HUD
        cv2.rectangle(frame, (0,0), (w,44), (0,0,0), -1)
        cv2.putText(frame, f"Frame:{frame_count}",
                    (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        cv2.putText(frame, f"Time:{round(elapsed,1)}s",
                    (120,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1)
        cv2.putText(frame,
                    f"Reps R:{right_ctr.rep_count} L:{left_ctr.rep_count}",
                    (230,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)
        cv2.putText(frame, f"Screenshots:{len(screenshot_log)}",
                    (w-170,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),1)
        cv2.putText(frame, "S=Save  Q=Quit",
                    (8,38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(160,160,160),1)

        # Countdown
        if countdown_active:
            tl = COUNTDOWN_SECONDS - (time.time() - countdown_start)
            if tl > 0:
                cv2.putText(frame, f"SAVING IN: {int(tl)+1}",
                            (w//2-160, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0,(0,0,255),4,cv2.LINE_AA)
            else:
                countdown_active = False
                if right_angle and r_hip_c:
                    save_to_csv(frame_count, elapsed,"RIGHT_MANUAL",
                                r_hip_c,r_knee_c,r_ankle_c,
                                right_angle,valgus_r,asym_val,
                                r_fatigue,r_risk,right_ctr.rep_count)
                if left_angle and l_hip_c:
                    save_to_csv(frame_count,elapsed,"LEFT_MANUAL",
                                l_hip_c,l_knee_c,l_ankle_c,
                                left_angle,valgus_l,asym_val,
                                l_fatigue,l_risk,left_ctr.rep_count)
                cv2.putText(frame,"SAVED!",(w//2-80,h//2),
                            cv2.FONT_HERSHEY_SIMPLEX,2.5,(0,255,0),4,cv2.LINE_AA)

        cv2.imshow("Injury Prevention — Motion Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print(f"\n[DONE] Reps R:{right_ctr.rep_count} L:{left_ctr.rep_count}")
            if right_ctr.rep_depths: print(f"       Right depths:{right_ctr.rep_depths}")
            if left_ctr.rep_depths:  print(f"       Left depths: {left_ctr.rep_depths}")
            break
        elif key == ord('s') or key == ord('S'):
            if not countdown_active:
                countdown_active = True
                countdown_start  = time.time()
                print(f"[TIMER] {COUNTDOWN_SECONDS}s countdown...")

    write_session_summary(frame_count, time.time() - start_time)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[SYSTEM] Done.")


if __name__ == "__main__":
    main()