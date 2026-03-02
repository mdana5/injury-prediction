#!/usr/bin/env python3
# full_report_from_video.py
"""
Create a full report for a video:
 - Extract keypoints (MediaPipe)
 - Extract corrected 28-dim features (per frame)
 - Sliding-window predictions (window size TIMESTEPS)
 - Save annotated output video with skeleton + risk text
 - Export CSV: per-frame features + predicted risk (for each window centered at frame)
 - Save plots (risk over time + example features)
 - Produce a PDF report with the final summary and plots
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import argparse
import csv
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------
TIMESTEPS = 32
mp_pose = mp.solutions.pose

# -------------------------
# Reuse corrected helper functions (angle, feature extraction, pad/trim)
# -------------------------
def angle_3pts(a, b, c):
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.dot(v1, v2)/(n1*n2)
    return np.arccos(np.clip(cosang, -1, 1))

def extract_all_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
        else:
            arr = np.zeros((33,4), dtype=np.float32)
        frames.append(arr)
    cap.release()
    return np.array(frames)

def extract_features_seq(kps):
    T = kps.shape[0]
    xy = kps[:, :, :2].astype(np.float32)
    norm = np.copy(xy)
    L_SH, L_HIP = 11, 23
    for i in range(T):
        ls, lh = xy[i, L_SH], xy[i, L_HIP]
        if not np.any(np.isnan(ls)) and not np.any(np.isnan(lh)):
            d = np.linalg.norm(ls - lh)
            if d > 1e-6:
                norm[i] = (xy[i] - lh) / d
    k = norm
    L_HIP, R_HIP = 23, 24
    L_KNE, R_KNE = 25, 26
    L_ANK, R_ANK = 27, 28
    L_SHO, R_SHO = 11, 12
    L_ELB, R_ELB = 13, 14
    L_WRI, R_WRI = 15, 16
    NOSE = 0
    left_knee   = np.array([angle_3pts(k[i][L_HIP], k[i][L_KNE], k[i][L_ANK]) for i in range(T)])
    right_knee  = np.array([angle_3pts(k[i][R_HIP], k[i][R_KNE], k[i][R_ANK]) for i in range(T)])
    left_hip    = np.array([angle_3pts(k[i][L_SHO], k[i][L_HIP], k[i][L_KNE]) for i in range(T)])
    right_hip   = np.array([angle_3pts(k[i][R_SHO], k[i][R_HIP], k[i][R_KNE]) for i in range(T)])
    offset = np.array([1e-3, 0])
    left_ank    = np.array([angle_3pts(k[i][L_KNE], k[i][L_ANK], k[i][L_ANK] + offset) for i in range(T)])
    right_ank   = np.array([angle_3pts(k[i][R_KNE], k[i][R_ANK], k[i][R_ANK] + offset) for i in range(T)])
    left_elb    = np.array([angle_3pts(k[i][L_SHO], k[i][L_ELB], k[i][L_WRI]) for i in range(T)])
    right_elb   = np.array([angle_3pts(k[i][R_SHO], k[i][R_ELB], k[i][R_WRI]) for i in range(T)])
    left_sho    = np.array([angle_3pts(k[i][NOSE], k[i][L_SHO], k[i][L_ELB]) for i in range(T)])
    right_sho   = np.array([angle_3pts(k[i][NOSE], k[i][R_SHO], k[i][R_ELB]) for i in range(T)])
    torso_bend  = np.array([angle_3pts(k[i][L_HIP], k[i][L_SHO], k[i][NOSE]) for i in range(T)])
    wrist_l     = np.array([angle_3pts(k[i][L_ELB], k[i][L_WRI], k[i][L_WRI] + offset) for i in range(T)])
    wrist_r     = np.array([angle_3pts(k[i][R_ELB], k[i][R_WRI], k[i][R_WRI] + offset) for i in range(T)])
    def vel(x): return np.diff(x, prepend=x[0])
    left_knee_vel   = vel(left_knee)
    right_knee_vel  = vel(right_knee)
    left_hip_vel    = vel(left_hip)
    right_hip_vel   = vel(right_hip)
    left_ank_vel    = vel(left_ank)
    right_ank_vel   = vel(right_ank)
    left_elb_vel    = vel(left_elb)
    right_elb_vel   = vel(right_elb)
    left_sho_vel    = vel(left_sho)
    right_sho_vel   = vel(right_sho)
    torso_vel       = vel(torso_bend)
    shoulder_width = np.array([np.linalg.norm(k[i][L_SHO] - k[i][R_SHO]) for i in range(T)])
    hip_width      = np.array([np.linalg.norm(k[i][L_HIP] - k[i][R_HIP]) for i in range(T)])
    torso_length   = np.array([np.linalg.norm(k[i][L_SHO] - k[i][L_HIP]) for i in range(T)])
    vertical_disp = np.array([(((k[i][L_HIP] + k[i][R_HIP]) / 2.0))[1] for i in range(T)])
    feats = np.column_stack([
        left_knee, right_knee, left_hip, right_hip,
        left_ank, right_ank, left_elb, right_elb,
        left_sho, right_sho, torso_bend, wrist_l, wrist_r,
        left_knee_vel, right_knee_vel, left_hip_vel, right_hip_vel,
        left_ank_vel, right_ank_vel, left_elb_vel, right_elb_vel,
        left_sho_vel, right_sho_vel, torso_vel,
        shoulder_width, hip_width, torso_length, vertical_disp
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

def pad_or_trim(feats, T=TIMESTEPS):
    if len(feats) == T: return feats
    if len(feats) > T: return feats[:T]
    pad_len = T - len(feats)
    pad = np.zeros((pad_len, feats.shape[1]), dtype=np.float32)
    return np.vstack([feats, pad])


# -------------------------
# Drawing helpers (overlay)
# -------------------------
def draw_landmarks_overlay(frame, landmarks_33):
    h, w = frame.shape[:2]
    # landmarks_33: (33,4) normalized x,y,... ; if zeroed, skip drawing
    for i, lm in enumerate(landmarks_33):
        x, y = int(lm[0]*w), int(lm[1]*h)
        v = lm[3] if len(lm) > 3 else 1.0
        if v > 0.1:
            cv2.circle(frame, (x,y), 3, (0,255,0), -1)
    # draw some connections (simple)
    skeleton_idx = [
        (11,13),(13,15),(12,14),(14,16), # arms
        (11,12),(23,24),(11,23),(12,24), # shoulders-hips
        (25,27),(26,28),(23,25),(24,26)  # legs
    ]
    for a,b in skeleton_idx:
        xa, ya = int(landmarks_33[a][0]*w), int(landmarks_33[a][1]*h)
        xb, yb = int(landmarks_33[b][0]*w), int(landmarks_33[b][1]*h)
        cv2.line(frame, (xa,ya), (xb,yb), (192,192,0), 2)
    return frame

# -------------------------
# CSV writer
# -------------------------
def save_csv(out_csv_path, feats, probs):
    """
    feats: (T,28); probs: (T,) predicted for window starting/centered at frame
    """
    header = ["frame_index"] + [f"f{i}" for i in range(feats.shape[1])] + ["risk"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(feats.shape[0]):
            row = [i] + feats[i].tolist() + [float(probs[i])]
            writer.writerow(row)

# -------------------------
# Plots
# -------------------------
def save_plots(out_dir, probs, feats):
    os.makedirs(out_dir, exist_ok=True)
    frames = np.arange(len(probs))
    # Risk over time
    plt.figure()
    plt.plot(frames, probs)
    plt.xlabel("Frame")
    plt.ylabel("Risk (probability)")
    plt.title("Injury Risk over frames")
    risk_png = os.path.join(out_dir, "risk_over_time.png")
    plt.savefig(risk_png)
    plt.close()
    # Example feature plots: left_knee (0), torso_bend (10), shoulder_width (24)
    plt.figure()
    plt.plot(frames, feats[:,0], label="left_knee")
    plt.plot(frames, feats[:,10], label="torso_bend")
    plt.plot(frames, feats[:,24], label="shoulder_width")
    plt.xlabel("Frame")
    plt.legend()
    feat_png = os.path.join(out_dir, "example_features.png")
    plt.savefig(feat_png)
    plt.close()
    return risk_png, feat_png

# -------------------------
# PDF report (simple)
# -------------------------
def create_pdf_report(pdf_path, final_prob, risk_png, feat_png, summary_text=""):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, height-50, "Injury Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height-80, f"Final predicted risk (last frame window): {final_prob*100:.2f}%")
    if summary_text:
        c.drawString(40, height-100, summary_text)
    # Insert risk plot
    y = height - 200
    try:
        c.drawImage(risk_png, 40, y-150, width=500, height=140)
    except Exception:
        pass
    # Insert feature plot
    try:
        c.drawImage(feat_png, 40, y-320, width=500, height=140)
    except Exception:
        pass
    c.showPage()
    c.save()

# -------------------------
# Main processing
# -------------------------
def process_video(video_path, model_path, scaler_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Extracting keypoints...")
    kps = extract_all_keypoints(video_path)  # (T,33,4)
    T = kps.shape[0]
    print(f"Frames: {T}")

    print("Extracting per-frame features...")
    feats = extract_features_seq(kps)  # (T,28)
    feats = np.nan_to_num(feats)

    print("Loading model and scaler...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Sliding-window predictions: produce one prob per frame by using window [i, i+TIMESTEPS)
    print("Computing sliding-window predictions...")
    probs = np.zeros(T, dtype=np.float32)
    for i in range(T):
        window = feats[i:i+TIMESTEPS]
        window = pad_or_trim(window, TIMESTEPS)
        # scaler expects (TIMESTEPS,28) transformed per-row
        norm = scaler.transform(window)
        # model expects (1, TIMESTEPS, 28)
        p = model.predict(norm[np.newaxis, ...], verbose=0)[0].reshape(-1)[0]
        if np.isnan(p): p = 0.0
        probs[i] = float(p)

    # Annotated video
    print("Writing annotated video...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path = os.path.join(out_dir, "annotated_output.mp4")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w,h))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_kp = kps[frame_idx] if frame_idx < T else np.zeros((33,4))
        frame = draw_landmarks_overlay(frame, frame_kp)
        risk_text = f"Risk: {probs[frame_idx]*100:.1f}%"
        cv2.putText(frame, risk_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        writer.write(frame)
        frame_idx += 1
    writer.release()
    cap.release()

    # Save CSV
    csv_path = os.path.join(out_dir, "features_and_risk.csv")
    save_csv(csv_path, feats, probs)

    # Save plots
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    risk_png, feat_png = save_plots(plots_dir, probs, feats)

    # PDF
    pdf_path = os.path.join(out_dir, "report.pdf")
    final_prob = float(probs[-1]) if len(probs)>0 else 0.0
    create_pdf_report(pdf_path, final_prob, risk_png, feat_png, summary_text=f"Processed {T} frames.")
    print("Report generated:")
    print(" - annotated video:", out_video_path)
    print(" - csv:", csv_path)
    print(" - plots:", risk_png, feat_png)
    print(" - pdf:", pdf_path)
    return {
        "annotated_video": out_video_path,
        "csv": csv_path,
        "plots": (risk_png, feat_png),
        "pdf": pdf_path,
        "final_prob": final_prob
    }

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video file")
    parser.add_argument("--model", required=True, help="keras model .h5")
    parser.add_argument("--scaler", required=True, help="scaler .pkl (joblib)")
    parser.add_argument("--out_dir", default="report_output", help="output directory")
    args = parser.parse_args()

    process_video(args.video, args.model, args.scaler, args.out_dir)
# running command - python full_report_from_video.py --video input_video.mp4 --model injury_risk_model.h5 --scaler scaler_windows.pkl --out_dir report_output
