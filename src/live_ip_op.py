#!/usr/bin/env python3
# live_injury_report_fixed.py
"""
Robust live injury pipeline:
- records webcam (preview window shown)
- extracts Mediapipe keypoints (with interpolation/filling)
- computes 28 features in given order
- sliding-window predictions (TIMESTEPS)
- annotated video + CSV + plots + PDF
- prevents NaN outputs with repair, clipping and fallbacks
"""

import os
import time
import cv2
import numpy as np
import joblib
import argparse
import csv
import math
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# try importing tensorflow with friendly message
try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError("TensorFlow import failed. Use Python <=3.10 and install tensorflow. Error: " + str(e))

import mediapipe as mp

# -------------------------
TIMESTEPS = 32
mp_pose = mp.solutions.pose

# -------------------------
# geometry helpers
# -------------------------
def angle_3pts(a, b, c):
    # a,b,c are 2D points (x,y) -- may contain nan
    if a is None or b is None or c is None:
        return np.nan
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.dot(v1, v2)/(n1*n2)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))

def safe_norm(vec):
    v = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n

# -------------------------
# Recording with visible window & debug preview save
# -------------------------
def record_video(duration, output_file="input_video.mp4", preview_file="preview.jpg"):
    # CAP_DSHOW gives better compatibility on Windows; remove if problematic
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # fallback without flag
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (try running as admin or check camera).")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    print("✔ Webcam opened (w,h,fps) =", (w, h, fps))
    print("🎥 Recording started — press 'q' to stop early. Stand back so full body is visible.")

    start_time = time.time()
    last_preview_time = 0
    cv2.namedWindow("Recording...", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recording...", min(960, w), min(720, h))

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        out.write(frame)
        cv2.imshow("Recording...", frame)
        # Save an occasional preview JPG so you can open it if imshow is not visible
        if time.time() - last_preview_time > 1.0:
            try:
                cv2.imwrite(preview_file, frame)
            except Exception:
                pass
            last_preview_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎥 Recording finished:", output_file)
    print("Preview image saved to:", preview_file)
    return output_file

# -------------------------
# Keypoint extraction + repair
# -------------------------
def extract_all_keypoints(video_path, min_visibility=0.4):
    # returns (T,33,4) array of x,y,z,visibility (normalized coords)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open recorded video: " + video_path)

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            # zero-out very low visibility points
            arr[arr[:,3] < min_visibility, :3] = np.nan
        else:
            arr = np.full((33,4), np.nan, dtype=np.float32)
        frames.append(arr)
    cap.release()
    # shape (T,33,4)
    kps = np.array(frames, dtype=np.float32)
    kps = repair_keypoints_timeseries(kps)
    return kps

def repair_keypoints_timeseries(kps):
    # kps: (T,33,4) with x,y,z,vis (vis may be nan if low)
    # We'll interpolate missing x,y,z per landmark across time (linear)
    T, N, C = kps.shape
    repaired = kps.copy()
    for j in range(N):
        for c in range(3):  # x,y,z
            series = repaired[:, j, c]
            mask = np.isfinite(series)
            if mask.sum() == 0:
                # no detection across all frames -> leave as nan
                continue
            if mask.sum() == 1:
                # only one valid point -> broadcast it
                idx = np.where(mask)[0][0]
                repaired[:, j, c] = series[idx]
                continue
            # linear interpolation for NaNs
            nans = ~mask
            if nans.any():
                x = np.arange(T)
                repaired[:, j, c] = np.interp(x, x[mask], series[mask])
    # For visibility column, set to 1 where coords finite else 0
    vis = np.isfinite(repaired[:,:,0]) & np.isfinite(repaired[:,:,1])
    repaired[:,:,3] = vis.astype(np.float32)
    return repaired

# -------------------------
# Feature extraction (exact 28 features, in order)
# -------------------------
def extract_features_seq(kps):
    # kps: (T,33,4) normalized coordinates (x,y)
    T = kps.shape[0]
    xy = kps[:, :, :2].astype(np.float32)
    # We'll normalize per-frame by hip distance if possible; otherwise keep as is
    L_SH, L_HIP = 11, 23
    R_SH, R_HIP = 12, 24

    norm = np.copy(xy)
    for i in range(T):
        lh = xy[i, L_HIP]
        rh = xy[i, R_HIP]
        if np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
            # fallback: try shoulders distance
            ls = xy[i, L_SH]; rs = xy[i, R_SH]
            if not (np.any(np.isnan(ls)) or np.any(np.isnan(rs))):
                d = np.linalg.norm(ls - rs)
                if d > 1e-6:
                    norm[i] = (xy[i] - ls) / d
                else:
                    # cannot normalize, leave raw
                    norm[i] = xy[i]
            else:
                norm[i] = xy[i]
        else:
            center = (lh + rh) / 2.0
            d = np.linalg.norm(lh - rh)
            if d > 1e-6:
                norm[i] = (xy[i] - center) / d
            else:
                norm[i] = xy[i]

    k = norm
    L_HIP, R_HIP = 23,24
    L_KNE, R_KNE = 25,26
    L_ANK, R_ANK = 27,28
    L_SHO, R_SHO = 11,12
    L_ELB, R_ELB = 13,14
    L_WRI, R_WRI = 15,16
    NOSE = 0

    offset = np.array([1e-3, 0], dtype=np.float32)

    def get_pt(i, idx):
        p = k[i, idx]
        if np.any(np.isnan(p)): return np.array([np.nan, np.nan])
        return p

    left_knee  = np.array([angle_3pts(get_pt(i,L_HIP), get_pt(i,L_KNE), get_pt(i,L_ANK)) for i in range(T)])
    right_knee = np.array([angle_3pts(get_pt(i,R_HIP), get_pt(i,R_KNE), get_pt(i,R_ANK)) for i in range(T)])
    left_hip   = np.array([angle_3pts(get_pt(i,L_SHO), get_pt(i,L_HIP), get_pt(i,L_KNE)) for i in range(T)])
    right_hip  = np.array([angle_3pts(get_pt(i,R_SHO), get_pt(i,R_HIP), get_pt(i,R_KNE)) for i in range(T)])
    left_ank   = np.array([angle_3pts(get_pt(i,L_KNE), get_pt(i,L_ANK), get_pt(i,L_ANK)+offset) for i in range(T)])
    right_ank  = np.array([angle_3pts(get_pt(i,R_KNE), get_pt(i,R_ANK), get_pt(i,R_ANK)+offset) for i in range(T)])
    left_elb   = np.array([angle_3pts(get_pt(i,L_SHO), get_pt(i,L_ELB), get_pt(i,L_WRI)) for i in range(T)])
    right_elb  = np.array([angle_3pts(get_pt(i,R_SHO), get_pt(i,R_ELB), get_pt(i,R_WRI)) for i in range(T)])
    left_sho   = np.array([angle_3pts(get_pt(i,NOSE), get_pt(i,L_SHO), get_pt(i,L_ELB)) for i in range(T)])
    right_sho  = np.array([angle_3pts(get_pt(i,NOSE), get_pt(i,R_SHO), get_pt(i,R_ELB)) for i in range(T)])
    torso_bend = np.array([angle_3pts(get_pt(i,L_HIP), get_pt(i,L_SHO), get_pt(i,NOSE)) for i in range(T)])
    wrist_l    = np.array([angle_3pts(get_pt(i,L_ELB), get_pt(i,L_WRI), get_pt(i,L_WRI)+offset) for i in range(T)])
    wrist_r    = np.array([angle_3pts(get_pt(i,R_ELB), get_pt(i,R_WRI), get_pt(i,R_WRI)+offset) for i in range(T)])

    def vel(x):
        x = np.array(x, dtype=np.float32)
        if len(x) == 0:
            return x
        d = np.diff(x, prepend=x[0])
        # replace any NaN diffs by 0
        d[~np.isfinite(d)] = 0.0
        return d

    left_knee_vel  = vel(left_knee)
    right_knee_vel = vel(right_knee)
    left_hip_vel   = vel(left_hip)
    right_hip_vel  = vel(right_hip)
    left_ank_vel   = vel(left_ank)
    right_ank_vel  = vel(right_ank)
    left_elb_vel   = vel(left_elb)
    right_elb_vel  = vel(right_elb)
    left_sho_vel   = vel(left_sho)
    right_sho_vel  = vel(right_sho)
    torso_vel      = vel(torso_bend)

    # widths/lengths: compute robustly with nan handling
    def pair_dist(i, a, b):
        pa = k[i, a]; pb = k[i, b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            return np.nan
        return float(np.linalg.norm(pa - pb))

    shoulder_width = np.array([pair_dist(i, L_SHO, R_SHO) for i in range(T)])
    hip_width      = np.array([pair_dist(i, L_HIP, R_HIP) for i in range(T)])
    torso_length   = np.array([pair_dist(i, L_SHO, L_HIP) for i in range(T)])
    vertical_disp  = np.array([(((k[i][L_HIP] + k[i][R_HIP]) / 2.0))[1] if (not np.any(np.isnan(k[i][L_HIP])) and not np.any(np.isnan(k[i][R_HIP]))) else np.nan for i in range(T)])

    feats = np.column_stack([
        left_knee, right_knee, left_hip, right_hip,
        left_ank, right_ank, left_elb, right_elb,
        left_sho, right_sho, torso_bend, wrist_l, wrist_r,
        left_knee_vel, right_knee_vel, left_hip_vel, right_hip_vel,
        left_ank_vel, right_ank_vel, left_elb_vel, right_elb_vel,
        left_sho_vel, right_sho_vel, torso_vel,
        shoulder_width, hip_width, torso_length, vertical_disp
    ])

    # Replace remaining NaNs with column-wise interpolation/fill:
    feats = fill_feature_nans(feats)
    return feats

def fill_feature_nans(feats):
    # feats: (T,28) -> interpolate along time, then fill edges with nearest or zero
    T, D = feats.shape
    out = feats.copy().astype(np.float32)
    for j in range(D):
        series = out[:, j]
        mask = np.isfinite(series)
        if mask.sum() == 0:
            out[:, j] = 0.0
            continue
        if mask.sum() == 1:
            out[:, j] = series[mask][0]
            continue
        x = np.arange(T)
        out[:, j] = np.interp(x, x[mask], series[mask])
    # ensure no nan remain
    out[~np.isfinite(out)] = 0.0
    return out

# -------------------------
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
    # landmarks_33: (33,4) normalized x,y,... ; may contain NaNs
    h, w = frame.shape[:2]
    for i, lm in enumerate(landmarks_33):
        x, y = lm[0], lm[1]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(x*w), int(y*h)
        cv2.circle(frame, (xi,yi), 3, (0,255,0), -1)
    skeleton_idx = [
        (11,13),(13,15),(12,14),(14,16),
        (11,12),(23,24),(11,23),(12,24),
        (25,27),(26,28),(23,25),(24,26)
    ]
    for a,b in skeleton_idx:
        pa = landmarks_33[a]; pb = landmarks_33[b]
        if not np.isfinite(pa[0]) or not np.isfinite(pa[1]) or not np.isfinite(pb[0]) or not np.isfinite(pb[1]):
            continue
        xa, ya = int(pa[0]*w), int(pa[1]*h)
        xb, yb = int(pb[0]*w), int(pb[1]*h)
        cv2.line(frame, (xa,ya), (xb,yb), (192,192,0), 2)
    return frame

# -------------------------
# CSV writer
# -------------------------
def save_csv(out_csv_path, feats, probs):
    header = ["frame_index"] + [f"f{i}" for i in range(feats.shape[1])] + ["risk"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(feats.shape[0]):
            row = [i] + feats[i].tolist() + [float(probs[i])]
            writer.writerow(row)

# -------------------------
# Plots and PDF (all features)
# -------------------------
def save_plots(out_dir, probs, feats, feature_names):
    os.makedirs(out_dir, exist_ok=True)
    frames = np.arange(len(probs))

    # Risk over time
    plt.figure(figsize=(8,4))
    plt.plot(frames, probs)
    plt.xlabel("Frame")
    plt.ylabel("Risk (probability)")
    plt.title("Injury Risk over frames")
    risk_png = os.path.join(out_dir, "risk_over_time.png")
    plt.tight_layout()
    plt.savefig(risk_png)
    plt.close()

    # Example features
    plt.figure()
    plt.plot(frames, feats[:,0], label=feature_names[0])
    plt.plot(frames, feats[:,10], label=feature_names[10])
    plt.plot(frames, feats[:,24], label=feature_names[24])
    plt.xlabel("Frame")
    plt.legend()
    feat_png = os.path.join(out_dir, "example_features.png")
    plt.savefig(feat_png)
    plt.close()

    # All features grid
    n = feats.shape[1]
    cols = 4
    rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*3, rows*2.5))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        ax.plot(frames, feats[:, i])
        ax.set_title(feature_names[i], fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    all_feat_png = os.path.join(out_dir, "all_features.png")
    plt.savefig(all_feat_png, dpi=150)
    plt.close()

    return risk_png, feat_png, all_feat_png

def create_pdf_report(pdf_path, final_prob, risk_png, feat_png, all_feat_png, summary_text=""):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, height-50, "Injury Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(30, height-80, f"Final predicted risk: {final_prob*100:.2f}%")
    if summary_text:
        c.drawString(30, height-100, summary_text)
    y = height - 160
    try:
        c.drawImage(risk_png, 30, y-140, width=520, height=120)
    except Exception:
        pass
    try:
        c.drawImage(feat_png, 30, y-300, width=520, height=120)
    except Exception:
        pass
    c.showPage()
    # second page: all features
    try:
        c.drawImage(all_feat_png, 20, 20, width=560, height=760)
    except Exception:
        pass
    c.showPage()
    c.save()

# -------------------------
# Main pipeline
# -------------------------
FEATURE_NAMES = [
    "left_knee_angle","right_knee_angle","left_hip_angle","right_hip_angle",
    "left_ankle_angle","right_ankle_angle","left_elbow_angle","right_elbow_angle",
    "left_shoulder_angle","right_shoulder_angle","torso_bend_angle",
    "wrist_angle_left","wrist_angle_right","left_knee_vel","right_knee_vel",
    "left_hip_vel","right_hip_vel","left_ankle_vel","right_ankle_vel",
    "left_elbow_vel","right_elbow_vel","left_shoulder_vel","right_shoulder_vel",
    "torso_bend_vel","shoulder_width","hip_width","torso_length","vertical_displacement"
]

def process_video(video_path, model_path, scaler_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Extracting keypoints...")
    kps = extract_all_keypoints(video_path)  # (T,33,4)
    T = kps.shape[0]
    print(f"Frames: {T}")

    # quick visibility summary
    vis_ratio = np.mean(kps[:,:,3] > 0.5)
    print(f"Landmark visibility ratio (fraction of points visible across frames): {vis_ratio:.3f}")
    if vis_ratio < 0.25:
        print("⚠ Warning: low overall landmark visibility. Ensure full body is in frame and lighting is good.")

    print("Extracting per-frame features...")
    feats = extract_features_seq(kps)  # (T,28)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    print("Loading model and scaler...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    # Sliding-window predictions: one prob per frame using window [i, i+TIMESTEPS)
    print("Computing sliding-window predictions...")
    probs = np.zeros(T, dtype=np.float32)
    last_valid = 0.0
    for i in range(T):
        window = feats[i:i+TIMESTEPS]
        window = pad_or_trim(window, TIMESTEPS)
        # safe scaling: replace any inf/nan and clip to reasonable range
        window_safe = np.nan_to_num(window, nan=0.0, posinf=1e6, neginf=-1e6)
        # try transform, but catch errors
        try:
            norm = scaler.transform(window_safe)
            # clip extreme values to avoid numerical blowups
            norm = np.clip(norm, -1e3, 1e3)
        except Exception as ex:
            # fallback: simple row-wise standardization
            mean = np.mean(window_safe, axis=0, keepdims=True)
            std = np.std(window_safe, axis=0, keepdims=True) + 1e-6
            norm = (window_safe - mean) / std

        # model expects (1, TIMESTEPS, 28)
        try:
            p_raw = model.predict(norm[np.newaxis, ...], verbose=0)
            # model may output shape (1,1) or (1,) etc.
            p = float(np.array(p_raw).reshape(-1)[0])
        except Exception as e:
            print("Model prediction error:", e)
            p = np.nan

        # ensure finite & in [0,1]
        if not np.isfinite(p):
            p = last_valid  # use last valid
        # if still nan (no last) -> 0.0
        if not np.isfinite(p):
            p = 0.0
        # clamp
        p = float(np.clip(p, 0.0, 1.0))
        probs[i] = p
        last_valid = p

    # small smoothing to remove spikes
    if len(probs) >= 3:
        probs = np.convolve(probs, np.ones(3)/3, mode='same')

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
        frame_kp = kps[frame_idx] if frame_idx < T else np.full((33,4), np.nan)
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
    risk_png, feat_png, all_feat_png = save_plots(plots_dir, probs, feats, FEATURE_NAMES)

    # PDF
    pdf_path = os.path.join(out_dir, "report.pdf")
    final_prob = float(probs[-1]) if len(probs)>0 else 0.0
    create_pdf_report(pdf_path, final_prob, risk_png, feat_png, all_feat_png, summary_text=f"Processed {T} frames. Visibility ratio {vis_ratio:.3f}")
    print("Report generated:")
    print(" - annotated video:", out_video_path)
    print(" - csv:", csv_path)
    print(" - plots:", risk_png, feat_png, all_feat_png)
    print(" - pdf:", pdf_path)
    return {
        "annotated_video": out_video_path,
        "csv": csv_path,
        "plots": (risk_png, feat_png, all_feat_png),
        "pdf": pdf_path,
        "final_prob": final_prob
    }

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=6, help="record duration seconds")
    parser.add_argument("--model", required=True, help="keras model .h5")
    parser.add_argument("--scaler", required=True, help="scaler .pkl (joblib)")
    parser.add_argument("--out_dir", default="report_output", help="output directory")
    args = parser.parse_args()

    # record
    input_video = "input_video.mp4"
    try:
        record_video(duration=args.duration, output_file=input_video, preview_file=os.path.join(args.out_dir, "preview.jpg"))
    except Exception as e:
        print("Recording error:", e)
        raise

    # process
    res = process_video(input_video, args.model, args.scaler, args.out_dir)
    print("\n================= DONE =================")
    print(res)
