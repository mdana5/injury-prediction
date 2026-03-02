#!/usr/bin/env python3
#run - python live_injury_report_pytorch.py --model lstm_classifier_best.pth --out_dir report_output --duration 10
"""
live_injury_report_pytorch_final.py

- 32-feature extractor matching training (17 base features + 2 vertical disp + 13 angular velocities)
- TIMESTEPS = 30
- Loads lstm_classifier_best.pth (state_dict)
- Reconstructs MinMaxScaler from uploaded normalized windows (fallback) or uses --scaler if provided
- Records webcam with visible preview, repairs keypoints, computes features, runs sliding-window inference
- Saves annotated video, CSV, plots, and multi-page PDF with all feature graphs
"""
import os
import time
import math
import argparse
import csv
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import torch
import torch.nn as nn

import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler

# -------------------------
TIMESTEPS = 30
FEATURE_NAMES = [
    "left_knee_angle","right_knee_angle","left_hip_angle","right_hip_angle",
    "left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle",
    "left_ankle_vert","right_ankle_vert","wrist_angle_left","wrist_angle_right",
    "torso_bend_angle","shoulder_width","hip_width","torso_length","mid_hip_y",
    "vert_disp_px","vert_disp_norm",
    "left_knee_vel","right_knee_vel","left_hip_vel","right_hip_vel",
    "left_elbow_vel","right_elbow_vel","left_shoulder_vel","right_shoulder_vel",
    "left_ankle_vert_vel","right_ankle_vert_vel","wrist_left_vel","wrist_right_vel","torso_bend_vel"
]
FEAT_DIM = len(FEATURE_NAMES)  # should be 32
mp_pose = mp.solutions.pose

# -------------------------
# LSTM model class (same as training)
# -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits.squeeze(1)

# -------------------------
# Infer model params from state_dict
# -------------------------
def infer_model_params_from_state(state_dict):
    for k in state_dict:
        if k.startswith("lstm.weight_ih_l0"):
            w = state_dict[k].cpu().numpy()
            out_dim, inp_dim = w.shape
            hidden = out_dim // 4
            input_size = inp_dim
            break
    else:
        raise RuntimeError("Could not find lstm.weight_ih_l0 in state_dict.")
    max_layer = 0
    for k in state_dict:
        if k.startswith("lstm.weight_ih_l"):
            idx = int(k.split("lstm.weight_ih_l")[1].split('.')[0])
            max_layer = max(max_layer, idx)
    num_layers = max_layer + 1
    return int(input_size), int(hidden), int(num_layers)

# -------------------------
# Recording with visible window
# -------------------------
def record_video(duration, output_file="input_video.mp4", preview_file="preview.jpg"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    print("✔ Webcam open: (w,h,fps)=", (w, h, fps))
    start = time.time()
    cv2.namedWindow("Recording...", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recording...", min(960, w), min(720, h))
    last_preview = 0
    while int(time.time() - start) < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ frame read failed")
            break
        out.write(frame)
        cv2.imshow("Recording...", frame)
        if time.time() - last_preview > 1.0:
            try:
                cv2.imwrite(preview_file, frame)
            except Exception:
                pass
            last_preview = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped by user.")
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("🎥 Recorded to:", output_file, " preview:", preview_file)
    return output_file, fps

# -------------------------
# Keypoint extraction & repair
# -------------------------
def extract_all_keypoints(video_path, min_visibility=0.4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.4, min_tracking_confidence=0.4)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            arr[arr[:,3] < min_visibility, :3] = np.nan
        else:
            arr = np.full((33,4), np.nan, dtype=np.float32)
        frames.append(arr)
    cap.release()
    kps = np.array(frames, dtype=np.float32)
    kps = repair_keypoints_timeseries(kps)
    return kps

def repair_keypoints_timeseries(kps):
    T, N, C = kps.shape
    repaired = kps.copy()
    for j in range(N):
        for c in range(3):
            series = repaired[:, j, c]
            mask = np.isfinite(series)
            if mask.sum() == 0:
                continue
            if mask.sum() == 1:
                val = series[mask][0]
                repaired[:, j, c] = val
                continue
            x = np.arange(T)
            repaired[:, j, c] = np.interp(x, x[mask], series[mask])
    vis = np.isfinite(repaired[:,:,0]) & np.isfinite(repaired[:,:,1])
    repaired[:,:,3] = vis.astype(np.float32)
    return repaired

# -------------------------
# Feature extraction EXACT matching training pipeline
# -------------------------
def angle_3pts(a, b, c):
    if a is None or b is None or c is None:
        return np.nan
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    return float(np.arccos(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)))

def fill_feature_nans(feats):
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
    out[~np.isfinite(out)] = 0.0
    return out

def extract_features_32(kps, fps):
    # kps: (T,33,4)
    T = kps.shape[0]
    xy = kps[:, :, :2].astype(np.float32)
    # normalization: center by hip midpoint and scale by hip distance when possible
    norm = np.copy(xy)
    L_SH, L_HIP = 11, 23
    R_SH, R_HIP = 12, 24
    for i in range(T):
        lh = xy[i, L_HIP]; rh = xy[i, R_HIP]
        if np.any(np.isnan(lh)) or np.any(np.isnan(rh)):
            ls = xy[i, L_SH]; rs = xy[i, R_SH]
            if not (np.any(np.isnan(ls)) or np.any(np.isnan(rs))):
                d = np.linalg.norm(ls - rs)
                if d > 1e-6:
                    norm[i] = (xy[i] - ls) / d
                else:
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
    # indexes
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

    # base angles and quantities (17 features)
    left_knee  = np.array([angle_3pts(get_pt(i,L_HIP), get_pt(i,L_KNE), get_pt(i,L_ANK)) for i in range(T)])
    right_knee = np.array([angle_3pts(get_pt(i,R_HIP), get_pt(i,R_KNE), get_pt(i,R_ANK)) for i in range(T)])
    left_hip   = np.array([angle_3pts(get_pt(i,L_SHO), get_pt(i,L_HIP), get_pt(i,L_KNE)) for i in range(T)])
    right_hip  = np.array([angle_3pts(get_pt(i,R_SHO), get_pt(i,R_HIP), get_pt(i,R_KNE)) for i in range(T)])
    left_elb   = np.array([angle_3pts(get_pt(i,L_SHO), get_pt(i,L_ELB), get_pt(i,L_WRI)) for i in range(T)])
    right_elb  = np.array([angle_3pts(get_pt(i,R_SHO), get_pt(i,R_ELB), get_pt(i,R_WRI)) for i in range(T)])
    left_sho   = np.array([angle_3pts(get_pt(i,NOSE), get_pt(i,L_SHO), get_pt(i,L_ELB)) for i in range(T)])
    right_sho  = np.array([angle_3pts(get_pt(i,NOSE), get_pt(i,R_SHO), get_pt(i,R_ELB)) for i in range(T)])
    # ankle vertical angle: angle between knee, ankle, and ankle+offset vertical
    left_ank_vert  = np.array([angle_3pts(get_pt(i,L_KNE), get_pt(i,L_ANK), get_pt(i,L_ANK)+offset) for i in range(T)])
    right_ank_vert = np.array([angle_3pts(get_pt(i,R_KNE), get_pt(i,R_ANK), get_pt(i,R_ANK)+offset) for i in range(T)])
    wrist_l = np.array([angle_3pts(get_pt(i,L_ELB), get_pt(i,L_WRI), get_pt(i,L_WRI)+offset) for i in range(T)])
    wrist_r = np.array([angle_3pts(get_pt(i,R_ELB), get_pt(i,R_WRI), get_pt(i,R_WRI)+offset) for i in range(T)])
    torso_bend = np.array([angle_3pts(get_pt(i,L_HIP), get_pt(i,L_SHO), get_pt(i,NOSE)) for i in range(T)])

    # widths and lengths
    def pair_dist(i,a,b):
        pa = k[i,a]; pb = k[i,b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            return np.nan
        return float(np.linalg.norm(pa - pb))

    shoulder_width = np.array([pair_dist(i, L_SHO, R_SHO) for i in range(T)])
    hip_width      = np.array([pair_dist(i, L_HIP, R_HIP) for i in range(T)])
    torso_length   = np.array([pair_dist(i, L_SHO, L_HIP) for i in range(T)])
    mid_hip_y      = np.array([ ( (k[i][L_HIP] + k[i][R_HIP]) / 2.0 )[1] if (not np.any(np.isnan(k[i][L_HIP])) and not np.any(np.isnan(k[i][R_HIP]))) else np.nan for i in range(T)])

    # vertical displacement (two features)
    # baseline = first frame mid_hip_y (standing)
    baseline = mid_hip_y[0] if np.isfinite(mid_hip_y[0]) else np.nanmean(mid_hip_y[np.isfinite(mid_hip_y)]) if np.any(np.isfinite(mid_hip_y)) else 0.0
    vert_disp_px = baseline - mid_hip_y
    vert_disp_norm = vert_disp_px / (torso_length + 1e-8)

    # velocities for 13 angle columns (in training order)
    angle_cols = [left_knee, right_knee, left_hip, right_hip,
                  left_elb, right_elb, left_sho, right_sho,
                  left_ank_vert, right_ank_vert, wrist_l, wrist_r, torso_bend]

    def vel_from_series(s):
        s = np.array(s, dtype=np.float32)
        if s.size == 0: return s
        d = np.diff(s, prepend=s[0]) * float(fps)
        d[~np.isfinite(d)] = 0.0
        return d

    left_knee_vel  = vel_from_series(left_knee)
    right_knee_vel = vel_from_series(right_knee)
    left_hip_vel   = vel_from_series(left_hip)
    right_hip_vel  = vel_from_series(right_hip)
    left_elb_vel   = vel_from_series(left_elb)
    right_elb_vel  = vel_from_series(right_elb)
    left_sho_vel   = vel_from_series(left_sho)
    right_sho_vel  = vel_from_series(right_sho)
    left_ank_vel   = vel_from_series(left_ank_vert)
    right_ank_vel  = vel_from_series(right_ank_vert)
    wrist_left_vel = vel_from_series(wrist_l)
    wrist_right_vel= vel_from_series(wrist_r)
    torso_bend_vel = vel_from_series(torso_bend)

    feats = np.column_stack([
        left_knee, right_knee, left_hip, right_hip,
        left_elb, right_elb, left_sho, right_sho,
        left_ank_vert, right_ank_vert, wrist_l, wrist_r,
        torso_bend, shoulder_width, hip_width, torso_length, mid_hip_y,
        vert_disp_px, vert_disp_norm,
        left_knee_vel, right_knee_vel, left_hip_vel, right_hip_vel,
        left_elb_vel, right_elb_vel, left_sho_vel, right_sho_vel,
        left_ank_vel, right_ank_vel, wrist_left_vel, wrist_right_vel, torso_bend_vel
    ])

    feats = fill_feature_nans(feats)
    return feats

# -------------------------
def pad_or_trim(feats, T=TIMESTEPS):
    if len(feats) == T: return feats
    if len(feats) > T: return feats[:T]
    pad_len = T - len(feats)
    pad = np.zeros((pad_len, feats.shape[1]), dtype=np.float32)
    return np.vstack([feats, pad])

# -------------------------
def draw_landmarks_overlay(frame, landmarks_33):
    h,w = frame.shape[:2]
    for p in landmarks_33:
        x,y = p[0], p[1]
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi, yi = int(x*w), int(y*h)
        cv2.circle(frame, (xi,yi), 3, (0,255,0), -1)
    skeleton_idx = [(11,13),(13,15),(12,14),(14,16),(11,12),(23,24),(11,23),(12,24),(25,27),(26,28),(23,25),(24,26)]
    for a,b in skeleton_idx:
        pa = landmarks_33[a]; pb = landmarks_33[b]
        if not (np.isfinite(pa[0]) and np.isfinite(pa[1]) and np.isfinite(pb[0]) and np.isfinite(pb[1])):
            continue
        xa,ya = int(pa[0]*w), int(pa[1]*h)
        xb,yb = int(pb[0]*w), int(pb[1]*h)
        cv2.line(frame, (xa,ya), (xb,yb), (192,192,0), 2)
    return frame

# -------------------------
def save_csv(out_csv_path, feats, probs):
    header = ["frame_index"] + FEATURE_NAMES + ["risk"]
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(feats.shape[0]):
            w.writerow([i] + feats[i].tolist() + [float(probs[i])])

# -------------------------
def save_plots(out_dir, probs, feats):
    os.makedirs(out_dir, exist_ok=True)
    frames = np.arange(len(probs))
    risk_png = os.path.join(out_dir, "risk_over_time.png")
    plt.figure(figsize=(8,3)); plt.plot(frames, probs); plt.xlabel("Frame"); plt.ylabel("Risk"); plt.title("Risk over time"); plt.tight_layout(); plt.savefig(risk_png); plt.close()
    feat_png = os.path.join(out_dir, "example_features.png")
    plt.figure(figsize=(8,3)); plt.plot(frames, feats[:,0], label=FEATURE_NAMES[0]); plt.plot(frames, feats[:,12], label=FEATURE_NAMES[12]); plt.plot(frames, feats[:,24], label=FEATURE_NAMES[24]); plt.legend(); plt.tight_layout(); plt.savefig(feat_png); plt.close()
    n = feats.shape[1]; cols = 4; rows = math.ceil(n/cols)
    plt.figure(figsize=(cols*3, rows*2.5))
    for i in range(n):
        ax = plt.subplot(rows, cols, i+1)
        ax.plot(frames, feats[:,i])
        ax.set_title(FEATURE_NAMES[i], fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
    plt.tight_layout()
    all_feat_png = os.path.join(out_dir, "all_features.png")
    plt.savefig(all_feat_png, dpi=150); plt.close()
    return risk_png, feat_png, all_feat_png

def create_pdf_report(pdf_path, final_prob, risk_png, feat_png, all_feat_png, summary_text=""):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w,h = letter
    c.setFont("Helvetica-Bold", 18); c.drawString(30, h-50, "Injury Prediction Report (PyTorch - final)")
    c.setFont("Helvetica", 12); c.drawString(30, h-80, f"Final predicted risk: {final_prob*100:.2f}%")
    if summary_text: c.drawString(30, h-100, summary_text)
    try: c.drawImage(risk_png, 30, h-200, width=520, height=120)
    except Exception: pass
    try: c.drawImage(feat_png, 30, h-340, width=520, height=120)
    except Exception: pass
    c.showPage()
    try: c.drawImage(all_feat_png, 20, 20, width=560, height=760)
    except Exception: pass
    c.showPage(); c.save()

# -------------------------
# Build/fit a MinMax scaler using uploaded normalized windows as best-effort reconstruction
def build_scaler_from_npz(npz_paths):
    # npz files likely already normalized; we fit scaler on these normalized windows so
    # we can transform live raw features into the same range roughly.
    arrays = []
    for p in npz_paths:
        if not os.path.exists(p): continue
        d = np.load(p)
        if "data" in d:
            arrays.append(d["data"].reshape(-1, d["data"].shape[-1]))
    if not arrays:
        return None
    all_arr = np.concatenate(arrays, axis=0)
    scaler = MinMaxScaler()
    scaler.fit(all_arr)  # fits on normalized data; used as best-effort mapping
    return scaler

# -------------------------
def process_video_final(video_path, model_path, npz_paths, scaler_path, out_dir, device):
    os.makedirs(out_dir, exist_ok=True)
    print("Extracting keypoints...")
    kps = extract_all_keypoints(video_path)
    T = kps.shape[0]
    print("Frames:", T)

    vis_ratio = np.mean(kps[:,:,3] > 0.5)
    print("Landmark visibility ratio:", vis_ratio)
    if vis_ratio < 0.25:
        print("⚠ Low visibility. Ensure full-body in frame and good lighting.")

    print("Extracting features...")
    feats = extract_features_32(kps, fps=30)  # produce (T,32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # get scaler: prefer passed scaler_path, else build from npz training windows
    if scaler_path and os.path.exists(scaler_path):
        print("Loading provided scaler:", scaler_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Reconstructing scaler from training npz files (best-effort).")
        scaler = build_scaler_from_npz(npz_paths)
        if scaler is None:
            print("No training npz found; using per-window minmax fallback.")
    # load model
    print("Loading PyTorch model...")
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and not any(k.startswith('lstm.') for k in state.keys()):
        possible = None
        for candidate in ['state_dict', 'model_state', 'model']:
            if candidate in state and isinstance(state[candidate], dict):
                possible = state[candidate]; break
        state_dict = possible if possible is not None else state
    else:
        state_dict = state
    try:
        in_sz, hidden_sz, num_layers = infer_model_params_from_state(state_dict)
        if in_sz != FEAT_DIM:
            print(f"Warning: inferred input_size={in_sz} != FEAT_DIM={FEAT_DIM}")
    except Exception as e:
        print("Could not infer params:", e)
        hidden_sz = 128; num_layers = 1
    model = LSTMClassifier(feat_dim=FEAT_DIM, hidden_dim=hidden_sz, num_layers=num_layers).to(device)
    # strip module prefix if present
    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.to(device).eval()

    # sliding-window predictions
    print("Computing sliding-window predictions (window size = 30)...")
    probs = np.zeros(T, dtype=np.float32)
    last_valid = 0.0
    for i in range(T):
        window = feats[i:i+TIMESTEPS]
        window = pad_or_trim(window, TIMESTEPS)  # (30,32)
        # scale with scaler (best-effort)
        try:
            if scaler is not None:
                norm_rows = scaler.transform(window)
            else:
                mn = np.nanmin(window, axis=0); mx = np.nanmax(window, axis=0)
                denom = (mx - mn); denom[denom == 0] = 1.0
                norm_rows = (window - mn) / denom
        except Exception as e:
            mn = np.nanmin(window, axis=0); mx = np.nanmax(window, axis=0)
            denom = (mx - mn); denom[denom == 0] = 1.0
            norm_rows = (window - mn) / denom

        norm_rows = np.nan_to_num(norm_rows, nan=0.0, posinf=1e6, neginf=-1e6)
        norm_rows = np.clip(norm_rows, -1e3, 1e3)
        x = torch.from_numpy(norm_rows.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            p = float(torch.sigmoid(logits).cpu().numpy().reshape(-1)[0])
        if not np.isfinite(p):
            p = last_valid
        if not np.isfinite(p):
            p = 0.0
        p = float(np.clip(p, 0.0, 1.0))
        probs[i] = p
        last_valid = p

    if len(probs) >= 3:
        probs = np.convolve(probs, np.ones(3)/3, mode='same')

    # annotated video
    print("Writing annotated video...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    out_video = os.path.join(out_dir, "annotated_pytorch_final.mp4")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w,h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_kp = kps[idx] if idx < T else np.full((33,4), np.nan, dtype=np.float32)
        frame = draw_landmarks_overlay(frame, frame_kp)
        cv2.putText(frame, f"Risk: {probs[idx]*100:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()

    # csv/plots/pdf
    csv_path = os.path.join(out_dir, "features_and_risk_pytorch_final.csv")
    save_csv(csv_path, feats, probs)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    risk_png, feat_png, all_png = save_plots(plots_dir, probs, feats)
    pdf_path = os.path.join(out_dir, "report_pytorch_final.pdf")
    final_prob = float(probs[-1]) if len(probs)>0 else 0.0
    create_pdf_report(pdf_path, final_prob, risk_png, feat_png, all_png, summary_text=f"Frames={T}, vis_ratio={np.mean(kps[:,:,3]>0.5):.3f}")
    print("Done. Outputs:")
    print("Annotated video:", out_video)
    print("CSV:", csv_path)
    print("Plots:", risk_png, feat_png, all_png)
    print("PDF:", pdf_path)
    return {"video": out_video, "csv": csv_path, "pdf": pdf_path}

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to lstm_classifier_best.pth")
    parser.add_argument("--scaler", default="", help="(optional) path to original MinMaxScaler .pkl")
    parser.add_argument("--out_dir", default="report_output")
    parser.add_argument("--duration", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    # recording
    try:
        video_file, rec_fps = record_video(args.duration, "input_video.mp4", preview_file=os.path.join(args.out_dir,"preview.jpg"))
    except Exception as e:
        print("Recording error:", e)
        raise
    # build npz list for scaler reconstruction
    npz_list = ["attack_windows_norm.npz","block_windows_norm.npz","defence_windows_norm.npz","all_actions_safe.npz"]
    npz_list = [p for p in npz_list if os.path.exists(p)]
    res = process_video_final(video_file, args.model, npz_list, args.scaler if args.scaler else None, args.out_dir, device)
    print("\n=== FINISHED ===")
    print(res)
