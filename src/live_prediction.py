# live_prediction.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time
import argparse
import os
from collections import deque
from sklearn.preprocessing import StandardScaler

# ----------------------
# Config
# ----------------------
TIMESTEPS = 32
FEATURE_DIM = 28
PRED_SMOOTH_ALPHA = 0.3
FEATURE_SMOOTH_ALPHA = 0.25
FPS_SMOOTH_ALPHA = 0.8

# indices (MediaPipe)
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
NOSE = 0

# ----------------------
# Utilities
# ----------------------
def angle_3pts(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cosang = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

def angular_velocity(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    vel = np.diff(arr)
    vel = np.append(vel, 0.0)
    return vel

def vertical_height(frame):
    l = frame[L_HIP]; r = frame[R_HIP]
    if np.any(np.isnan(l)) or np.any(np.isnan(r)):
        return np.nan
    return ((l + r)/2.0)[1]

def ema(prev, value, alpha):
    if prev is None:
        return value
    return alpha * value + (1 - alpha) * prev

def ema_smooth_1d(arr, alpha=FEATURE_SMOOTH_ALPHA):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    out = arr.copy()
    for i in range(1, out.shape[0]):
        out[i] = alpha * out[i] + (1 - alpha) * out[i-1]
    return out

# ----------------------
# Keypoint extraction per frame
# ----------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints_from_results(results, frame_w, frame_h):
    if not results or not results.pose_landmarks:
        return np.full((33,2), np.nan, dtype=np.float32)
    lm = results.pose_landmarks.landmark
    coords = np.array([[p.x * frame_w, p.y * frame_h] for p in lm], dtype=np.float32)
    return coords

# ----------------------
# Full 28-feature extraction for sequence of frames
# frames_kps: list of (33,2) arrays (pixel coords)
# returns (T, 28)
# ----------------------
def extract_features_seq(landmarks_list):
    T = len(landmarks_list)
    if T == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)

    # stack into array (T,33,2)
    arr = np.stack(landmarks_list, axis=0).astype(np.float32)  # (T,33,2)

    # normalize per-frame by left_shoulder-left_hip distance when available
    norm = arr.copy()
    for i in range(T):
        ls = arr[i, L_SHOULDER]; lh = arr[i, L_HIP]
        if not np.any(np.isnan(ls)) and not np.any(np.isnan(lh)):
            d = np.linalg.norm(ls - lh)
            if d > 1e-6:
                norm[i] = arr[i] / d
    k = norm

    # angles
    left_knee = np.array([angle_3pts(k[i, L_HIP], k[i, L_KNEE], k[i, L_ANKLE]) for i in range(T)])
    right_knee = np.array([angle_3pts(k[i, R_HIP], k[i, R_KNEE], k[i, R_ANKLE]) for i in range(T)])
    left_hip = np.array([angle_3pts(k[i, L_SHOULDER], k[i, L_HIP], k[i, L_KNEE]) for i in range(T)])
    right_hip = np.array([angle_3pts(k[i, R_SHOULDER], k[i, R_HIP], k[i, R_KNEE]) for i in range(T)])
    left_ankle = np.array([angle_3pts(k[i, L_KNEE], k[i, L_ANKLE], k[i, L_ANKLE]) for i in range(T)])
    right_ankle = np.array([angle_3pts(k[i, R_KNEE], k[i, R_ANKLE], k[i, R_ANKLE]) for i in range(T)])
    left_elbow = np.array([angle_3pts(k[i, L_SHOULDER], k[i, L_ELBOW], k[i, L_WRIST]) for i in range(T)])
    right_elbow = np.array([angle_3pts(k[i, R_SHOULDER], k[i, R_ELBOW], k[i, R_WRIST]) for i in range(T)])
    left_shoulder = np.array([angle_3pts(k[i, NOSE], k[i, L_SHOULDER], k[i, L_ELBOW]) for i in range(T)])
    right_shoulder = np.array([angle_3pts(k[i, NOSE], k[i, R_SHOULDER], k[i, R_ELBOW]) for i in range(T)])
    torso_bend = np.array([angle_3pts(k[i, L_HIP], k[i, L_SHOULDER], k[i, NOSE]) for i in range(T)])
    wrist_left = np.array([angle_3pts(k[i, L_ELBOW], k[i, L_WRIST], k[i, L_WRIST]) for i in range(T)])
    wrist_right = np.array([angle_3pts(k[i, R_ELBOW], k[i, R_WRIST], k[i, R_WRIST]) for i in range(T)])

    # velocities for angles (take first 11 angles -> velocities shape (T,11))
    angle_stack = np.stack([
        left_knee, right_knee,
        left_hip, right_hip,
        left_ankle, right_ankle,
        left_elbow, right_elbow,
        left_shoulder, right_shoulder,
        torso_bend, wrist_left, wrist_right
    ], axis=1)  # (T,13)

    vel_cols = []
    for col in range(angle_stack.shape[1]):
        vel_cols.append(angular_velocity(angle_stack[:, col]))
    vel_arr = np.stack(vel_cols, axis=1)  # (T,13)
    vel_selected = vel_arr[:, :11]  # (T,11) matching chosen velocities

    # distances
    shoulder_width = np.array([np.linalg.norm(k[i, L_SHOULDER] - k[i, R_SHOULDER]) for i in range(T)])
    hip_width = np.array([np.linalg.norm(k[i, L_HIP] - k[i, R_HIP]) for i in range(T)])
    torso_length = np.array([np.linalg.norm(k[i, L_SHOULDER] - k[i, L_HIP]) for i in range(T)])
    vertical_disp = np.array([vertical_height(k[i]) for i in range(T)])

    # compose features: angles (13) + vel_selected (11) + distances (3) + vertical (1) = 28
    feats = np.column_stack([
        angle_stack,          # 13
        vel_selected,         # 11
        shoulder_width[:, None], hip_width[:, None], torso_length[:, None], vertical_disp[:, None]
    ])
    feats = feats.astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats  # (T,28)

# ----------------------
# pad / trim
# ----------------------
def pad_or_trim(seq, target=TIMESTEPS):
    T = seq.shape[0]
    if T == 0:
        return np.zeros((target, seq.shape[1]), dtype=np.float32)
    if T < target:
        last = seq[-1:]
        pad = np.repeat(last, target - T, axis=0)
        return np.concatenate([seq, pad], axis=0)
    return seq[:target]

# ----------------------
# Scaler loader / rebuild if mismatch
# ----------------------
def load_or_rebuild_scaler(scaler_path, training_npz_path=None):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        try:
            n_feats = int(getattr(scaler, "mean_").shape[0])
            if n_feats == FEATURE_DIM:
                return scaler
            else:
                print(f"[scaler] saved scaler expects {n_feats} features, need {FEATURE_DIM}.")
        except Exception:
            print("[scaler] invalid scaler on disk.")
    # fallback: fit a scaler on a tiny identity (so it won't fail) or training NPZ if provided
    if training_npz_path and os.path.exists(training_npz_path):
        d = np.load(training_npz_path)
        X = d["X"]  # (N,T,F)
        X_flat = X.reshape(-1, X.shape[2])
        scaler = StandardScaler().fit(X_flat)
        joblib.dump(scaler, scaler_path)
        print("[scaler] rebuilt from training NPZ.")
        return scaler
    # final fallback: create StandardScaler that will be fitted later (we'll fit on the first window)
    print("[scaler] no usable scaler found; will fit fallback during runtime.")
    return None

# ----------------------
# Drawing helpers
# ----------------------
def draw_skeleton(frame, results, joints_only=False):
    h, w = frame.shape[:2]
    if results and results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            if idx in (L_HIP, L_KNEE, L_ANKLE, L_SHOULDER, L_ELBOW, L_WRIST):
                color = (0, 200, 255)
            elif idx in (R_HIP, R_KNEE, R_ANKLE, R_SHOULDER, R_ELBOW, R_WRIST):
                color = (0, 255, 100)
            else:
                color = (200, 200, 200)
            if joints_only:
                cv2.circle(frame, (cx, cy), 3, color, -1)
        if not joints_only:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(125,125,125), thickness=1))

def draw_risk_meter(frame, risk, x=18, y=12, w=240, h=28):
    cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,50), -1)
    safe_r = 0.0 if (risk is None or not np.isfinite(risk)) else float(np.clip(risk, 0.0, 1.0))
    filled_w = int(w * safe_r)
    if safe_r < 0.5:
        color = (0,255,0)
    elif safe_r < 0.75:
        color = (0,200,255)
    else:
        color = (0,0,255)
    cv2.rectangle(frame, (x,y), (x+filled_w, y+h), color, -1)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (200,200,200), 1)
    txt = f"Risk: {safe_r*100:5.1f}%"
    cv2.putText(frame, txt, (x+6, y+h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def draw_feature_bars(frame, shoulder_val, knee_val, torso_val, x=18, y=48, box_w=140):
    # Expect values normalized 0..1
    vals = [shoulder_val, knee_val, torso_val]
    labs = ["Shoulder", "Knee", "Torso"]
    for i, (lab, val) in enumerate(zip(labs, vals)):
        yy = y + i*26
        cv2.rectangle(frame, (x,yy), (x+box_w, yy+20), (50,50,50), -1)
        cv2.rectangle(frame, (x,yy), (x+int(box_w*val), yy+20), (100,200,255), -1)
        cv2.putText(frame, lab, (x+box_w+8, yy+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ----------------------
# Main loop
# ----------------------
def run_realtime(model_path, scaler_path, training_npz=None, cam_index=0, video_file=None):
    # load model
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found: " + model_path)
    model = tf.keras.models.load_model(model_path)
    print("[model] Loaded:", model_path)

    scaler = load_or_rebuild_scaler(scaler_path, training_npz_path=training_npz)

    cap = cv2.VideoCapture(video_file if video_file else cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera/video")

    lm_buffer = deque(maxlen=TIMESTEPS)  # stores (33,2) per frame
    pred_ema = None
    last_pred = 0.0
    fps = None
    last_time = time.time()
    heatmap_mode = False
    joints_only = False
    auto_crop = False

    print("Controls: q -> quit | h -> toggle heatmap | j -> toggle joints-only | c -> toggle crop")
    print("Prediction begins after buffer has", TIMESTEPS, "frames.")

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, exiting.")
                break
            orig = frame.copy()
            h, w = frame.shape[:2]

            # Optional auto-crop (keeps display experience snappy)
            if auto_crop and len(lm_buffer) > 0:
                last = lm_buffer[-1]
                xs = last[:,0]; ys = last[:,1]
                xsv = xs[~np.isnan(xs)]; ysv = ys[~np.isnan(ys)]
                if xsv.size > 0:
                    x1 = max(int(xsv.min())-30, 0); x2 = min(int(xsv.max())+30, w)
                    y1 = max(int(ysv.min())-30, 0); y2 = min(int(ysv.max())+30, h)
                    frame = orig[y1:y2, x1:x2]
                else:
                    frame = orig

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            lm_xy = extract_keypoints_from_results(results, frame.shape[1], frame.shape[0])
            # if full NaN and we have previous, copy previous to avoid frames of all NaN
            if len(lm_buffer) > 0 and np.isnan(lm_xy).all():
                lm_xy = lm_buffer[-1].copy()
            lm_buffer.append(lm_xy)

            # Only predict when we have full buffer
            norm_seq = None
            if len(lm_buffer) == TIMESTEPS:
                lm_seq = list(lm_buffer)  # list of (33,2)
                feats = extract_features_seq(lm_seq)  # (TIMESTEPS,28)

                # sanitize and smooth
                feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
                for c in range(feats.shape[1]):
                    feats[:, c] = ema_smooth_1d(feats[:, c], alpha=FEATURE_SMOOTH_ALPHA)

                # ensure we have scaler, if None fit on this window (fallback)
                if scaler is None:
                    scaler = StandardScaler().fit(feats.reshape(-1, feats.shape[1]))
                    joblib.dump(scaler, scaler_path)
                    print("[scaler] fitted fallback scaler and saved to", scaler_path)

                # check scaler feature size
                if scaler is not None:
                    try:
                        if scaler.mean_.shape[0] != FEATURE_DIM:
                            raise ValueError("scaler-feature-mismatch")
                    except Exception:
                        # attempt rebuild from training NPZ if provided
                        if training_npz and os.path.exists(training_npz):
                            d = np.load(training_npz)
                            X = d["X"]; X_flat = X.reshape(-1, X.shape[2])
                            scaler = StandardScaler().fit(X_flat)
                            joblib.dump(scaler, scaler_path)
                            print("[scaler] rebuilt from training npz")
                        else:
                            scaler = StandardScaler().fit(feats.reshape(-1, feats.shape[1]))
                            joblib.dump(scaler, scaler_path)
                            print("[scaler] fallback scaler created")

                # normalize and predict
                norm_seq = scaler.transform(feats)  # (TIMESTEPS, 28)
                norm_seq = np.nan_to_num(norm_seq)
                seq_batch = norm_seq[np.newaxis, ...]  # (1,TIMESTEPS,28)
                try:
                    pred_raw = float(np.array(model.predict(seq_batch, verbose=0)).reshape(-1)[0])
                except Exception as e:
                    print("[model] predict error:", e)
                    pred_raw = 0.0
                if not np.isfinite(pred_raw):
                    pred_raw = 0.0
                pred_ema = ema(pred_ema, pred_raw, PRED_SMOOTH_ALPHA)
                last_pred = pred_ema

                # For UI bars: compute 3 normalized values (0..1)
                # Shoulder: normalized shoulder_width (use last frame, but scale to a sensible range)
                shoulder = norm_seq[-1, 24]  # shoulder_width (after scaler) -> but normalized; we want visualization in 0..1
                # knee: use left_knee angle (index 0)
                left_knee_ang = norm_seq[-1, 0]
                # torso: torso_bend_angle (index 10)
                torso_ang = norm_seq[-1, 10]

                # To map UI nicely: apply tanh-style mapping to scaled values
                def ui_map(v): 
                    # v is scaled by scaler -> could be large -> compress to 0..1
                    return float(1.0/(1.0 + np.exp(-v)))  # sigmoid mapping
                shoulder_ui = ui_map(shoulder)
                knee_ui = ui_map(left_knee_ang)
                torso_ui = ui_map(torso_ang)
            else:
                shoulder_ui = knee_ui = torso_ui = 0.0

            # Draw UI on original frame copy
            display = orig.copy()
            draw_skeleton(display, results, joints_only=joints_only)

            if len(lm_buffer) == TIMESTEPS and norm_seq is not None:
                draw_feature_bars(display, shoulder_ui, knee_ui, torso_ui, x=18, y=48)
            else:
                draw_feature_bars(display, 0,0,0, x=18, y=48)

            draw_risk_meter(display, last_pred if last_pred is not None else 0.0, x=18, y=12)
            # FPS
            now = time.time()
            dt = now - last_time
            last_time = now
            cur_fps = 1.0/dt if dt > 0 else 0.0
            fps = fps*FPS_SMOOTH_ALPHA + cur_fps*(1-FPS_SMOOTH_ALPHA) if fps is not None else cur_fps
            cv2.putText(display, f"FPS: {fps:4.1f}", (18, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(display, "q:quit  h:heatmap  j:joints  c:crop", (12, display.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)

            cv2.imshow("Real-time Injury Predictor", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('h'):
                heatmap_mode = not heatmap_mode
            if key == ord('j'):
                joints_only = not joints_only
            if key == ord('c'):
                auto_crop = not auto_crop

    cap.release()
    cv2.destroyAllWindows()

# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="injury_risk_model.h5", help="Path to trained LSTM model (.h5)")
    parser.add_argument("--scaler", default="scaler.pkl", help="Path to scaler.pkl (will be created if missing)")
    parser.add_argument("--training_npz", default="dataset_volleyball_sequence_scaled.npz",
                        help="Training NPZ used to rebuild scaler if needed")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--video_file", default=None, help="Optional video file path instead of webcam")
    args = parser.parse_args()

    run_realtime(model_path=args.model, scaler_path=args.scaler, training_npz=args.training_npz,
                 cam_index=args.cam, video_file=(args.video_file if args.video_file else None))
