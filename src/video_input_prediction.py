import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
import argparse
import os

# =============================
# SETTINGS
# =============================
TIMESTEPS = 32
mp_pose = mp.solutions.pose


# =====================================================
# 1) EXTRACT KEYPOINTS (T,33,4)
# =====================================================
def extract_all_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: " + video_path)

    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            arr = np.array([[p.x, p.y, p.z, p.visibility] for p in lm],
                           dtype=np.float32)
        else:
            arr = np.zeros((33, 4), dtype=np.float32)

        frames.append(arr)

    cap.release()
    return np.array(frames)


# =====================================================
# 2) MATH HELPERS
# =====================================================
def angle_3pts(a, b, c):
    """Angle ABC in radians."""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan

    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cosang, -1, 1))


def safe_angle(a, b, c):
    ang = angle_3pts(a, b, c)
    return 0.0 if np.isnan(ang) else ang


def vel(x):
    """Angular velocity"""
    return np.diff(x, append=x[-1])


def safe_dist(a, b):
    d = np.linalg.norm(a - b)
    return 0.0 if np.isnan(d) else d


def vertical_height(frame):
    L_HIP, R_HIP = 23, 24
    l, r = frame[L_HIP], frame[R_HIP]
    if np.any(np.isnan(l)) or np.any(np.isnan(r)):
        return 0.0
    return (l[1] + r[1]) / 2


# =====================================================
# 3) FEATURE EXTRACTION (28-D PER FRAME)
# =====================================================
def extract_features_seq(kps):
    T = kps.shape[0]
    xy = kps[:, :, :2]

    # --- Safe normalization ---
    norm = np.copy(xy)
    L_SH, L_HIP = 11, 23
    for i in range(T):
        ls, lh = xy[i, L_SH], xy[i, L_HIP]
        d = np.linalg.norm(ls - lh)
        if d < 1e-6:
            d = 1.0
        norm[i] = xy[i] / d

    k = norm

    # landmark indices
    idx = {
        "nose": 0,
        "l_sh": 11, "r_sh": 12,
        "l_elb": 13, "r_elb": 14,
        "l_wri": 15, "r_wri": 16,
        "l_hip": 23, "r_hip": 24,
        "l_kne": 25, "r_kne": 26,
        "l_ank": 27, "r_ank": 28
    }

    left_knee   = np.array([safe_angle(k[i][idx["l_hip"]], k[i][idx["l_kne"]], k[i][idx["l_ank"]]) for i in range(T)])
    right_knee  = np.array([safe_angle(k[i][idx["r_hip"]], k[i][idx["r_kne"]], k[i][idx["r_ank"]]) for i in range(T)])

    left_hip    = np.array([safe_angle(k[i][idx["l_sh"]], k[i][idx["l_hip"]], k[i][idx["l_kne"]]) for i in range(T)])
    right_hip   = np.array([safe_angle(k[i][idx["r_sh"]], k[i][idx["r_hip"]], k[i][idx["r_kne"]]) for i in range(T)])

    left_ank    = np.array([safe_angle(k[i][idx["l_kne"]], k[i][idx["l_ank"]], k[i][idx["l_wri"]]) for i in range(T)])
    right_ank   = np.array([safe_angle(k[i][idx["r_kne"]], k[i][idx["r_ank"]], k[i][idx["r_wri"]]) for i in range(T)])

    left_elb    = np.array([safe_angle(k[i][idx["l_sh"]], k[i][idx["l_elb"]], k[i][idx["l_wri"]]) for i in range(T)])
    right_elb   = np.array([safe_angle(k[i][idx["r_sh"]], k[i][idx["r_elb"]], k[i][idx["r_wri"]]) for i in range(T)])

    left_sho    = np.array([safe_angle(k[i][idx["nose"]], k[i][idx["l_sh"]], k[i][idx["l_elb"]]) for i in range(T)])
    right_sho   = np.array([safe_angle(k[i][idx["nose"]], k[i][idx["r_sh"]], k[i][idx["r_elb"]]) for i in range(T)])

    torso_bend  = np.array([safe_angle(k[i][idx["l_hip"]], k[i][idx["l_sh"]], k[i][idx["nose"]]) for i in range(T)])

    # wrist angles (safe)
    wrist_l     = np.array([safe_angle(k[i][idx["l_elb"]], k[i][idx["l_wri"]], k[i][idx["l_sh"]]) for i in range(T)])
    wrist_r     = np.array([safe_angle(k[i][idx["r_elb"]], k[i][idx["r_wri"]], k[i][idx["r_sh"]]) for i in range(T)])

    # velocities
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

    # distances
    shoulder_width = np.array([safe_dist(k[i][idx["l_sh"]], k[i][idx["r_sh"]]) for i in range(T)])
    hip_width      = np.array([safe_dist(k[i][idx["l_hip"]], k[i][idx["r_hip"]]) for i in range(T)])
    torso_length   = np.array([safe_dist(k[i][idx["l_sh"]], k[i][idx["l_hip"]]) for i in range(T)])

    vertical_disp  = np.array([vertical_height(k[i]) for i in range(T)])

    feats = np.column_stack([
        left_knee, right_knee, left_hip, right_hip,
        left_ank, right_ank, left_elb, right_elb,
        left_sho, right_sho, torso_bend, wrist_l, wrist_r,
        left_knee_vel, right_knee_vel, left_hip_vel, right_hip_vel,
        left_ank_vel, right_ank_vel, left_elb_vel, right_elb_vel,
        left_sho_vel, right_sho_vel, torso_vel,
        shoulder_width, hip_width, torso_length, vertical_disp
    ])

    return feats


# =====================================================
# 4) PAD/TRIM TO 32 FRAMES
# =====================================================
def pad_or_trim(feats, T=TIMESTEPS):
    if len(feats) == T:
        return feats
    if len(feats) > T:
        return feats[:T]
    pad = np.zeros((T - len(feats), feats.shape[1]), dtype=np.float32)
    return np.vstack([feats, pad])


# =====================================================
# 5) PREDICT FUNCTION
# =====================================================
def predict_video(video_path, model_path, scaler_path):
    print("Extracting keypoints...")
    kps = extract_all_keypoints(video_path)

    print("Extracting features...")
    feats = extract_features_seq(kps)
    feats = pad_or_trim(feats)
    feats = np.nan_to_num(feats)

    print("Loading model & scaler...")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    norm = scaler.transform(feats)
    prob = float(model.predict(norm[np.newaxis, ...], verbose=0).reshape(-1)[0])

    return prob


# =====================================================
# 6) MAIN
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="input_video.mp4")
    parser.add_argument("--model", default="injury_risk_model.h5")
    parser.add_argument("--scaler", default="scaler.pkl")
    args = parser.parse_args()

    prob = predict_video(args.video, args.model, args.scaler)

    print("\n====================================")
    print(f"Injury Risk: {prob*100:.2f}%")
    print("====================================\n")
