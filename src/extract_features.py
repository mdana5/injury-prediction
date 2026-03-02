import os
import numpy as np
import pandas as pd
import math

INPUT_ROOT = "unsafe_smoothed_data"
OUTPUT_ROOT = "unsafe_features_data_volleyball"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# MediaPipe indices for easier reference
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

NOSE = 0
MID_SPINE = 24  # shoulder → hip mid approx

# ============ Utility functions ============

def angle_3pts(a, b, c):
    """Angle at b (in radians). a, b, c are (2,)"""
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

def joint_angle(fr, a, b, c):
    return angle_3pts(fr[a], fr[b], fr[c])

def angular_velocity(angles):
    vel = np.diff(angles)
    vel = np.append(vel, 0)  # pad to keep length
    return vel

def vertical_height(frame):
    """Use midpoint of hips as center of mass proxy"""
    l = frame[L_HIP]
    r = frame[R_HIP]
    if np.any(np.isnan(l)) or np.any(np.isnan(r)):
        return np.nan
    mid = (l + r) / 2.0
    return mid[1]  # y normalized height

# ============ Feature names ============
FEATURE_NAMES = [
    # Angles
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "torso_bend_angle",
    "wrist_angle_left", "wrist_angle_right",

    # Angular velocities
    "left_knee_vel", "right_knee_vel",
    "left_hip_vel", "right_hip_vel",
    "left_ankle_vel", "right_ankle_vel",
    "left_elbow_vel", "right_elbow_vel",
    "left_shoulder_vel", "right_shoulder_vel",
    "torso_bend_vel",

    # Distances / posture
    "shoulder_width", "hip_width", "torso_length",

    # Jump / COM data
    "vertical_displacement"
]


# ============ Feature extraction ============

def extract_volleyball_features(kps):
    """kps = (T, 33, 4)"""
    T = kps.shape[0]
    xy = kps[:, :, :2]

    # Torso length normalization
    norm = np.copy(xy)
    for i in range(T):
        ls, lh = xy[i, L_SHOULDER], xy[i, L_HIP]
        if not np.any(np.isnan(ls)) and not np.any(np.isnan(lh)):
            d = np.linalg.norm(ls - lh)
            if d > 1e-6:
                norm[i] = xy[i] / d
    k = norm  # easier variable

    # --- Angles per frame ---
    left_knee = np.array([joint_angle(k[i], L_HIP, L_KNEE, L_ANKLE) for i in range(T)])
    right_knee = np.array([joint_angle(k[i], R_HIP, R_KNEE, R_ANKLE) for i in range(T)])

    left_hip = np.array([joint_angle(k[i], L_SHOULDER, L_HIP, L_KNEE) for i in range(T)])
    right_hip = np.array([joint_angle(k[i], R_SHOULDER, R_HIP, R_KNEE) for i in range(T)])

    left_ankle = np.array([joint_angle(k[i], L_KNEE, L_ANKLE, L_ANKLE) for i in range(T)])
    right_ankle = np.array([joint_angle(k[i], R_KNEE, R_ANKLE, R_ANKLE) for i in range(T)])

    left_elbow = np.array([joint_angle(k[i], L_SHOULDER, L_ELBOW, L_WRIST) for i in range(T)])
    right_elbow = np.array([joint_angle(k[i], R_SHOULDER, R_ELBOW, R_WRIST) for i in range(T)])

    left_shoulder = np.array([joint_angle(k[i], NOSE, L_SHOULDER, L_ELBOW) for i in range(T)])
    right_shoulder = np.array([joint_angle(k[i], NOSE, R_SHOULDER, R_ELBOW) for i in range(T)])

    torso_bend = np.array([
        joint_angle(k[i], L_HIP, L_SHOULDER, NOSE) for i in range(T)
    ])

    wrist_left = np.array([joint_angle(k[i], L_ELBOW, L_WRIST, L_WRIST) for i in range(T)])
    wrist_right = np.array([joint_angle(k[i], R_ELBOW, R_WRIST, R_WRIST) for i in range(T)])

    # --- Angular velocities ---
    left_knee_vel   = angular_velocity(left_knee)
    right_knee_vel  = angular_velocity(right_knee)
    left_hip_vel    = angular_velocity(left_hip)
    right_hip_vel   = angular_velocity(right_hip)
    left_ankle_vel  = angular_velocity(left_ankle)
    right_ankle_vel = angular_velocity(right_ankle)
    left_elbow_vel  = angular_velocity(left_elbow)
    right_elbow_vel = angular_velocity(right_elbow)
    left_shoulder_vel  = angular_velocity(left_shoulder)
    right_shoulder_vel = angular_velocity(right_shoulder)
    torso_bend_vel = angular_velocity(torso_bend)

    # --- Distances ---
    shoulder_width = np.array([np.linalg.norm(k[i, L_SHOULDER] - k[i, R_SHOULDER]) for i in range(T)])
    hip_width      = np.array([np.linalg.norm(k[i, L_HIP] - k[i, R_HIP]) for i in range(T)])
    torso_length   = np.array([np.linalg.norm(k[i, L_SHOULDER] - k[i, L_HIP]) for i in range(T)])

    # --- Jump height (vertical displacement) ---
    vertical_disp = np.array([vertical_height(k[i]) for i in range(T)])

    # Combine into final (T, F)
    feats = np.column_stack([
        left_knee, right_knee,
        left_hip, right_hip,
        left_ankle, right_ankle,
        left_elbow, right_elbow,
        left_shoulder, right_shoulder,
        torso_bend,
        wrist_left, wrist_right,

        left_knee_vel, right_knee_vel,
        left_hip_vel, right_hip_vel,
        left_ankle_vel, right_ankle_vel,
        left_elbow_vel, right_elbow_vel,
        left_shoulder_vel, right_shoulder_vel,
        torso_bend_vel,

        shoulder_width, hip_width, torso_length,
        vertical_disp
    ])

    return feats  # shape (T, len(FEATURE_NAMES))


# ============ PROCESS ALL ============

def process_all():
    print("🏐 Starting VOLLEYBALL biomechanical feature extraction...\n")

    for root, dirs, files in os.walk(INPUT_ROOT):
        for fn in files:
            if not fn.endswith(".npz"):
                continue

            full_path = os.path.join(root, fn)
            rel_path = os.path.relpath(full_path, INPUT_ROOT)
            save_dir = os.path.join(OUTPUT_ROOT, os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)

            print(f"🎥 Processing {rel_path}")

            try:
                kps = np.load(full_path)["keypoints"]
            except Exception as e:
                print(f"❌ Error reading {fn}: {e}")
                continue

            feats = extract_volleyball_features(kps)

            out_npz = os.path.join(save_dir, fn)
            out_csv = out_npz.replace(".npz", ".csv")

            np.savez_compressed(out_npz, features=feats)
            pd.DataFrame(feats, columns=FEATURE_NAMES).to_csv(out_csv, index=False)

            print(f"   ✅ Saved NPZ → {out_npz}")
            print(f"   📄 Saved CSV → {out_csv}")

    print("\n🎯 Volleyball Feature Extraction Complete!")


if __name__ == "__main__":
    process_all()
