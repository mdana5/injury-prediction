import os
import numpy as np
from scipy.signal import savgol_filter

# Input (with attack/block/defence)
INPUT_ROOT = "unsafe_keypoints_data"
OUTPUT_ROOT = "unsafe_smoothed_data"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def smooth_keypoints(kps, window=7, poly=2):
    """Smooth keypoint trajectories for x, y, z using Savitzky-Golay filter."""
    smoothed = np.copy(kps)
    for j in range(kps.shape[1]):  # 33 joints
        for d in range(3):  # x, y, z (skip confidence channel)
            coords = kps[:, j, d]
            mask = ~np.isnan(coords)
            if mask.sum() > window:
                try:
                    smoothed[mask, j, d] = savgol_filter(coords[mask], window, poly)
                except Exception:
                    pass
    return smoothed

print("🚀 Starting recursive smoothing of ALL keypoint files...")

# Walk through attack/block/defence
for root, dirs, files in os.walk(INPUT_ROOT):
    for file in files:
        if not file.endswith(".npz"):
            continue

        full_path = os.path.join(root, file)

        # relative path (e.g., "attack/A1.npz")
        rel_path = os.path.relpath(full_path, INPUT_ROOT)
        save_subfolder = os.path.dirname(rel_path)

        # create output subfolder
        out_dir = os.path.join(OUTPUT_ROOT, save_subfolder)
        os.makedirs(out_dir, exist_ok=True)

        print(f"🎥 Processing: {rel_path}")

        # load keypoints
        try:
            kps = np.load(full_path)["keypoints"]
        except Exception as e:
            print(f"❌ Could not load {rel_path}: {e}")
            continue

        smoothed = smooth_keypoints(kps)

        # save to same subfolder structure
        save_path = os.path.join(out_dir, file)
        np.savez_compressed(save_path, keypoints=smoothed)

        print(f"   ✅ Saved smoothed: {save_path} ({smoothed.shape})")

print("\n🎯 Smoothing completed for ALL keypoint files.")
