import os
import cv2
import mediapipe as mp
import numpy as np

# Root folder containing attack/block/defence
input_root = os.path.join("datasett", "front_view")
output_dir = "unsafe_keypoints_data"
os.makedirs(output_dir, exist_ok=True)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

print("🚀 Script started successfully")

# Recursively collect videos from ALL subfolders under front_view
video_files = []
for root, dirs, files in os.walk(input_root):
    for f in files:
        if f.lower().endswith((".mp4", ".mov", ".avi")):
            video_files.append(os.path.join(root, f))

if not video_files:
    print("⚠️ No video files found under subfolders in", input_root)
    exit()

print(f"✅ Found {len(video_files)} videos\n")

for video_path in video_files:
    video_name = os.path.basename(video_path)
    print(f"🎥 Processing: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_name}")
        continue

    all_kps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kps = np.full((33, 4), np.nan, dtype=np.float32)

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                kps[i] = [lm.x, lm.y, lm.z, lm.visibility]

        all_kps.append(kps)

    cap.release()
    all_kps = np.array(all_kps)

    # Save keypoints using same folder structure (attack/block/defence)
    rel_path = os.path.relpath(video_path, input_root)  # e.g., attack/A1.mp4
    save_subfolder = os.path.dirname(rel_path)
    final_output_dir = os.path.join(output_dir, save_subfolder)
    os.makedirs(final_output_dir, exist_ok=True)

    save_path = os.path.join(
        final_output_dir,
        f"{os.path.splitext(video_name)[0]}.npz"
    )
    
    np.savez_compressed(save_path, keypoints=all_kps)

    print(f"   ✅ Saved: {save_path}  ({all_kps.shape})")

print("\n🎯 Finished processing ALL videos successfully.")
