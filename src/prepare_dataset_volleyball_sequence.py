import os
import numpy as np
import pandas as pd

# ---- CONFIG ----
FEATURES_ROOT = "unsafe_features_data_volleyball"   # extracted features
OUTPUT_NPZ = "unsafe_dataset_volleyball_sequence.npz"
OUTPUT_CSV = "unsafe_dataset_volleyball_sequence.csv"

# Class labels
CLASS_LABELS = {
    "attack": 0,
    "block": 1,
    "defence": 2
}

# Sliding window params
WINDOW_LEN = 32
WINDOW_STRIDE = 16
# ----------------

def sliding_windows(arr, L=WINDOW_LEN, stride=WINDOW_STRIDE):
    """
    arr = (T, F)
    return list of (L, F)
    """
    T = arr.shape[0]
    if T < L:
        # Pad with zeros if too short
        pad = np.zeros((L - T, arr.shape[1]))
        return [np.vstack([arr, pad])]
    
    windows = []
    for start in range(0, T - L + 1, stride):
        windows.append(arr[start:start+L])
    return windows

def load_sequence_dataset():
    X = []
    y = []
    meta = []

    print("🏐 Building sliding window dataset from volleyball features...\n")

    for root, dirs, files in os.walk(FEATURES_ROOT):
        folder_name = os.path.basename(root).lower()

        # Only process folders that match class names
        if folder_name not in CLASS_LABELS:
            continue

        class_label = CLASS_LABELS[folder_name]

        for file in sorted(files):
            if not file.endswith(".npz"):
                continue

            fpath = os.path.join(root, file)

            try:
                arr = np.load(fpath)["features"]   # shape (T, F)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue

            windows = sliding_windows(arr)

            for w in windows:
                X.append(w)
                y.append(class_label)
                meta.append(f"{folder_name}/{file}")

            print(f"   🪟 {file}: {len(windows)} windows extracted.")

    X = np.array(X)  # (N, L, F)
    y = np.array(y)
    meta = np.array(meta)

    print(f"\n✅ Total windows: {X.shape[0]}")
    print(f"   Window shape: {X.shape[1:]} (L, F)")
    return X, y, meta


def save_dataset(X, y, meta):
    # Save as NPZ
    np.savez_compressed(OUTPUT_NPZ, X=X, y=y, meta=meta)
    print(f"\n💾 Saved NPZ → {OUTPUT_NPZ}")

    # Flatten windows for CSV (debugging)
    flat = X.reshape(X.shape[0], -1)
    df = pd.DataFrame(flat)
    df["label"] = y
    df["file"] = meta
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"📄 Saved CSV → {OUTPUT_CSV}")


if __name__ == "__main__":
    X, y, meta = load_sequence_dataset()
    save_dataset(X, y, meta)
    print("\n🎯 Volleyball sequence dataset is ready!")
