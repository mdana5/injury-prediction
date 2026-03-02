# normalize_windows.py
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

IN_PATH = "unsafe_dataset_volleyball_sequence.npz"
OUT_PATH = "unsafe_dataset_volleyball_sequence_scaled.npz"
SCALER_PATH = "scaler_windows.pkl"

d = np.load(IN_PATH)
X, y, meta = d["X"], d["y"], d["meta"]   # X: (N,L,F)
N, L, F = X.shape
print("Loaded X shape:", X.shape)

# reshape to (N*L, F) then scale
X_flat = X.reshape(-1, F)
scaler = StandardScaler()
X_flat_scaled = scaler.fit_transform(X_flat)

# reshape back to (N,L,F)
X_scaled = X_flat_scaled.reshape(N, L, F)

# save
np.savez_compressed(OUT_PATH, X=X_scaled, y=y, meta=meta)
joblib.dump(scaler, SCALER_PATH)
print("Saved scaled dataset:", OUT_PATH)
print("Saved scaler:", SCALER_PATH)
