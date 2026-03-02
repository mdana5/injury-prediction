import numpy as np

d = np.load("dataset_combined_safe_unsafe.npz", allow_pickle=True)
X, y, meta = d["X"], d["y"], d["meta"]

print("Before cleaning:", np.isnan(X).sum(), "NaNs found")

# Replace NaN, INF, -INF with 0
X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Clip extreme values (avoid exploding gradients)
X_clean = np.clip(X_clean, -5, 5)

# Verify
print("After cleaning:", np.isnan(X_clean).sum(), "NaNs found")

np.savez_compressed("dataset_combined_clean.npz",
                    X=X_clean, y=y, meta=meta)

print("Saved → dataset_combined_clean.npz")
