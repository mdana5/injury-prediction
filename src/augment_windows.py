# augment_windows.py
import numpy as np

IN_PATH = "dataset_volleyball_sequence_scaled.npz"
OUT_PATH = "dataset_volleyball_sequence_augmented.npz"
augment_factor = 3

d = np.load(IN_PATH)
X, y, meta = d["X"], d["y"], d["meta"]
N, L, F = X.shape

def add_noise(window, sigma=0.03):
    return window + np.random.normal(0, sigma, window.shape)


def time_warp(window, scale=0.9):
    """
    Time-warp a (L, F) window by stretching/compressing in time,
    then resample back to length L.
    """
    L, F = window.shape
    new_L = max(4, int(L * scale))  # minimum 4 frames

    # Stretch/compress
    orig_idx = np.linspace(0, 1, L)
    new_idx = np.linspace(0, 1, new_L)

    stretched = np.zeros((new_L, F))
    for f in range(F):
        stretched[:, f] = np.interp(new_idx, orig_idx, window[:, f])

    # Resample back to original L frames
    final = np.zeros((L, F))
    final_idx = np.linspace(0, 1, L)
    for f in range(F):
        final[:, f] = np.interp(final_idx, new_idx, stretched[:, f])

    return final


X_new = [X]
y_new = [y]
meta_new = [meta]

for i in range(N):
    win = X[i]
    label = y[i]
    m = meta[i]

    for a in range(augment_factor):
        if a % 2 == 0:
            aug = add_noise(win, sigma=0.02)
        else:
            aug = time_warp(win, scale=0.9 + 0.2 * np.random.rand())

        X_new.append(np.expand_dims(aug, 0))
        y_new.append(np.array([label]))
        meta_new.append(np.array([m + "_aug"]))

X_all = np.vstack(X_new)
y_all = np.concatenate(y_new)
meta_all = np.concatenate(meta_new)

np.savez_compressed(OUT_PATH, X=X_all, y=y_all, meta=meta_all)
print("Saved augmented:", OUT_PATH, "shape:", X_all.shape)
