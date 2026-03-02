import numpy as np

safe = np.load("dataset_volleyball_sequence_scaled.npz")
unsafe = np.load("unsafe_dataset_volleyball_sequence_scaled.npz")

X_safe, meta_safe = safe["X"], safe["meta"]
X_unsafe, meta_unsafe = unsafe["X"], unsafe["meta"]

y_safe = np.zeros(len(X_safe))
y_unsafe = np.ones(len(X_unsafe))

X_all = np.concatenate([X_safe, X_unsafe], axis=0)
y_all = np.concatenate([y_safe, y_unsafe], axis=0)
meta_all = np.concatenate([meta_safe, meta_unsafe], axis=0)

print("Final dataset shape:", X_all.shape, y_all.shape)

np.savez_compressed("dataset_combined_safe_unsafe.npz",
                    X=X_all, y=y_all, meta=meta_all)

print("Saved → dataset_combined_safe_unsafe.npz")
