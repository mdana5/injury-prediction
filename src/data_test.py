import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# -------------------------
# Load Files
# -------------------------
aug_npz = np.load("dataset_volleyball_sequence_scaled.npz")
gan_data = np.load("synthetic_volleyball_timegan.npy")

# Extract augmented
if "data" in aug_npz:
    aug_data = aug_npz["data"]
elif "X" in aug_npz:
    aug_data = aug_npz["X"]
else:
    aug_data = aug_npz[list(aug_npz.keys())[0]]

# Flatten both
aug_flat = aug_data.reshape(len(aug_data), -1)
gan_flat = gan_data.reshape(len(gan_data), -1)

print("Original shapes:")
print("Augmented:", aug_flat.shape)
print("GAN:      ", gan_flat.shape)

# -------------------------
# 1. DROP columns that are fully NaN in either dataset
# -------------------------
aug_valid = ~np.all(np.isnan(aug_flat), axis=0)
gan_valid = ~np.all(np.isnan(gan_flat), axis=0)

# columns valid in BOTH datasets
common_valid = aug_valid & gan_valid

aug_flat = aug_flat[:, common_valid]
gan_flat = gan_flat[:, common_valid]

print("After keeping common valid columns:")
print("Augmented:", aug_flat.shape)
print("GAN:      ", gan_flat.shape)

# -------------------------
# 2. Impute remaining NaNs
# -------------------------
imputer = SimpleImputer(strategy="mean")
aug_flat = imputer.fit_transform(aug_flat)
gan_flat = imputer.fit_transform(gan_flat)

# -------------------------
# 3. Reduce BOTH datasets to same dimension (e.g., 50)
# -------------------------
target_dim = 50
pca_align = PCA(n_components=target_dim)

combined = np.vstack([aug_flat, gan_flat])
aligned = pca_align.fit_transform(combined)

aug_pca = aligned[:len(aug_flat)]
gan_pca = aligned[len(aug_flat):]

# -------------------------
# 4. Final PCA for 2D visualization
# -------------------------
pca2 = PCA(n_components=2)
vis = pca2.fit_transform(np.vstack([aug_pca, gan_pca]))

aug_vis = vis[:len(aug_pca)]
gan_vis = vis[len(aug_pca):]

# -------------------------
# 5. Plot PCA Scatter
# -------------------------
plt.figure(figsize=(8,6))
plt.scatter(aug_vis[:,0], aug_vis[:,1], alpha=0.5, label="Augmented Data")
plt.scatter(gan_vis[:,0], gan_vis[:,1], alpha=0.5, label="GAN Data")
plt.title("PCA: Augmented vs GAN Generated Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# 6. Histogram Comparison
# -------------------------
plt.figure(figsize=(8,6))
plt.hist(aug_pca.flatten(), bins=50, alpha=0.5, label="Augmented Data")
plt.hist(gan_pca.flatten(), bins=50, alpha=0.5, label="GAN Data")
plt.title("Distribution Comparison (PCA Reduced Features)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()
