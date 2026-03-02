# timegan_volleyball_full.py
"""
Stable TimeGAN for Volleyball sequences (pretrain + full training).
Designed for small dataset N ~ 280 and total 1500 epochs:
 - PRETRAIN_EPOCHS = 300  (embedder+recovery)
 - GAN_EPOCHS = 1200      (full TimeGAN)
Outputs:
 - checkpoints under /mnt/data/timegan_checkpoints/
 - final synthetic sequences at /mnt/data/synthetic_volleyball_timegan.npy
Data file expected (already uploaded): /mnt/data/dataset_volleyball_sequence_scaled.npz
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, InputLayer
from tensorflow.keras.optimizers import Adam

# --- CONFIG ---
DATA_PATH = "/mnt/data/dataset_volleyball_sequence_scaled.npz"   # <-- your uploaded file path
OUT_PATH = "/mnt/data/synthetic_volleyball_timegan.npy"
CHECKPOINT_DIR = "/mnt/data/timegan_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training schedule
PRETRAIN_EPOCHS = 300
GAN_EPOCHS = 1200
TOTAL_EPOCHS = PRETRAIN_EPOCHS + GAN_EPOCHS

# Hyperparams
LEARNING_RATE = 1e-4
BATCH_SIZE = None  # will set to min(128, N)
HIDDEN_DIM = None  # will set to features
PRINT_EVERY_PRE = 50
PRINT_EVERY_GAN = 100

# -----------------------
# Load & inspect data
# -----------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

d = np.load(DATA_PATH, allow_pickle=True)
print("NPZ keys:", d.files)

# Heuristic: prefer arrays named 'X' and 'y'
if 'X' in d:
    X = d['X']
else:
    # fallback: take first array with 3 dims
    cand = [k for k in d.files if d[k].ndim == 3]
    if len(cand) == 0:
        raise RuntimeError("No 3D array found in .npz. Expecting shape (N, seq_len, features). Keys: " + ", ".join(d.files))
    X = d[cand[0]]
    print("Using key:", cand[0], "as X")

# replace NaN/Inf and normalize (already scaled but safe)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
minv = X.min()
maxv = X.max()
if maxv - minv > 1e-8:
    X = (X - minv) / (maxv - minv)
else:
    X = X - minv  # constant data -> zero it

N, seq_len, features = X.shape
print(f"Loaded X shape: N={N}, seq_len={seq_len}, features={features}")

# safe batch size
BATCH_SIZE = min(128, N)
print("Using batch size:", BATCH_SIZE)

# set hidden dim
HIDDEN_DIM = features
print("Hidden dim set to features:", HIDDEN_DIM)

# -----------------------
# Helper: random noise Z
# -----------------------
def random_Z(batch):
    # Normal noise in latent dimension per timestep
    return np.random.normal(size=(batch, seq_len, HIDDEN_DIM)).astype(np.float32)

# -----------------------
# Network builders (with TimeDistributed Dense)
# -----------------------
def build_embedder():
    model = Sequential(name="Embedder")
    model.add(InputLayer(input_shape=(seq_len, features)))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(HIDDEN_DIM)))
    return model

def build_recovery():
    model = Sequential(name="Recovery")
    model.add(InputLayer(input_shape=(seq_len, HIDDEN_DIM)))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    return model

def build_generator():
    model = Sequential(name="Generator")
    model.add(InputLayer(input_shape=(seq_len, HIDDEN_DIM)))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(HIDDEN_DIM)))
    return model

def build_supervisor():
    model = Sequential(name="Supervisor")
    model.add(InputLayer(input_shape=(seq_len, HIDDEN_DIM)))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(HIDDEN_DIM)))
    return model

def build_discriminator():
    model = Sequential(name="Discriminator")
    model.add(InputLayer(input_shape=(seq_len, HIDDEN_DIM)))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    return model

# instantiate networks
embedder = build_embedder()
recovery = build_recovery()
generator = build_generator()
supervisor = build_supervisor()
discriminator = build_discriminator()

# print summaries (short)
print(embedder.summary())
print(recovery.summary())
print(generator.summary())
print(supervisor.summary())
print(discriminator.summary())

# -----------------------
# Optimizers (separate)
# -----------------------
embedder_opt = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
recovery_opt = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
generator_opt = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
supervisor_opt = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
discriminator_opt = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Utility to save checkpoints
def save_checkpoint(epoch):
    prefix = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:05d}")
    embedder.save_weights(prefix + "_embedder.h5")
    recovery.save_weights(prefix + "_recovery.h5")
    generator.save_weights(prefix + "_generator.h5")
    supervisor.save_weights(prefix + "_supervisor.h5")
    discriminator.save_weights(prefix + "_discriminator.h5")
    print("Saved checkpoint:", prefix + "_*.h5")

# -----------------------
# Pretraining: Embedder + Recovery
# -----------------------
print(f"Starting pretraining Embedder+Recovery for {PRETRAIN_EPOCHS} epochs...")
for pre_epoch in range(1, PRETRAIN_EPOCHS + 1):
    idx = np.random.randint(0, N, BATCH_SIZE)
    X_batch = X[idx].astype(np.float32)

    with tf.GradientTape() as tape:
        H = embedder(X_batch, training=True)              # (batch, seq_len, HIDDEN_DIM)
        X_tilde = recovery(H, training=True)              # (batch, seq_len, features)
        E_loss = tf.reduce_mean(tf.square(X_batch - X_tilde))

    grads = tape.gradient(E_loss, embedder.trainable_weights + recovery.trainable_weights)
    # split gradients to respective optimizers
    n_embed = len(embedder.trainable_weights)
    embedder_grads = grads[:n_embed]
    recovery_grads = grads[n_embed:]

    embedder_opt.apply_gradients(zip(embedder_grads, embedder.trainable_weights))
    recovery_opt.apply_gradients(zip(recovery_grads, recovery.trainable_weights))

    if pre_epoch % PRINT_EVERY_PRE == 0 or pre_epoch == 1:
        print(f"[Pretrain {pre_epoch}/{PRETRAIN_EPOCHS}] E_loss={E_loss.numpy():.6f}")

    # checkpoint occasionally
    if pre_epoch % 200 == 0:
        save_checkpoint(pre_epoch)

print("Pretraining done.")

# -----------------------
# Full TimeGAN training
# -----------------------
print(f"Starting full TimeGAN training for {GAN_EPOCHS} epochs...")
for epoch in range(1, GAN_EPOCHS + 1):
    # sample real batch
    idx = np.random.randint(0, N, BATCH_SIZE)
    X_batch = X[idx].astype(np.float32)

    # 1) train embedder + recovery (reconstruction)
    with tf.GradientTape() as tape:
        H = embedder(X_batch, training=True)
        X_tilde = recovery(H, training=True)
        E_loss = tf.reduce_mean(tf.square(X_batch - X_tilde))
    grads = tape.gradient(E_loss, embedder.trainable_weights + recovery.trainable_weights)
    n_embed = len(embedder.trainable_weights)
    embedder_opt.apply_gradients(zip(grads[:n_embed], embedder.trainable_weights))
    recovery_opt.apply_gradients(zip(grads[n_embed:], recovery.trainable_weights))

    # 2) train generator + supervisor (adversarial + supervised)
    Z = random_Z(BATCH_SIZE)
    with tf.GradientTape() as tape:
        H_fake = generator(Z, training=True)                     # (batch, seq_len, H)
        H_hat_super = supervisor(H_fake, training=True)          # (batch, seq_len, H)
        Y_fake = discriminator(H_hat_super, training=True)       # (batch, seq_len, 1)

        # adversarial loss (make discriminator predict ones)
        G_loss_U = bce(tf.ones_like(Y_fake), Y_fake)
        # supervised loss (match H_hat_super to H_fake)
        G_loss_S = tf.reduce_mean(tf.square(H_hat_super - H_fake))
        # total generator loss (weighted)
        G_loss = G_loss_U + 0.1 * G_loss_S

    g_vars = generator.trainable_weights + supervisor.trainable_weights
    g_grads = tape.gradient(G_loss, g_vars)
    n_gen = len(generator.trainable_weights)
    generator_opt.apply_gradients(zip(g_grads[:n_gen], generator.trainable_weights))
    supervisor_opt.apply_gradients(zip(g_grads[n_gen:], supervisor.trainable_weights))

    # 3) train discriminator
    with tf.GradientTape() as tape:
        Y_real = discriminator(embedder(X_batch, training=False), training=True)  # (batch, seq_len, 1)
        Y_fake_for_D = discriminator(H_hat_super, training=True)
        D_loss_real = bce(tf.ones_like(Y_real), Y_real)
        D_loss_fake = bce(tf.zeros_like(Y_fake_for_D), Y_fake_for_D)
        D_loss = D_loss_real + D_loss_fake
    d_grads = tape.gradient(D_loss, discriminator.trainable_weights)
    discriminator_opt.apply_gradients(zip(d_grads, discriminator.trainable_weights))

    # logging
    if (epoch % PRINT_EVERY_GAN == 0) or (epoch == 1):
        # compute numeric scalars (may be expensive; ok for periodic logging)
        print(f"[GAN {epoch}/{GAN_EPOCHS}] E={E_loss.numpy():.6f} G={G_loss.numpy():.6f} D={D_loss.numpy():.6f}")

    # periodic checkpoint
    if epoch % 200 == 0:
        save_checkpoint(pre_epoch + epoch)

print("Full GAN training complete.")

# -----------------------
# Generate synthetic sequences
# -----------------------
SYN_COUNT = 500  # number of synthetic sequences to generate
print("Generating synthetic sequences count:", SYN_COUNT)
Z = random_Z(SYN_COUNT)
H_fake = generator(Z, training=False)
H_out = supervisor(H_fake, training=False)
SYN = recovery(H_out, training=False).numpy()

np.save(OUT_PATH, SYN)
print("Saved synthetic data to:", OUT_PATH)
print("Synthetic shape:", SYN.shape)
