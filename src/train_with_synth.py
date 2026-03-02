import os
import numpy as np # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Masking, LSTM, GRU, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Paths
REAL_NPZ = "/mnt/data/dataset_volleyball_sequence_scaled.npz"
SYN_NPY = "/mnt/data/synthetic_volleyball_timegan.npy"
OUT_MODEL = "/mnt/data/final_model_with_synth.h5"

USE_GRU = False  # set True to use GRU
EPOCHS = 60
BATCH_SIZE = 32

# Load real data
d = np.load(REAL_NPZ, allow_pickle=True)
X_real = d["X"].astype(np.float32)   # (N, T, F)
y_real = d["y"]
if y_real.ndim > 1 and y_real.shape[1] == 1:
    y_real = y_real.ravel()

# Load synthetic
X_synth = np.load(SYN_NPY).astype(np.float32)  # (M, T, F)

# Label synthetic by nearest neighbor in real dataset (mean pooling)
def seq_mean_repr(X):
    return X.mean(axis=1)  # (samples, features)

real_repr = seq_mean_repr(X_real)
synth_repr = seq_mean_repr(X_synth)

nbr = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(real_repr)
dist, idx = nbr.kneighbors(synth_repr)
idx = idx.ravel()
y_synth = y_real[idx]   # copy labels from nearest real sequence

# Concatenate
X_all = np.concatenate([X_real, X_synth], axis=0)
y_all = np.concatenate([y_real, y_synth], axis=0)
print("Combined shape:", X_all.shape, y_all.shape)

# Shuffle + split
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)

# If classification with integer labels -> one-hot
is_classification = np.issubdtype(y_all.dtype, np.integer) or np.unique(y_all).size <= 20
if is_classification:
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.transform(y_val)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1,1)
        y_val = y_val.reshape(-1,1)

timesteps = X_all.shape[1]
features = X_all.shape[2]

# Build LSTM/GRU model
def build_model(timesteps, features, use_gru=False, units=128, dropout=0.3, task='regression'):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    if use_gru:
        model.add(GRU(units, return_sequences=False))
    else:
        model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    if task == 'classification':
        model.add(Dense(y_train.shape[1], activation='softmax'))
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(Dense(1))
        loss = 'mse'
        metrics = ['mse', 'mae']
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model

task = 'classification' if is_classification else 'regression'
model = build_model(timesteps, features, use_gru=USE_GRU, task=task)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ModelCheckpoint("/mnt/data/best_model_with_synth.h5", monitor='val_loss', save_best_only=True)
]

hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

model.save(OUT_MODEL)
print("Saved model:", OUT_MODEL)

# Plot loss
plt.figure()
plt.plot(hist.history['loss'], label='train_loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.legend(); plt.grid(True)
plt.savefig("/mnt/data/training_loss_with_synth.png")
print("Saved loss plot:", "/mnt/data/training_loss_with_synth.png")
