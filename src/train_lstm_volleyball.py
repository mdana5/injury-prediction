# train_lstm_volleyball.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --------- CONFIG ----------
DATA_FILE = "/mnt/data/dataset_volleyball_sequence_scaled.npz"  # change if needed
USE_GRU = False  # set True to use GRU instead of LSTM
EPOCHS = 60
BATCH_SIZE = 32
# --------------------------

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Upload file or change DATA_FILE path.")

npz = np.load(DATA_FILE, allow_pickle=True)
print("Keys in .npz:", list(npz.keys()))

# Heuristic to pick X and y
def pick_array(npz, prefer='X'):
    keys = list(npz.keys())
    if prefer in npz:
        return npz[prefer]
    # fallback heuristics
    for k in keys:
        kl = k.lower()
        if 'x' in kl or 'data' in kl or 'features' in kl:
            return npz[k]
    raise KeyError("Could not find features array (X). Keys: " + ", ".join(keys))

def pick_label(npz):
    keys = list(npz.keys())
    if 'y' in npz:
        return npz['y']
    for k in keys:
        kl = k.lower()
        if 'y' in kl or 'label' in kl or 'target' in kl:
            return npz[k]
    raise KeyError("Could not find labels/targets in .npz. Keys: " + ", ".join(keys))

X = pick_array(npz)
y = pick_label(npz)

print("X shape:", X.shape)
print("y shape:", y.shape, "dtype:", y.dtype)

# flatten y if needed
if y.ndim > 1 and y.shape[1] == 1:
    y = y.ravel()

# detect classification vs regression
task_type = "regression"
if np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
    n_unique = np.unique(y).size
    if n_unique <= 20:
        task_type = "classification"
else:
    # if numeric but few unique values maybe classification
    if np.unique(y).size <= 20:
        task_type = "classification"

print("Detected task:", task_type)

if task_type == "classification":
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    if y_enc.ndim == 1:
        y_enc = y_enc.reshape(-1,1)
    y = y_enc
    print("y one-hot shape:", y.shape)

# train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print("Train shapes:", X_train.shape, y_train.shape)
print("Val shapes:", X_val.shape, y_val.shape)

timesteps = X.shape[1]
features = X.shape[2] if X.ndim == 3 else 1

def build_model(timesteps, features, task_type="regression", use_gru=False, units=64, dropout=0.2):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
    if use_gru:
        model.add(GRU(units, return_sequences=False))
    else:
        model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    if task_type == "classification":
        model.add(Dense(y.shape[1], activation='softmax'))
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        model.add(Dense(1))
        loss = 'mse'
        metrics = ['mse', 'mae']
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model

model = build_model(timesteps, features, task_type=task_type, use_gru=USE_GRU, units=128, dropout=0.3)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("/mnt/data/best_model.h5", monitor='val_loss', save_best_only=True)
]

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# save final model
final_model_path = "/mnt/data/final_model.h5"
model.save(final_model_path)
print("Saved final model to:", final_model_path)

# plot loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/training_loss_plot.png")
plt.show()

# if classification and accuracy present
if task_type == "classification" and 'accuracy' in history.history:
    plt.figure(figsize=(8,5))
    # Keras sometimes uses 'accuracy' or 'acc'
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_' + acc_key
    plt.plot(history.history[acc_key], label='train_acc')
    plt.plot(history.history[val_acc_key], label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/mnt/data/training_accuracy_plot.png")
    plt.show()

# small predictions table
preds = model.predict(X_val[:20])
if task_type == "classification":
    pred_labels = preds.argmax(axis=1)
    true_labels = y_val[:20].argmax(axis=1)
    df = pd.DataFrame({'pred_label': pred_labels, 'true_label': true_labels})
else:
    df = pd.DataFrame({'predicted': preds.ravel(), 'true': y_val[:20].ravel()})
print(df.head(20))

print("Output files saved under /mnt/data:")
for f in os.listdir("/mnt/data"):
    if f.startswith("final_model") or f.startswith("best_model") or f.endswith(".png") or f.endswith(".h5"):
        print(" -", f)
