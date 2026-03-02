import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("injury_risk_model.h5")

def clean(window):
    window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
    window = np.clip(window, -5, 5)
    return window

def predict_risk(window):
    window = clean(window)    # <-- CLEAN HERE
    p = model.predict(window[np.newaxis, ...])[0][0]
    if np.isnan(p):
        return None
    return int(p * 100), p

d_unsafe = np.load("unsafe_dataset_volleyball_sequence_scaled.npz")
sample_unsafe = clean(d_unsafe["X"][0])   # <-- CLEAN HERE

risk_percent, raw = predict_risk(sample_unsafe)

print("Raw:", raw)
print("Risk:", risk_percent, "%")
