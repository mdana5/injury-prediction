import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Masking

d = np.load("dataset_combined_safe_unsafe.npz")
X, y = d["X"], d["y"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

timesteps = X.shape[1]
features = X.shape[2]

model = Sequential([
    Masking(mask_value=0., input_shape=(timesteps, features)),
    LSTM(128),
    Dropout(0.3),
    BatchNormalization(),
    Dense(1, activation='sigmoid')   # returns unsafe probability
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

model.save("injury_risk_model.h5")
print("Saved → injury_risk_model.h5")
