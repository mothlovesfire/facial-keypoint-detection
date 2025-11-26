"""
STEPS TO IMPLEMENT & IMPROVE CNN
--------------------------------
1. Data preparation
   - Convert 'Image' strings to arrays.
   - Normalize pixels to [0,1].
   - Reshape to (N, 96, 96, 1).
   - Normalize keypoints by dividing by 96.

2. Build baseline CNN
   - 2–3 Conv2D + MaxPooling blocks.
   - Flatten → Dense(512) → Dense(256) → Dense(output_dim).

3. Train with validation / k-fold
   - Use EarlyStopping(monitor='val_loss', patience=5).
   - Batch size ~32–128, epochs ~30–100.

4. Improve performance
   - Add BatchNormalization after conv layers.
   - Add Dropout(0.3–0.5) in dense layers.
   - Add more conv filters (e.g. 32 → 64 → 128 → 256).
   - Use data augmentation (random flip, rotation, zoom).
   - Tune learning rate (1e-3, 5e-4, 1e-4).
   - Use learning-rate scheduler (ReduceLROnPlateau).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
# you need Python 3.12 for this. if you want to run this beautiful program & need help installing that, ping me (i'm brian)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
"EarlyStopping Prevents Overfitting CNNs. Stopping when validation loss stops improving helps maintain performance."
"No necessary but useful."
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

folds = 5
input_size = 96
n_channels = 1   # grayscale
# num keypoints = train_points.shape[1] / 2  (x,y pairs)

# 1. LOAD DATA (same as before)
train_frame = pd.read_csv('data\\training.csv').dropna()
train_points = train_frame.values[:, :-1].astype(float)
train_images_str = train_frame.values[:, -1]

# Convert image strings to arrays
train_images = np.array([
    np.array(img_str.split(), dtype=float) for img_str in train_images_str
])

# Normalize pixel values to [0,1]
train_images = train_images / 255.0

# Reshape to (N, 96, 96, 1) for CNN input
train_images = train_images.reshape(-1, input_size, input_size, n_channels)

# Optionally normalize keypoints to [0,1] by dividing by image size
# (Improves training stability)
#train_points_norm = train_points / input_size
pre_mean = train_points.mean(axis = 0)
pre_std = train_points.std(axis = 0)
train_points_norm = (train_points - pre_mean) / pre_std

def build_cnn_model(output_dim):
    """
    Build a CNN model for facial keypoint regression.

    output_dim = number of keypoint coordinates (e.g. 30 for 15 points).
    """
    model = Sequential()

    # ---- CNN BLOCK 1 ----
    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=(input_size, input_size, n_channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ---- CNN BLOCK 2 ----
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ---- CNN BLOCK 3 (optional deeper block to improve performance) ----
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten conv features and use fully connected layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))  # Dropout helps reduce overfitting
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer: one neuron per keypoint coordinate
    model.add(Dense(output_dim))

    # Compile with MSE loss for regression
    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm = 1.0),
        loss='mse',
        metrics=['mae']
        )
    return model

# Helper: split into k folds (reuse your kfold logic but adapted for images & points)
def kfold_indices(n_samples, k):
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size
    return folds

n_samples = train_images.shape[0]
fold_indices = kfold_indices(n_samples, folds)

all_preds = np.zeros_like(train_points_norm)

print("Starting CNN k-fold training...")
for fold_idx in range(folds):
    print(f"\n=== Fold {fold_idx+1}/{folds} ===")

    # Validation indices for this fold
    val_idx = fold_indices[fold_idx]
    # Training indices are all others
    train_idx = np.concatenate([fold_indices[i] for i in range(folds) if i != fold_idx])

    X_train = train_images[train_idx]
    y_train = train_points_norm[train_idx]
    X_val, y_val = train_images[val_idx], train_points_norm[val_idx]

    # Build a fresh CNN for this fold
    model = build_cnn_model(output_dim=train_points_norm.shape[1])

    # Early stopping to improve generalization and save time
    es = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
        )
    lr_scheduler = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.3,
        patience = 3,
        min_lr = 1e-7,
        verbose = 1
    )

    # ---- TRAIN CNN ----
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=150,        # can increase for better performance
        batch_size=16,    # hyperparameter to tune
        callbacks=[es, lr_scheduler],
        verbose=1
    )

    # ---- PREDICT ON VALIDATION FOLD ----
    fold_preds = model.predict(X_val)
    all_preds[val_idx] = fold_preds

print("\nTraining finished (CNN).")

# Rescale predictions back to original coordinates
all_preds_rescaled = (all_preds * pre_std) + pre_mean

print("CNN Metrics:")
print("MSE:", mean_squared_error(train_points, all_preds_rescaled))
print("MAE:", mean_absolute_error(train_points, all_preds_rescaled))
