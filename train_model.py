"""
train_model.py
--------------
Trains a CNN on CIFAR-10 dataset and saves the model.
Run this ONCE to generate: cnn_model.h5

Requirements:
    pip install tensorflow
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# ── 1. Load CIFAR-10 dataset 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to 0–1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test,  10)

# ── 2. Build the CNN model 
model = models.Sequential([

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Fully Connected
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")   # 10 classes
])

model.summary()

# ── 3. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 4. Train ──────────────────────────────────────────────────────────────────
# epochs=15 gives ~75% accuracy on a low-end laptop in ~10–15 min
model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ── 5. Evaluate & Save ────────────────────────────────────────────────────────
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

model.save("cnn_model.h5")
print("Model saved as cnn_model.h5")
