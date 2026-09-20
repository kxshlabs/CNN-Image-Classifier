"""
train_model.py
--------------
Trains a CNN on CIFAR-10 dataset with data augmentation and callbacks, then saves models and training history.
"""

import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_and_save():
    os.makedirs("models", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        zoom_range=0.1
    )
    datagen.fit(x_train)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(filepath="models/best_model.h5", save_best_only=True, monitor="val_accuracy")
    ]

    model = build_model()
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=128),
        epochs=30,
        validation_data=(x_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )

    np.save("models/training_history.npy", history.history)
    model.save("models/cnn_model.h5")

    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"\nOverall Test Loss: {loss:.4f} - Test Accuracy: {acc * 100:.2f}%")

    predictions = model.predict(x_test)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = y_train.flatten() if y_test.ndim > 1 else y_test

    print("\n--- PER-CLASS ACCURACY ---")
    for i in range(10):
        class_mask = (true_labels == i)
        class_acc = np.mean(pred_labels[class_mask] == true_labels[class_mask]) * 100
        print(f"{CLASS_NAMES[i]:12s}: {class_acc:.2f}%")

    return model

if __name__ == "__main__":
    train_and_save()
