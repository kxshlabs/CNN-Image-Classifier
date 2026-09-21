import ssl
import numpy as np
import pytest
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from train_model import build_model

ssl._create_default_https_context = ssl._create_unverified_context

def test_dataset_loading_and_shapes():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)

def test_normalization():
    (x_train, _), _ = cifar10.load_data()
    x_normalized = x_train.astype("float32") / 255.0
    assert x_normalized.min() >= 0.0
    assert x_normalized.max() <= 1.0

def test_one_hot_encoding_shape():
    (_, y_train), _ = cifar10.load_data()
    y_cat = to_categorical(y_train, 10)
    assert y_cat.shape == (50000, 10)

def test_model_output_shape():
    model = build_model()
    assert model.output_shape == (None, 10)

def test_model_compiles_without_error():
    model = build_model()
    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"

def test_single_batch_forward_pass():
    batch_size = 16
    dummy_batch = np.random.rand(batch_size, 32, 32, 3).astype("float32")
    model = build_model()
    output = model.predict(dummy_batch)
    assert output.shape == (batch_size, 10)
