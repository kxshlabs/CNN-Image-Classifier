import os
import pytest
from train_model import train_and_save

def test_model_training_file_creation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = train_and_save()
    assert model is not None
    assert os.path.exists("cnn_model.h5")
