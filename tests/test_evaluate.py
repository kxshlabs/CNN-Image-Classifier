import os
import pytest
from PIL import Image
import evaluate

def test_fallback_model_path_logic(monkeypatch):
    def mock_exists_best(path):
        return path == "models/best_model.h5"

    monkeypatch.setattr(os.path, "exists", mock_exists_best)
    assert evaluate.get_model_path() == "models/best_model.h5"

    def mock_exists_cnn(path):
        return path == "models/cnn_model.h5"

    monkeypatch.setattr(os.path, "exists", mock_exists_cnn)
    assert evaluate.get_model_path() == "models/cnn_model.h5"

    def mock_exists_root(path):
        return path == "cnn_model.h5"

    monkeypatch.setattr(os.path, "exists", mock_exists_root)
    assert evaluate.get_model_path() == "cnn_model.h5"

    def mock_exists_none(path):
        return False

    monkeypatch.setattr(os.path, "exists", mock_exists_none)
    with pytest.raises(FileNotFoundError):
        evaluate.get_model_path()

def test_evaluate_execution_and_artifacts(capsys):
    evaluate.main()
    captured = capsys.readouterr()
    
    assert "┌──────────────┬──────────┐" in captured.out
    assert "└──────────────┴──────────┘" in captured.out
    assert os.path.exists("evaluation_results")
    
    cm_path = os.path.join("evaluation_results", "confusion_matrix.png")
    acc_path = os.path.join("evaluation_results", "per_class_accuracy.png")
    
    assert os.path.exists(cm_path)
    assert os.path.exists(acc_path)
    
    with Image.open(cm_path) as img:
        img.verify()
        
    with Image.open(acc_path) as img:
        img.verify()
