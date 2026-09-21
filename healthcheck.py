import io
import os
import sys
import subprocess
import numpy as np
from PIL import Image

def run_healthcheck():
    results = {}
    
    # 1. File existence checks
    files_to_check = [
        ("models/cnn_model.h5 exists or root cnn_model.h5 fallback", "cnn_model.h5")
    ]

    for check_name, filepath in files_to_check:
        if os.path.exists(filepath) or os.path.exists("models/cnn_model.h5"):
            results[f"CHECK: {check_name}"] = "PASS"
        else:
            results[f"CHECK: {check_name}"] = f"FAIL (File not found: {filepath})"

    # 2. Model loading check
    model = None
    model_path = "models/best_model.h5" if os.path.exists("models/best_model.h5") else ("models/cnn_model.h5" if os.path.exists("models/cnn_model.h5") else "cnn_model.h5")
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        results["CHECK: model loads without error"] = "PASS"
    except Exception as e:
        results["CHECK: model loads without error"] = f"FAIL ({e})"

    # 3. Model prediction check on random array
    if model is not None:
        try:
            dummy_input = np.random.rand(1, 32, 32, 3).astype("float32")
            pred = model.predict(dummy_input)
            if pred.shape == (1, 10):
                results["CHECK: model.predict on random (1,32,32,3) returns shape (1,10)"] = "PASS"
            else:
                results["CHECK: model.predict on random (1,32,32,3) returns shape (1,10)"] = f"FAIL (Shape: {pred.shape})"
        except Exception as e:
            results["CHECK: model.predict on random (1,32,32,3) returns shape (1,10)"] = f"FAIL ({e})"

    # 4. Flask app endpoint tests
    try:
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        res_index = client.get("/")
        if res_index.status_code == 200 and b"CNN_IMAGE_CLASSIFIER" in res_index.data:
            results["CHECK: Endpoint GET /"] = "PASS"
        else:
            results["CHECK: Endpoint GET /"] = f"FAIL (Status: {res_index.status_code})"

        res_pred_err = client.post("/predict")
        if res_pred_err.status_code == 400 and res_pred_err.json.get("error") == "No image uploaded":
            results["CHECK: Endpoint POST /predict (Error Handling)"] = "PASS"
        else:
            results["CHECK: Endpoint POST /predict (Error Handling)"] = f"FAIL (Status: {res_pred_err.status_code})"

        img = Image.new("RGB", (32, 32), color="green")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        res_pred = client.post("/predict", data={"image": (img_byte_arr, "test.png")}, content_type="multipart/form-data")
        if res_pred.status_code == 200 and "top_prediction" in res_pred.json:
            results["CHECK: Endpoint POST /predict (Valid Request)"] = "PASS"
        else:
            results["CHECK: Endpoint POST /predict (Valid Request)"] = f"FAIL (Status: {res_pred.status_code})"

    except Exception as e:
        results["CHECK: Flask App Endpoints"] = f"FAIL ({e})"

    # 5. evaluate.py checks
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        sub_res = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, encoding="utf-8", env=env)
        if sub_res.returncode == 0:
            results["CHECK: evaluate.py runs without exception"] = "PASS"
        else:
            results["CHECK: evaluate.py runs without exception"] = f"FAIL ({sub_res.stderr})"
    except Exception as e:
        results["CHECK: evaluate.py runs without exception"] = f"FAIL ({e})"

    cm_file = os.path.join("evaluation_results", "confusion_matrix.png")
    if os.path.exists(cm_file):
        results["CHECK: evaluation_results/confusion_matrix.png exists after run"] = "PASS"
    else:
        results["CHECK: evaluation_results/confusion_matrix.png exists after run"] = "FAIL (File not found)"

    pca_file = os.path.join("evaluation_results", "per_class_accuracy.png")
    if os.path.exists(pca_file):
        results["CHECK: evaluation_results/per_class_accuracy.png exists after run"] = "PASS"
    else:
        results["CHECK: evaluation_results/per_class_accuracy.png exists after run"] = "FAIL (File not found)"

    print("\n--- HEALTH CHECK RESULTS ---")
    all_pass = True
    for test_name, status in results.items():
        print(f"[{'PASS' if 'PASS' in status else 'FAIL'}] {test_name}: {status}")
        if "PASS" not in status:
            all_pass = False

    if not all_pass:
        sys.exit(1)

if __name__ == "__main__":
    run_healthcheck()

