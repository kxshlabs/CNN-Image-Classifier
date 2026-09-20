import io
import sys
from PIL import Image
import numpy as np

def run_healthcheck():
    results = {}
    
    # 1. Verify Model file existence
    try:
        from tensorflow.keras.models import load_model
        model = load_model("cnn_model.h5")
        results["Model Load Check"] = "PASS"
    except Exception as e:
        results["Model Load Check"] = f"FAIL ({e})"

    # 2. Test Flask app client endpoints
    try:
        from app import app
        app.config["TESTING"] = True
        client = app.test_client()

        # Check Endpoint: GET /
        res_index = client.get("/")
        if res_index.status_code == 200 and b"CNN_IMAGE_CLASSIFIER" in res_index.data:
            results["Endpoint GET /"] = "PASS"
        else:
            results["Endpoint GET /"] = f"FAIL (Status: {res_index.status_code})"

        # Check Endpoint: POST /predict (No file)
        res_pred_err = client.post("/predict")
        if res_pred_err.status_code == 400 and res_pred_err.json.get("error") == "No image uploaded":
            results["Endpoint POST /predict (Error Handling)"] = "PASS"
        else:
            results["Endpoint POST /predict (Error Handling)"] = f"FAIL (Status: {res_pred_err.status_code})"

        # Check Endpoint: POST /predict (Valid Image)
        img = Image.new("RGB", (32, 32), color="green")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        
        res_pred = client.post("/predict", data={"image": (img_byte_arr, "test.png")}, content_type="multipart/form-data")
        if res_pred.status_code == 200 and "top_prediction" in res_pred.json:
            results["Endpoint POST /predict (Valid Request)"] = "PASS"
        else:
            results["Endpoint POST /predict (Valid Request)"] = f"FAIL (Status: {res_pred.status_code})"

    except Exception as e:
        results["Flask App Initialization"] = f"FAIL ({e})"

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
