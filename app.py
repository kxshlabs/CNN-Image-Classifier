"""
app.py
------
Flask backend — loads trained CNN and serves predictions.

Requirements:
    pip install flask tensorflow pillow

Run:
    python app.py
Then open:
    http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load the saved model once at startup
print("Loading model...")
model = load_model("cnn_model.h5")
print("Model loaded!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Check file was sent
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Read and preprocess the image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((32, 32))                        # CIFAR-10 expects 32x32

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)     # shape: (1, 32, 32, 3)

    # Run prediction
    predictions = model.predict(img_array)[0]          # shape: (10,)

    # Build top-5 results
    top5_indices = predictions.argsort()[-5:][::-1]
    results = [
        {
            "label": CLASS_NAMES[i],
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top5_indices
    ]

    return jsonify({
        "top_prediction": results[0]["label"],
        "confidence":     results[0]["confidence"],
        "all_predictions": results
    })


if __name__ == "__main__":
    app.run(debug=True)
