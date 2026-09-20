"""
app.py
------
Flask backend — loads trained CNN and serves predictions.
"""

import io
import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = "models/best_model.h5" if os.path.exists("models/best_model.h5") else ("models/cnn_model.h5" if os.path.exists("models/cnn_model.h5") else "cnn_model.h5")
model = load_model(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((32, 32))
    img_array = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)

    predictions = model.predict(img_array)[0]
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
        "confidence": results[0]["confidence"],
        "all_predictions": results
    })

if __name__ == "__main__":
    app.run(debug=True)
