from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

# Load trained ML model and LabelEncoder
model = joblib.load("model.pkl")          # your trained model
le = joblib.load("label_encoder.pkl")     # label encoder used during training

app = Flask(__name__)
CORS(app)

# Model features
FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]

# Frontend name mapping
NAME_MAP = {
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
    "ph": "pH",
    "temperature": "temperature",
    "humidity": "humidity",
    "rainfall": "rainfall"
}

# Defaults for missing features
DEFAULTS = {
    "N": 0,
    "P": 0,
    "K": 0,
    "pH": 6.5,
    "temperature": 25,
    "humidity": 70,
    "rainfall": 100
}

# Prepare input in correct order
def prepare_input(data):
    mapped = {}
    for f in FEATURES:
        rev_map = {v: k for k, v in NAME_MAP.items()}
        frontend_name = rev_map.get(f, f)
        mapped[f] = data.get(frontend_name, data.get(f, DEFAULTS[f]))
    return [mapped[f] for f in FEATURES]

# Convert numpy types to native Python types
def to_native(val):
    if isinstance(val, (np.integer, np.int64)):
        return int(val)
    if isinstance(val, (np.floating, np.float64)):
        return float(val)
    return val

# -----------------------
# Manual prediction
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = [prepare_input(data)]
        pred_num = model.predict(X)[0]
        pred_num = to_native(pred_num)
        pred_name = le.inverse_transform([pred_num])[0]  # map number -> crop name
        return jsonify({"recommended_crop": pred_name})
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------
# GeoJSON prediction
# -----------------------
@app.route("/predict-geojson", methods=["POST"])
def predict_geojson():
    try:
        geo = request.get_json()
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            # Map frontend names to model features
            for k in list(props.keys()):
                if k in NAME_MAP:
                    props[NAME_MAP[k]] = props[k]
                    del props[k]
            X = [prepare_input(props)]
            try:
                pred_num = model.predict(X)[0]
                pred_num = to_native(pred_num)
                props["recommended_crop"] = le.inverse_transform([pred_num])[0]
            except Exception as e:
                props["recommended_crop_error"] = str(e)
            feat["properties"] = props
        return jsonify(geo)
    except Exception as e:
        return jsonify({"error": str(e)})

# -----------------------
# Health check
# -----------------------
@app.route("/")
def home():
    return jsonify({"message": "Crop ML API is running!"})

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
