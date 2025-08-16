from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# ----------------------------
# Load your ML model
# ----------------------------
model = joblib.load("model.pkl")  # replace with your model filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# ----------------------------
# Feature configuration
# ----------------------------
FEATURES = ["N", "P", "K", "pH", "temperature", "humidity", "rainfall"]

# Map frontend names to model names
NAME_MAP = {
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
    "ph": "pH",
    "temperature": "temperature",
    "humidity": "humidity",
    "rainfall": "rainfall"
}

# Default values if a property is missing
DEFAULTS = {
    "N": 0,
    "P": 0,
    "K": 0,
    "pH": 6.5,
    "temperature": 25,
    "humidity": 70,
    "rainfall": 100
}

# ----------------------------
# Prepare input for model
# ----------------------------
def prepare_input(data):
    mapped = {}
    for f in FEATURES:
        # Try frontend name mapping first
        rev_map = {v:k for k,v in NAME_MAP.items()}
        frontend_name = rev_map.get(f, f)
        mapped[f] = data.get(frontend_name, data.get(f, DEFAULTS[f]))
    return [mapped[f] for f in FEATURES]

# ----------------------------
# Manual prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        X = [prepare_input(data)]
        pred = model.predict(X)[0]
        return jsonify({"recommended_crop": pred})
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------
# GeoJSON prediction endpoint
# ----------------------------
@app.route("/predict-geojson", methods=["POST"])
def predict_geojson():
    try:
        geo = request.get_json()
        for feat in geo.get("features", []):
            props = feat.get("properties", {})
            X = [prepare_input(props)]
            try:
                pred = model.predict(X)[0]
                props["recommended_crop"] = pred
            except Exception as e:
                props["recommended_crop_error"] = str(e)
            feat["properties"] = props
        return jsonify(geo)
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------------
# Health check
# ----------------------------
@app.route("/")
def home():
    return jsonify({"message": "Crop ML API is running!"})

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
