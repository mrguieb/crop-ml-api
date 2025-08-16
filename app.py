from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

# ----------------------------
# Load your ML model
# ----------------------------
# Make sure your trained model file (e.g., model.pkl) is in the project root
model = joblib.load("model.pkl")  # replace with your model filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# ----------------------------
# Helper: map frontend names to model features
# ----------------------------
def map_features(data):
    mapping = {
        "nitrogen": "N",
        "phosphorus": "P",
        "potassium": "K",
        "ph": "pH",
        "temperature": "temperature",
        "humidity": "humidity",
        "rainfall": "rainfall"
    }
    mapped = {}
    for k, v in data.items():
        if k in mapping:
            mapped[mapping[k]] = v
    return mapped

# ----------------------------
# Manual prediction endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        mapped = map_features(data)
        # Ensure order matches training features if needed
        prediction = model.predict([list(mapped.values())])
        return jsonify({"recommended_crop": prediction[0]})
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
            mapped = map_features(props)
            try:
                pred = model.predict([list(mapped.values())])[0]
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
