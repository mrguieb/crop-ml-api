# api.py
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
LE_PATH = os.getenv("LE_PATH", "label_encoder.pkl")

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PATH)
le = joblib.load(LE_PATH)

FEATURES = ["nitrogen","phosphorus","potassium","temperature","humidity","ph","rainfall"]
# Map external keys to dataset column names (N,P,K,...)
RENAMES = {
    "nitrogen":"N", "phosphorus":"P", "potassium":"K",
    "temperature":"temperature", "humidity":"humidity",
    "ph":"ph", "rainfall":"rainfall"
}

def to_row(payload):
    # supports either external keys or dataset keys
    row = {
        "N": payload.get("N", payload.get("nitrogen")),
        "P": payload.get("P", payload.get("phosphorus")),
        "K": payload.get("K", payload.get("potassium")),
        "temperature": payload.get("temperature"),
        "humidity": payload.get("humidity"),
        "ph": payload.get("ph"),
        "rainfall": payload.get("rainfall")
    }
    if any(v is None for v in row.values()):
        missing = [k for k,v in row.items() if v is None]
        raise ValueError(f"Missing features: {missing}")
    return [row["N"], row["P"], row["K"], row["temperature"], row["humidity"], row["ph"], row["rainfall"]]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "model": "crop-recommender", "features": list(RENAMES.values())})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    x = np.array([to_row(data)])
    pred_idx = model.predict(x)[0]
    crop = le.inverse_transform([pred_idx])[0]
    return jsonify({"recommended_crop": crop})

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    items = request.get_json(force=True)
    X = np.array([to_row(it) for it in items])
    preds = model.predict(X)
    crops = le.inverse_transform(preds)
    return jsonify([{"recommended_crop": c} for c in crops])

@app.route("/predict-geojson", methods=["POST"])
def predict_geojson():
    gj = request.get_json(force=True)
    if gj.get("type") != "FeatureCollection":
        return jsonify({"error": "Expecting FeatureCollection"}), 400
    for f in gj.get("features", []):
        props = f.get("properties", {})
        try:
            row = to_row(props)
            pred = le.inverse_transform(model.predict([row]))[0]
            f.setdefault("properties", {})["recommended_crop"] = pred
        except Exception as e:
            f.setdefault("properties", {})["recommended_crop_error"] = str(e)
    return jsonify(gj)
