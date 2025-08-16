from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model.pkl")   # make sure the filename matches your saved model

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Crop ML API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Convert to DataFrame (expects dict with column names)
        df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(df)[0]

        return jsonify({"prediction": str(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
