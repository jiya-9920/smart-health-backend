from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Models
# -------------------------------
model_dir = os.path.join("models")

def load_model(filename):
    path = os.path.join(model_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {filename} not found in {model_dir}")
    with open(path, "rb") as f:
        return pickle.load(f)

diabetes_model = load_model("diabetes_model.pkl")
heart_model = load_model("heart_disease_model.pkl")
hypertension_model = load_model("hypertension_model.pkl")

# -------------------------------
# Helper function
# -------------------------------
def pred_to_text(pred, model_type):
    if model_type == "diabetes":
        return "High Diabetes Risk ⚠️" if pred == 1 else "No Diabetes Risk ✅"
    elif model_type == "heart":
        return "High Heart Disease Risk ⚠️" if pred == 1 else "No Heart Disease Risk ✅"
    elif model_type == "hypertension":
        return "High Hypertension Risk ⚠️" if pred == 1 else "No Hypertension Risk ✅"
    else:
        return "Unknown Prediction"

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data or "model_type" not in data:
            return jsonify({"prediction": "model_type is required"}), 400
        model_type = data.get("model_type").lower()

        # Diabetes
        if model_type == "diabetes":
            required_fields = ["age", "bp", "glucose", "bmi"]
            if not all(f in data for f in required_fields):
                return jsonify({"prediction": f"Missing fields: {required_fields}"}), 400
            features = [[float(data[f]) for f in required_fields]]
            pred = diabetes_model.predict(features)[0]
            return jsonify({"prediction": pred_to_text(pred, model_type)})

        # Heart Disease
        elif model_type == "heart":
            required_fields = ["age", "bp", "cholesterol", "max_heart_rate", "sex", "cp"]
            if not all(f in data for f in required_fields):
                return jsonify({"prediction": f"Missing fields: {required_fields}"}), 400
            features = [
                [
                    float(data["age"]),
                    int(data["sex"]),
                    int(data["cp"]),
                    float(data["bp"]),
                    float(data["cholesterol"]),
                    float(data["max_heart_rate"])
                ]
            ]
            pred = heart_model.predict(features)[0]
            return jsonify({"prediction": pred_to_text(pred, model_type)})

        # Hypertension
        elif model_type == "hypertension":
            required_fields = ["age", "bp", "cholesterol", "max_heart_rate"]
            if not all(f in data for f in required_fields):
                return jsonify({"prediction": f"Missing fields: {required_fields}"}), 400
            features = [[float(data[f]) for f in required_fields]]
            pred = hypertension_model.predict(features)[0]
            return jsonify({"prediction": pred_to_text(pred, model_type)})

        else:
            return jsonify({"prediction": "Model type not supported"}), 400

    except Exception as e:
        return jsonify({"prediction": f"Error: {str(e)}"}), 500

# -------------------------------
# Health Check
# -------------------------------
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

# -------------------------------
# Serve React Frontend
# -------------------------------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    react_build_path = "frontend/build"  # Adjust this path based on your folder structure
    if path != "" and os.path.exists(os.path.join(react_build_path, path)):
        return send_from_directory(react_build_path, path)
    else:
        return send_from_directory(react_build_path, "index.html")

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
