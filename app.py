from flask import Flask, request, jsonify
import pandas as pd
import joblib
import shap
import os

# Load model & scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Create Flask app
app = Flask(__name__)

# Create SHAP explainer (for explanations)
background = pd.read_csv("data/X_train.csv").sample(200, random_state=42)
background_scaled = scaler.transform(background)
background_df = pd.DataFrame(background_scaled, columns=background.columns)

explainer = shap.Explainer(model, background_df)

@app.route("/")
def home():
    return "Customer Churn Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Scale input
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # SHAP explanation
    shap_values = explainer(input_scaled_df)
    feature_impact = dict(
        zip(
            input_df.columns,
            shap_values.values[0, :, 1]
        )
    )

    # Top 3 churn drivers
    top_features = sorted(
        feature_impact.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    return jsonify({
        "Churn_Prediction": int(prediction),
        "Churn_Probability": round(probability, 3),
        "Top_Churn_Drivers": top_features
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
