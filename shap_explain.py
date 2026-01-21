import pandas as pd
import shap
import joblib

# Load model and scaler
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load training data
X_train = pd.read_csv("data/X_train.csv")

# Sample for speed
X_train = X_train.sample(300, random_state=42)

# Scale data
X_train_scaled = scaler.transform(X_train)

# Convert to DataFrame with column names
X_train_scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

# ✅ NEW SHAP API (THIS IS THE FIX)
explainer = shap.Explainer(model, X_train_scaled_df)

# Get SHAP values
shap_values = explainer(X_train_scaled_df)

# For binary classification → use class 1
shap.summary_plot(
    shap_values.values[:, :, 1],
    X_train_scaled_df
)
