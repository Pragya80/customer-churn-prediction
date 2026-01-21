import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/churn.csv")

# Drop customerID (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# -----------------------------
# 2. Data cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -----------------------------
# 3. Encode categorical columns
# -----------------------------
cat_cols = df.select_dtypes(include="object").columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# 4. Split features & target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save X_train for SHAP
os.makedirs("data", exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)

# -----------------------------
# 5. Scale numerical features
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 6. Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate model
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# -----------------------------
# 8. Save model & scaler
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully!")
