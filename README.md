# 🚀 Customer Churn Prediction – End-to-End Machine Learning Project

## 📌 Overview
This project implements an **end-to-end Customer Churn Prediction system** using Machine Learning, Explainable AI (SHAP), and a deployed Flask REST API.

The application predicts whether a customer is likely to churn and provides **interpretable churn drivers**, enabling data-driven business decisions.

---

## 🧠 Key Features
- End-to-end ML pipeline (data preprocessing → training → inference)
- Explainable AI using SHAP for model transparency
- RESTful API for real-time predictions
- Cloud deployment on Render
- Production-ready inference using saved model artifacts

---

## 🛠 Tech Stack
- **Programming:** Python  
- **Data & ML:** Pandas, NumPy, Scikit-learn  
- **Explainability:** SHAP  
- **Backend:** Flask  
- **Deployment:** Render Cloud  

---

## 🔄 Project Workflow
1. Data preprocessing and feature engineering  
2. Model training and evaluation (ROC-AUC ≈ 0.83)  
3. Model explainability using SHAP  
4. REST API development with Flask  
5. Cloud deployment on Render  

---

## 🌍 Live Deployment
**Base URL: https://customer-churn-prediction-41z1.onrender.com ** 

 
### 🔮 Prediction Endpoint

POST /predict
#### Sample Request
```json
{
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 12,
  "PhoneService": 1,
  "MultipleLines": 0,
  "InternetService": 1,
  "OnlineSecurity": 0,
  "OnlineBackup": 1,
  "DeviceProtection": 0,
  "TechSupport": 0,
  "StreamingTV": 1,
  "StreamingMovies": 1,
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": 2,
  "MonthlyCharges": 70,
  "TotalCharges": 800
}


📊 Model Explainability (SHAP)

SHAP is used to:

Identify global feature importance

Explain individual customer predictions

Improve trust and transparency in ML predictions

📁 Project Structure

customer-churn-prediction/
├── data/
│   ├── churn.csv
│   └── X_train.csv
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
├── src/
│   └── model_training.py
├── app.py
├── shap_explain.py
├── test_api.py
├── requirements.txt
├── Procfile
└── README.md

▶️ Run Locally
pip install -r requirements.txt
python app.py


Test API:

python test_api.py
