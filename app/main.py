from typing import Dict, Any
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path("models/latest_model.joblib")
 
app = FastAPI(title="Customer Churn Prediction API")
model = None
 
 
class ChurnRequest(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    Contract: str = "Month-to-month"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 850.0
 
 
@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
 
 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
 
 
@app.post("/predict")
def predict(payload: ChurnRequest) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
 
    input_df = pd.DataFrame([payload.model_dump()])
    input_df = pd.get_dummies(input_df, drop_first=True)

    expected_columns = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
 
    # Спрощення для лабораторної: у реальному сервісі потрібно зберігати повний preprocessing pipeline.
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
 
    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
        "risk_label": "high" if probability >= 0.5 else "low",
    }
