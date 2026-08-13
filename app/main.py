from fastapi import FastAPI, HTTPException
from pydantic import Field, BaseModel
from typing import List
import joblib
from pathlib import Path
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="Turbofan RUL Forecasting API", version="1.0.0")

model = joblib.load("../models/xgb_fd001.pkl")
FEATURE_COUNT = model.n_features_in_
MODEL_VERSION = "xgb_fd001_v1"
TRAINED_ON = "2026-07-31"

class SensorReading(BaseModel):
    features: List[float] = Field(
        ..., description=f"Engineered feature vector, must contain exactly {FEATURE_COUNT} values matching training schema"
    )

class PredictionResponse(BaseModel):
    predicted_rul: float
    risk_level: str
    model_version: str
    timestamp: str
    
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model-info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "trained_on": TRAINED_ON,
        "model_type": "XGBoost Regressor",
        "expected_features": FEATURE_COUNT
    }
    
def get_risk_level(rul: float) -> str:
    if rul < 30:
        return "critical"
    elif rul < 60:
        return "warning"
    return "healthy"

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if len(reading.features) != FEATURE_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {FEATURE_COUNT} features, received {len(reading.features)}"
        )

    X = np.array(reading.features).reshape(1, -1)
    pred = float(model.predict(X)[0])

    return PredictionResponse(
        predicted_rul=pred,
        risk_level=get_risk_level(pred),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat()
    )        