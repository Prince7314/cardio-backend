from fastapi import FastAPI

from pydantic import BaseModel
import pandas as pd
import joblib

# --------------------------------------------------
# Load trained pipeline (preprocessor + RF model)
# --------------------------------------------------
MODEL_PATH = "rf_pipeline.pkl"
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Cardio Risk Prediction API",
    version="1.0.0"
)

# --------------------------------------------------
# CORS
# --------------------------------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Input Schema
# --------------------------------------------------
class InputData(BaseModel):
    age: int
    gender: int
    height: float
    weight: float
    systolic_bp: int
    diastolic_bp: int
    cholesterol: int
    glucose: int
    smoking: int
    alcohol: int
    active: int

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: InputData):
    # Map API fields to TRAINING column names
    payload = {
        "AgeInYears": data.age,
        "Gender": data.gender,
        "HeightCm": data.height,
        "WeightKg": data.weight,
        "SystolicBP": data.systolic_bp,
        "DiastolicBP": data.diastolic_bp,
        "Cholesterol": data.cholesterol,
        "Glucose": data.glucose,
        "Smoking": data.smoking,
        "Alcohol": data.alcohol,
        "PhysicallyActive": data.active,
    }

    df = pd.DataFrame([payload])

    # IMPORTANT: compute BMI exactly like training
    df["BMI"] = df["WeightKg"] / ((df["HeightCm"] / 100) ** 2)

    probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(probability >= 0.5),
        "risk_probability": round(probability * 100, 2)
    }

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "API running successfully"}
