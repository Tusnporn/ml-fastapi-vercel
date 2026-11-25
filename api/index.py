from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="ML Decision Tree API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reg_model = joblib.load(os.path.join(base_dir, 'decision_tree_regression_model.pkl'))
clf_model = joblib.load(os.path.join(base_dir, 'decision_tree_hr_model.pkl'))

class InsuranceInput(BaseModel):
    age: int
    bmi: float
    children: int
    smoker: int

class HRInput(BaseModel):
    Age: int
    MonthlyIncome: float
    DistanceFromHome: int
    OverTime: int

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict/insurance")
def predict_insurance(data: InsuranceInput):
    df = pd.DataFrame([data.dict()])
    pred = reg_model.predict(df)[0]
    return {"predicted_cost": round(float(pred), 2)}

@app.post("/predict/hr")
def predict_hr(data: HRInput):
    df = pd.DataFrame([data.dict()])
    pred = clf_model.predict(df)[0]
    result = "เสี่ยงที่จะลาออก (Resign)" if int(pred) == 1 else "ทำงานอยู่ต่อ (Stay)"
    return {"prediction": int(pred), "status": result}
