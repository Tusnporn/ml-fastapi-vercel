import os
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Insurance & HR Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
reg_model = joblib.load(os.path.join(base_dir, "decision_tree_regression_model.pkl"))
clf_model = joblib.load(os.path.join(base_dir, "decision_tree_hr_model.pkl"))

# 1. หน้าแรก (Root) แสดงหน้าเว็บจาก index.html
@app.get("/")
def read_root():
    html_path = os.path.join(base_dir, "index.html")
    return FileResponse(html_path)

# 2. ตรวจสอบสถานะ API
@app.get("/health")
def health():
    return {"status": "ok"}

# 3. ข้อมูล Feature สำหรับทำนาย
@app.get("/info")
def info():
    return {
        "insurance_features": ["age", "bmi", "children", "smoker"],
        "hr_features": ["Age", "MonthlyIncome", "DistanceFromHome", "OverTime"],
        "example": {"age": 35, "bmi": 28.5, "children": 1, "smoker": 0}
    }

# 4. โครงสร้างข้อมูลขาเข้า (รองรับทั้ง 2 โมเดลใน /predict เดียว)
class PredictRequest(BaseModel):
    model_type: Optional[str] = "insurance"
    age: Optional[int] = 35
    bmi: Optional[float] = 28.5
    children: Optional[int] = 1
    smoker: Optional[int] = 0
    Age: Optional[int] = 30
    MonthlyIncome: Optional[float] = 4000
    DistanceFromHome: Optional[int] = 20
    OverTime: Optional[int] = 0

# 5. Endpoint ทำนายผล
@app.post("/predict")
def predict(data: PredictRequest):
    if data.model_type == "hr":
        df = pd.DataFrame([{
            "Age": data.Age,
            "MonthlyIncome": data.MonthlyIncome,
            "DistanceFromHome": data.DistanceFromHome,
            "OverTime": data.OverTime
        }])
        pred = clf_model.predict(df)[0]
        label = "เสี่ยงที่จะลาออก (Resign)" if int(pred) == 1 else "ทำงานอยู่ต่อ (Stay)"
        return {"prediction": int(pred), "label": label}
    else:
        df = pd.DataFrame([{
            "age": data.age,
            "bmi": data.bmi,
            "children": data.children,
            "smoker": data.smoker
        }])
        pred = reg_model.predict(df)[0]
        return {"prediction": round(float(pred), 2)}
