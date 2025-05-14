from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from pathlib import Path

app = FastAPI()

# Загрузка модели и скейлера
model = torch.jit.load('saved_model/model_scripted.pt', map_location='cpu')
model.eval()
scaler = joblib.load('saved_model/scaler.pkl')


class CreditApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    term: int
    annual_inc: float
    Source_Verified: bool
    Verified: bool
    emp_length: float
    dti: float
    fico_range_low: float
    open_acc: float
    delinq_2yrs: float
    mort_acc: float
    total_acc: float
    revol_util: float
    inq_last_6mths: float
    OWN: bool
    RENT: bool
    OTHER: bool
    purpose_home_improvement: bool
    purpose_debt_consolidation: bool
    purpose_other: bool
    pub_rec_bankruptcies: float


@app.post("/predict/")
async def predict(data: CreditApplication):
    try:
        # Подготовка входных данных в правильном порядке
        input_data = [
            data.fico_range_low, data.dti, data.revol_util, data.open_acc,
            data.total_acc, data.pub_rec_bankruptcies, data.annual_inc,
            data.loan_amnt, data.emp_length, data.term, data.int_rate,
            data.mort_acc, data.delinq_2yrs, data.inq_last_6mths,
            int(data.OTHER), int(data.OWN), int(data.RENT),
            int(data.purpose_debt_consolidation),
            int(data.purpose_home_improvement),
            int(data.purpose_other),
            int(data.Source_Verified),
            int(data.Verified)
        ]

        # Масштабирование и преобразование в тензор
        input_scaled = scaler.transform([input_data])
        input_tensor = torch.FloatTensor(input_scaled)

        # Предсказание
        with torch.no_grad():
            prob = model(input_tensor).item()

        threshold = 0.7
        decision = "Кредит одобрен!" if prob > threshold else "Кредит не одобрен..."

        return {
            "decision": decision,
            "probability": round(prob, 4),
            "threshold": threshold
        }
    except Exception as e:
        return {"error": str(e)}