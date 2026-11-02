from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.data_pipeline import load_real_retail_data, transform_to_rfm
from src.models.clv_bgnbd import RetailCLVPipeline



app = FastAPI(
    title="Churn & CLV Prediction API",
    description="Production API serving real-time customer survival, churn risk, and expected CLV.",
    version="1.0.0"
)

# Global Model Initialization with Real Data
pipeline = RetailCLVPipeline()
raw_data = load_real_retail_data("online_retail_II.csv")
rfm_data = transform_to_rfm(raw_data)
pipeline.fit(rfm_data)

class CustomerPayload(BaseModel):
    customer_id: int
    frequency: int
    recency: float
    T: float
    monetary_value: float

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "ChurnPulse-CLV Inference Engine"}

@app.post("/predict")
def predict_customer_metrics(payload: CustomerPayload):
    df_input = pd.DataFrame([payload.model_dump()])
    try:
        results = pipeline.predict(df_input, time_horizon_days=90)
        res = results.iloc[0]
        return {
            "customer_id": int(res['customer_id']),
            "churn_probability": round(float(res['p_churn']), 4),
            "p_active": round(float(res['p_active']), 4),
            "predicted_90d_purchases": round(float(res['predicted_purchases']), 2),
            "predicted_90d_clv": round(float(res['predicted_clv']), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))