from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = joblib.load("xgboost_model.pkl")

# Define the input data format
class TimeSeriesData(BaseModel):
    week_data: list  # List of feature values for the given week

@app.post("/predict")
async def predict(data: TimeSeriesData):
    # Convert input data to a DataFrame
    try:
        input_data = pd.DataFrame([data.week_data])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Predict anomaly
        anomaly_score = model.predict_proba(scaled_data)[:, 1][0]  # Probability of being an anomaly
        anomaly_label = model.predict(scaled_data)[0]             # Predicted label (0 or 1)
        
        # Return results
        return {
            "anomaly_label": int(anomaly_label),
            "anomaly_score": float(anomaly_score)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))