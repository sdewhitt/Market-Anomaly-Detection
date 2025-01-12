from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
import yfinance as yf

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and scaler
model = joblib.load("xgboost_model.pkl") # Trained XGBoost model
scaler = joblib.load("scaler.pkl")

features = [
    'VIX', 
    'DXY',
    'BDIY', 
    'LUMSTRUU', 
    'USGG30YR', 
    'GT10', 
    'GTDEM10Y', 
    'GTITL10YR', 
    'GTJPY10YR'
]

class TimeSeriesData(BaseModel):
    week_data: list  # List of feature values for the given week

@app.post("/predict")
async def predict(data: TimeSeriesData):
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.week_data])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Predict anomaly
        anomaly_score = model.predict_proba(scaled_data)[:, 1][0]  # Probability of being an anomaly
        anomaly_label = model.predict(scaled_data)[0]  # Predicted label (0 or 1)
        
        # Return results
        return {
            "anomaly_label": int(anomaly_label),
            "anomaly_score": float(anomaly_score)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/uploadfile")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(await file.read()))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Ensure the DataFrame is in the correct format
        if df.empty or df.shape[1] != len(TimeSeriesData.__annotations__['week_data']):
            raise HTTPException(status_code=400, detail="Invalid data format")
        




        focused_features = df[['Date'] + features]

        window_count = 4 # each window is about a month long. TODO: try different window sizes
        for f in features: # moving averages
            focused_features.loc[:, f'{f}_MA'] = focused_features[f].rolling(window=window_count).mean()

        # patch up NaN values with smaller temp windows until the intended window size is met
        for i in range(window_count - 1): 
            focused_features.loc[i, f'{f}_MA'] = focused_features.loc[i, f]
            focused_features.loc[i, f'{f}_MA'] = focused_features.loc[:i, f'{f}_MA'].mean()


        # Scale the input data
        scaled_data = scaler.transform(df)
        
        # Predict anomalies
        anomaly_scores = model.predict_proba(scaled_data)[:, 1]  # Probabilities of being anomalies
        anomaly_labels = model.predict(scaled_data)  # Predicted labels (0 or 1)
        
        # Return results
        return {
            "anomaly_labels": anomaly_labels.tolist(),
            "anomaly_scores": anomaly_scores.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))