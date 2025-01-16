from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
import yfinance as yf
import datetime
import aiofiles

### Create FastAPI instance with custom docs and openapi url
app = FastAPI(docs_url="/api/py/docs", openapi_url="/api/py/openapi.json")
# http://127.0.0.1:8000/api/py/docs

@app.get("/api/helloFastApi")
def hello_fast_api():
    return {"message": "Hello from FastAPI"}


# Load the trained model and scaler
model = joblib.load("xgboost_model.pkl") # Trained XGBoost model
scaler = joblib.load("scaler.pkl")


@app.post("/api/predict")
async def predict():
    try:
        # Convert input data to a DataFrame
        #input_data = pd.DataFrame([data.week_data])
        async with aiofiles.open('EWS.csv', mode='r') as file:
            content = await file.read()
            input_data = pd.read_csv(io.StringIO(content))
            input_data.rename(columns={'Data': 'Date'}, inplace=True)

        data = format_data(input_data)

        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Predict anomaly
        anomaly_score = model.predict_proba(scaled_data)[:, 1][0]  # Probability of being an anomaly
        anomaly_label = model.predict(scaled_data)[0]  # Predicted label (0 or 1)
        
        # Return results
        return {
            "Date": data["Date"],
            "anomaly_label": anomaly_label.tolist(),
            "anomaly_score": anomaly_score.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
def format_data(data: pd.DataFrame = None) -> pd.DataFrame:
    features = [
        'VIX', 
        'DXY',
        #'BDIY', 
       # 'LUMSTRUU', 
        'USGG30YR', 
        'GT10', 
        #'GTDEM10Y', 
        #'GTITL10YR', 
        #'GTJPY10YR'
    ]


    if data != None:
        focused_features = data[['Y', 'Date'] + features]
    else:
        focused_features = fetch_data()

    window_count = 4 # each window is about a month long. TODO: try different window sizes
    for f in features: # moving averages
        focused_features.loc[:, f'{f}_MA'] = focused_features[f].rolling(window=window_count).mean()

    # patch up NaN values with smaller temp windows until the intended window size is met
    for i in range(window_count - 1):
        focused_features.loc[i, f'{f}_MA'] = focused_features.loc[i, f]
        focused_features.loc[i, f'{f}_MA'] = focused_features.loc[:i, f'{f}_MA'].mean()

    return focused_features

def fetch_data() -> pd.DataFrame:
    yfinance_tickers = {
        "VIX": "^VIX",  # Volatility Index
        "DXY": "DX-Y.NYB",  # US Dollar Index
        "USGG30YR": "^TYX",  # US 30-Year Treasury Yield
        "GT10": "^TNX",  # US 10-Year Treasury Yield
    }

    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_data = pd.DataFrame()

    # Fetch data from Yahoo Finance
    for label, ticker in yfinance_tickers.items():
        if ticker:
            print(f"Fetching data for {label} ({ticker})...")
            ticker_data = yf.download(ticker, start=start_date, end=end_date)["Close"]  # Use Adjusted Close price
            ticker_data.name = label  # Rename the series to match the label
            all_data = pd.concat([all_data, ticker_data], axis=1)  # Merge into the main DataFrame
        else:
            print(f"Ticker for {label} is not available on yfinance.")


    all_data.rename(columns={value: key for key, value in yfinance_tickers.items()}, inplace=True)

    # Resample to weekly frequency and aggregate using mean
    all_data_weekly = all_data.resample('W').mean()

    # Save the weekly data to CSV
    #all_data_weekly.to_csv("yfinance_time_series_weekly.csv")

    return all_data_weekly