from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import io
import yfinance as yf
from datetime import datetime
import aiofiles
from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(level=logging.INFO)

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

        #logging.info(f"\nInput Data:\n{input_data.head()}")

        data = format_data(input_data)
        logging.info(f"\nFormatted Data:\n{data.head()}")

        # Scale the input data
        #logging.info('\nScaling Data\n')
        #scaled_data = scaler.transform(input_data)
        #logging.info(f"\nScaled Data:\n{scaled_data}")

        std_scaler = StandardScaler()
        X = data.iloc[:, 2:]
        scaled_data = std_scaler.fit_transform(X)

        
        # Predict anomaly
        anomaly_score = model.predict_proba(scaled_data)[:, 1]  # Probability of being an anomaly
        anomaly_label = model.predict(scaled_data)  # Predicted label (0 or 1)
        
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
        'BDIY', 
        'LUMSTRUU', 
        'USGG30YR', 
        'GT10', 
        'GTDEM10Y', 
        'GTITL10YR', 
        'GTJPY10YR'
    ]


    logging.info(f'\nFetching data')
    if data is not None:
        focused_features = data[['Date', 'Y'] + features]
    else:
        focused_features = fetch_data()

    #logging.info(f'\nData Fetched:\n{focused_features.head()}')

    window_count = 4 # each window is about a month long. TODO: try different window sizes
    for f in features: # moving averages
        #logging.info(f'\nCalculating Moving Averages for: {f}\n')
        focused_features = focused_features.assign(**{f'{f}_MA': focused_features[f].rolling(window=window_count).mean()})

        # Calculate the moving average
        rolling_mean = focused_features[f].rolling(window=window_count).mean()

        # Fill NaN values with the original values in the initial window
        rolling_mean_filled = rolling_mean.fillna(focused_features[f])
        
        # Assign the filled rolling mean to the new column
        focused_features = focused_features.assign(**{f'{f}_MA': rolling_mean_filled})

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

    logging.info('\npenis\n')
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

    all_data_weekly.insert(0, 'Y', np.nan)
    # Save the weekly data to CSV
    #all_data_weekly.to_csv("yfinance_time_series_weekly.csv")

    return all_data_weekly