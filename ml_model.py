import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime, timedelta

MODEL_PATH = os.path.join("models", "energy_predictor.pkl")

# ----------------------
# Train the prediction model
# ----------------------
def train_model(data_path="data/energy_logs.csv"):
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].dt.dayofyear

    features = ['Hour', 'DayOfYear', 'Temperature', 'Humidity']
    target = 'Irradiance'

    X = df[features]
    y = df[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    return model

# ----------------------
# Load trained model
# ----------------------
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_model()

# ----------------------
# Predict tomorrow's irradiance
# ----------------------
def predict_tomorrow(model, weather_data, date=None):
    """
    weather_data should be a list of dicts with:
    [
        {"Hour": 0, "Temperature": 25, "Humidity": 80},
        {"Hour": 1, "Temperature": 24, "Humidity": 82},
        ...
    ]
    """
    if not date:
        date = datetime.today() + timedelta(days=1)

    day_of_year = date.timetuple().tm_yday

    df_pred = pd.DataFrame(weather_data)
    df_pred['DayOfYear'] = day_of_year

    X_pred = df_pred[['Hour', 'DayOfYear', 'Temperature', 'Humidity']]
    df_pred['Predicted_Irradiance'] = model.predict(X_pred)
    return df_pred[['Hour', 'Predicted_Irradiance']]
