import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from utils.ml_model import load_model, predict_tomorrow
from utils.visualizer import plot_irradiance
import matplotlib.pyplot as plt

# NASA POWER API base URL
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"

def fetch_nasa_power_data(lat, lon, start_date, end_date):
    """
    Fetch hourly data for solar irradiance, temperature, and humidity from NASA POWER.
    Returns DataFrame with columns: Date, Hour, ALLSKY_SFC_SW_DWN (irradiance), T2M (temp), RH2M (humidity)
    """
    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": "RE",
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,RH2M",
        "format": "JSON",
        "time-standard": "UTC"
    }
    response = requests.get(NASA_POWER_BASE, params=params)
    if response.status_code != 200:
        st.error("Failed to fetch data from NASA POWER API.")
        return None

    data = response.json()
    try:
        hourly_data = data['properties']['parameter']
    except KeyError:
        st.error("Unexpected API response structure.")
        return None

    records = []
    irradiance = hourly_data['ALLSKY_SFC_SW_DWN']
    temperature = hourly_data['T2M']
    humidity = hourly_data['RH2M']

    # Extract hour from keys like '2023052700', '2023052701', ...
    for time_str in irradiance.keys():
        date_str = time_str[:-2]
        hour = int(time_str[-2:])
        date = datetime.strptime(date_str, "%Y%m%d")
        records.append({
            "Date": date,
            "Hour": hour,
            "Irradiance": irradiance[time_str],
            "Temperature": temperature.get(time_str, None),
            "Humidity": humidity.get(time_str, None)
        })

    df = pd.DataFrame(records)
    return df

def main():
    st.title("Energy Prediction & Load Scheduling")

    # Input for location
    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2090, format="%.4f")

    # Fetch today and yesterday data for better training
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    if st.button("Fetch Data & Predict"):

        # Fetch historic data (yesterday + today) for training and today's data for actual comparison
        historic_df = fetch_nasa_power_data(lat, lon, yesterday, today)
        if historic_df is None or historic_df.empty:
            st.error("No historic data available")
            return
        
        st.success("Fetched historic weather data")

        # Save fetched historic data temporarily for training
        historic_df.to_csv("data/energy_logs.csv", index=False)

        # Load or train model on historic data
        model = load_model()

        # Prepare weather data for tomorrow prediction (temperature and humidity hourly)
        # Use today's temperature/humidity as proxy forecast if future data not available
        tomorrow_weather = []
        for hour in range(24):
            # We approximate tomorrow's temperature/humidity by taking today's same hour (or fallback)
            temp_row = historic_df[(historic_df['Date'] == today) & (historic_df['Hour'] == hour)]
            if not temp_row.empty:
                temperature = temp_row['Temperature'].values[0]
                humidity = temp_row['Humidity'].values[0]
            else:
                temperature = 25  # fallback
                humidity = 60     # fallback
            tomorrow_weather.append({
                "Hour": hour,
                "Temperature": temperature,
                "Humidity": humidity
            })

        # Predict tomorrow's irradiance
        pred_df = predict_tomorrow(model, tomorrow_weather, date=tomorrow)

        # Filter today's data for plotting
        today_df = historic_df[historic_df['Date'] == today][['Hour', 'Irradiance']]

        st.subheader("Solar Irradiance Today vs Tomorrow Prediction")

        # Plot graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(today_df['Hour'], today_df['Irradiance'], label='Today - Actual', marker='o')
        ax.plot(pred_df['Hour'], pred_df['Predicted_Irradiance'], label='Tomorrow - Predicted', marker='x')
        ax.set_xlabel("Hour of Day (0-24)")
        ax.set_ylabel("Solar Irradiance (W/m²)")
        ax.set_title("Solar Irradiance: Today vs Tomorrow Prediction")
        ax.legend()
        ax.grid(True)
        ax.set_xticks(range(0, 25))
        st.pyplot(fig)

if __name__ == "__main__":
    main()
