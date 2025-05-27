import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from utils.ml_model import load_model, predict_tomorrow
from utils.visualizer import plot_irradiance
from utils.scheduler import schedule_loads
import matplotlib.pyplot as plt

# NASA POWER API base URL
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"

def fetch_nasa_power_data(lat, lon, start_date, end_date):
    # (Same as before)
    # ... [omitted for brevity, keep your existing code here] ...
    pass

def main():
    st.title("Energy Prediction & Load Scheduling")

    lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.6139, format="%.4f")
    lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.2090, format="%.4f")

    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)

    threshold = st.slider("Irradiance Threshold for Load Scheduling (W/m²)", 0, 1000, 200)

    if st.button("Fetch Data & Predict"):

        historic_df = fetch_nasa_power_data(lat, lon, yesterday, today)
        if historic_df is None or historic_df.empty:
            st.error("No historic data available")
            return
        
        st.success("Fetched historic weather data")

        historic_df.to_csv("data/energy_logs.csv", index=False)

        model = load_model()

        tomorrow_weather = []
        for hour in range(24):
            temp_row = historic_df[(historic_df['Date'] == today) & (historic_df['Hour'] == hour)]
            if not temp_row.empty:
                temperature = temp_row['Temperature'].values[0]
                humidity = temp_row['Humidity'].values[0]
            else:
                temperature = 25
                humidity = 60
            tomorrow_weather.append({
                "Hour": hour,
                "Temperature": temperature,
                "Humidity": humidity
            })

        pred_df = predict_tomorrow(model, tomorrow_weather, date=tomorrow)

        today_df = historic_df[historic_df['Date'] == today][['Hour', 'Irradiance']]

        # Plot irradiance graph
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

        # Load scheduling based on predicted irradiance and threshold
        optimal_hours = schedule_loads(pred_df, threshold=threshold)
        if optimal_hours:
            st.subheader("Recommended Hours to Run Appliances (Based on Solar Irradiance)")
            st.write(f"Optimal hours (irradiance ≥ {threshold} W/m²):")
            st.write(", ".join(str(hr) + ":00" for hr in optimal_hours))
        else:
            st.warning(f"No hours with irradiance above {threshold} W/m² found for tomorrow.")

if __name__ == "__main__":
    main()
