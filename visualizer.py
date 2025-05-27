# visualizer.py

import matplotlib.pyplot as plt

def plot_irradiance(today_df, tomorrow_df):
    """
    Plots solar irradiance for today and predicted irradiance for tomorrow.

    Args:
      today_df (DataFrame): Must have columns 'Hour' and 'Irradiance'
      tomorrow_df (DataFrame): Must have columns 'Hour' and 'Predicted_Irradiance'
    """
    plt.figure(figsize=(10, 6))
    plt.plot(today_df['Hour'], today_df['Irradiance'], label='Today - Actual', marker='o')
    plt.plot(tomorrow_df['Hour'], tomorrow_df['Predicted_Irradiance'], label='Tomorrow - Predicted', marker='x')
    plt.xlabel('Hour of Day (0-24)')
    plt.ylabel('Solar Irradiance (W/m²)')
    plt.title('Solar Irradiance: Today vs Tomorrow Prediction')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(0, 25))
    plt.tight_layout()
