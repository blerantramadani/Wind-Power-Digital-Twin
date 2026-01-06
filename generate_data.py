import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SETTINGS FOR REALISTIC SMOOTH WIND ---
RATED_POWER_KW = 2300  
CUT_IN_SPEED = 3.0     
RATED_SPEED = 12.0     
CUT_OUT_SPEED = 25.0   

print("--- 1. Generating REALISTIC (SMOOTH) Synthetic Data ---")

dates = pd.date_range(start='2020-01-01', end='2021-12-31 23:00:00', freq='H')
n_samples = len(dates)

# --- THE FIX: SMOOTH WIND GENERATION ---
# We start with random noise
noise = np.random.normal(0, 1, n_samples)
# We create a "Cumulative Sum" (Random Walk) to make it continuous
smooth_curve = np.cumsum(noise)
# We normalize it to fit typical wind speeds (e.g., 0 to 20 m/s)
# This creates weather that changes slowly over hours/days
min_val = np.min(smooth_curve)
max_val = np.max(smooth_curve)
# Scale to 0-1 then multiply by max wind speed (approx 18 m/s avg peaks)
wind_speed = ((smooth_curve - min_val) / (max_val - min_val)) * 22

# Add a little daily pattern (Wind is usually stronger in afternoon)
hour_boost = 2 * np.sin(2 * np.pi * dates.hour / 24)
wind_speed = wind_speed + hour_boost
# Ensure wind doesn't go below 0
wind_speed = np.abs(wind_speed) 

# Temperature (Seasonal)
day_of_year = dates.dayofyear
temp_trend = 15 - 10 * np.cos(2 * np.pi * day_of_year / 365) 
temperature = temp_trend + np.random.normal(0, 1, n_samples) 

# Calculate Power
power_output = []

for ws in wind_speed:
    if ws < CUT_IN_SPEED or ws > CUT_OUT_SPEED:
        p = 0
    elif ws < RATED_SPEED:
        ratio = (ws - CUT_IN_SPEED) / (RATED_SPEED - CUT_IN_SPEED)
        p = RATED_POWER_KW * (ratio ** 3)
    else:
        p = RATED_POWER_KW
    
    power_output.append(p)

# Tiny sensor noise (1%)
power_output = np.array(power_output) * np.random.uniform(0.99, 1.01, n_samples)
power_output = np.clip(power_output, 0, RATED_POWER_KW)

df = pd.DataFrame({
    'Timestamp': dates,
    'WindSpeed_m_s': np.round(wind_speed, 2),
    'Temperature_C': np.round(temperature, 2),
    'ActivePower_kW': np.round(power_output, 2)
})

filename = 'Synthetic_Bogdanci_Data.csv'
df.to_csv(filename, index=False)
print(f"✅ SUCCESS: Smooth Wind Dataset saved.")

# Plot to prove it looks real
plt.figure(figsize=(12, 4))
plt.plot(df['Timestamp'][:200], df['WindSpeed_m_s'][:200])
plt.title("Wind Speed over First 200 Hours (Notice it is smooth now!)")
plt.ylabel("m/s")
plt.grid(True)
plt.show()
