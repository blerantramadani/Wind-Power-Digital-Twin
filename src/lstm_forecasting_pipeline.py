import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# --- 1. DATA LOADING & FEATURE ENGINEERING ---
# This section implements the input vector Xt = [vt, Tt, Pt-1, Ht, Mt, mu_3h] [cite: 71]
df = pd.read_csv('data/Synthetic_Bogdanci_Data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Cyclic Encoding for Hour and Month (preserves temporal periodicity) [cite: 75]
df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# Low-pass filter: 3-hour rolling mean to attenuate high-frequency turbulence [cite: 76]
df['Rolling_Mean'] = df['ActivePower_kW'].rolling(window=3).mean()
df = df.dropna()

# Selecting the 6 distinct features defined in Eq. (3) [cite: 70-71]
# Features: WindSpeed, Temp, Prev_Power, Hour_Cycles, Month_Cycles, Rolling_Mean
features = ['WindSpeed_m_s', 'Temperature_C', 'ActivePower_kW', 'Hour_sin', 'Month_sin', 'Rolling_Mean']
data = df[features].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# --- 2. SLIDING WINDOW PREPARATION ---
LOOK_BACK = 48 # 48-hour look-back period [cite: 81]
X, y = [], []
for i in range(LOOK_BACK, len(scaled_data)):
    X.append(scaled_data[i-LOOK_BACK:i])
    y.append(scaled_data[i, 2]) # Target: ActivePower_kW

X, y = np.array(X), np.array(y)
train_size = int(len(X) * 0.8) # 80/20 Train-Test split [cite: 90]
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 3. DEEP LSTM ARCHITECTURE ---
# Optimized two-layer stacked LSTM with Dropout Regularization [cite: 78, 80]
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.05),
    LSTM(64, return_sequences=False),
    Dropout(0.05),
    Dense(1) # Regression head for power output [cite: 78]
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
print("Training Physics-Aware LSTM...")
model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# --- 4. EVALUATION & INVERSE SCALING ---
predictions = model.predict(X_test)
dummy = np.zeros((len(predictions), len(features)))
dummy[:, 2] = predictions.flatten()
final_predictions = scaler.inverse_transform(dummy)[:, 2]

dummy_real = np.zeros((len(y_test), len(features)))
dummy_real[:, 2] = y_test.flatten()
final_real = scaler.inverse_transform(dummy_real)[:, 2]

# Calculating nRMSE (Target result: 2.25%) 
rmse = math.sqrt(mean_squared_error(final_real, final_predictions))
nrmse = (rmse / 2300) * 100 # Normalized to 2.3 MW rated power [cite: 59]
print(f"Final nRMSE: {nrmse:.2f}%")

# --- 5. VISUALIZATION (Figure 4) ---
plt.figure(figsize=(12, 6))
plt.plot(final_real[:200], label='Actual Power (Synthetic Ground Truth)', color='blue')
plt.plot(final_predictions[:200], label='Predicted Power (LSTM)', color='orange', linestyle='--')
plt.title(f'Figure 4: Comparative Analysis of Dynamic Response (nRMSE: {nrmse:.2f}%)')
plt.ylabel('Active Power (kW)')
plt.legend()
plt.grid(True)
plt.show()
