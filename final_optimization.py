import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import math

print("--- 1. ADVANCED FEATURE ENGINEERING ---")

df = pd.read_csv('Synthetic_Bogdanci_Data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# ADDING NEW FEATURES (The "Clues")
df['Hour'] = df.index.hour
df['Month'] = df.index.month
# Rolling Mean: The average of the last 3 hours (helps smooth out noise)
df['Rolling_Mean'] = df['ActivePower_kW'].rolling(window=3).mean()

# Drop the first few rows that have NaN because of the rolling mean
df = df.dropna()

print("New Data Header (with Time Features):")
print(df[['ActivePower_kW', 'Hour', 'Month', 'Rolling_Mean']].head())

# Select ALL features: Wind, Temp, Power, Hour, Month, RollingMean
# We now have 6 Inputs instead of 3
data = df[['WindSpeed_m_s', 'Temperature_C', 'ActivePower_kW', 'Hour', 'Month', 'Rolling_Mean']].values

# Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create Windows
LOOK_BACK = 48
X, y = [], []

for i in range(LOOK_BACK, len(scaled_data)):
    # Input: All 6 columns
    X.append(scaled_data[i-LOOK_BACK:i])
    # Output: ONLY the Power column (Index 2)
    y.append(scaled_data[i, 2]) 

X, y = np.array(X), np.array(y)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nNew Input Shape: {X_train.shape} (Note: The '6' means 6 features)")

print("\n--- 2. TRAINING FINAL MODEL ---")

model = Sequential()
# Input shape now expects 6 features
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 6)))
model.add(Dropout(0.1))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(units=1))

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)

print("\n--- 3. CALCULATING FINAL RMSE ---")

predictions = model.predict(X_test)

# Inverse Scaling Logic for 6 columns
dummy_array = np.zeros((len(predictions), 6)) # 6 Columns now!
dummy_array[:, 2] = predictions.flatten() # Put prediction in Power column (index 2)
inverse_pred = scaler.inverse_transform(dummy_array)
final_predictions = inverse_pred[:, 2]

dummy_array_2 = np.zeros((len(y_test), 6))
dummy_array_2[:, 2] = y_test.flatten()
inverse_real = scaler.inverse_transform(dummy_array_2)
final_real = inverse_real[:, 2]

rmse = math.sqrt(mean_squared_error(final_real, final_predictions))
print(f"FINAL RMSE: {rmse:.2f} kW")

plt.figure(figsize=(14, 6))
plt.plot(final_real[:200], label='Actual')
plt.plot(final_predictions[:200], linestyle='--', label='AI Prediction (Feature Engineered)')
plt.title(f'Final PhD Model Results (RMSE: {rmse:.2f} kW)')
plt.legend()
plt.grid(True)
plt.show()
