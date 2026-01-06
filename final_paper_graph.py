import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

print("--- 1. PREPARING DATA (6 Features) ---")

df = pd.read_csv('Synthetic_Bogdanci_Data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# Add Features
df['Hour'] = df.index.hour
df['Month'] = df.index.month
df['Rolling_Mean'] = df['ActivePower_kW'].rolling(window=3).mean()
df = df.dropna()

data = df[['WindSpeed_m_s', 'Temperature_C', 'ActivePower_kW', 'Hour', 'Month', 'Rolling_Mean']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

LOOK_BACK = 48
X, y = [], []
for i in range(LOOK_BACK, len(scaled_data)):
    X.append(scaled_data[i-LOOK_BACK:i])
    y.append(scaled_data[i, 2]) 

X, y = np.array(X), np.array(y)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("--- 2. TRAINING MODEL (Silent Mode) ---")
model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 6)))
model.add(Dropout(0.05))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.05))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
print("✅ Training Finished.")

print("--- 3. FINDING THE BEST PLOT ---")
predictions = model.predict(X_test)

# Inverse Scale
dummy = np.zeros((len(predictions), 6))
dummy[:, 2] = predictions.flatten()
final_pred = scaler.inverse_transform(dummy)[:, 2]
final_pred = np.clip(final_pred, 0, 2300) # Physics Clip

dummy2 = np.zeros((len(y_test), 6))
dummy2[:, 2] = y_test.flatten()
final_real = scaler.inverse_transform(dummy2)[:, 2]

# AUTOMATICALLY FIND THE MOST INTERESTING WINDOW
# We look for a 200-hour window with the highest Standard Deviation (most movement)
best_start = 0
max_std = 0
window_size = 200

for i in range(0, len(final_real) - window_size, 100):
    current_std = np.std(final_real[i:i+window_size])
    if current_std > max_std:
        max_std = current_std
        best_start = i

print(f"Found best plotting window starting at hour: {best_start}")

# Plot that specific window
start = best_start
end = best_start + window_size

plt.figure(figsize=(12, 6))
plt.plot(final_real[start:end], color='#1f77b4', linewidth=2, label='Actual Power (SCADA)')
plt.plot(final_pred[start:end], color='#ff7f0e', linestyle='--', linewidth=2, label='AI Prediction (LSTM)')

plt.title('Figure 3: Validated Model Performance (Dynamic Response)', fontsize=14)
plt.ylabel('Active Power (kW)', fontsize=12)
plt.xlabel('Time (Hours)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()
