import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, TFBertModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima

df_titles = pd.read_csv(
    "./Data/baselineData_PreProcessed_Clean.csv",
    parse_dates=["Publication Date"],
    dayfirst=True,
)
df_energy = pd.read_csv(
    "./Data/EnergyPrice_Clean.csv", parse_dates=["Date"], dayfirst=True
)

df_titles["Title"] = df_titles["Title"].astype(str)
df_energy["EnergyPrice"] = df_energy["EnergyPrice"].astype(int)

merged_df = pd.merge(
    df_energy,
    df_titles,
    left_on="Date",
    right_on="Publication Date",
    how="outer",
)

sequence_length_title = 10

merged_df["Title"].fillna("-", inplace=True)
merged_df.dropna(subset=["Date"], inplace=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
title_sequences = merged_df["Title"].apply(
    lambda x: tokenizer.encode(x, add_special_tokens=True)
)
title_sequences = title_sequences.apply(
    lambda x: x[:sequence_length_title] + [0] * (sequence_length_title - len(x))
)

scaler = MinMaxScaler()
merged_df["EnergyPrice"] = scaler.fit_transform(merged_df[["EnergyPrice"]])

# Define X (input features) and y (target variable)
X_title = np.array(title_sequences.tolist())
X_title = X_title[:-sequence_length_title]
X_energy = []
y_energy = []

for i in range(len(merged_df) - sequence_length_title):
    X_sequence = merged_df.iloc[i : i + sequence_length_title]["EnergyPrice"].values
    y_target = merged_df.iloc[i + sequence_length_title]["EnergyPrice"]

    X_energy.append(X_sequence)
    y_energy.append(y_target)

dates = merged_df["Date"]
dates = dates[:-sequence_length_title]

X_energy = np.array(X_energy)
y_energy = np.array(y_energy)

# Split the data into train and test sets
(
    X_title_train,
    X_title_test,
    X_energy_train,
    X_energy_test,
    y_energy_train,
    y_energy_test,
    X_date_train,
    X_date_test,
) = train_test_split(X_title, X_energy, y_energy, dates, test_size=0.2, random_state=42)

model = auto_arima(X_energy_train.ravel(), seasonal=False, suppress_warnings=True)
p, d, q = model.order

# Use the numerical values in the SARIMAX model
model = SARIMAX(X_energy_train.ravel(), order=(p, d, q))
model_fit = model.fit()

# Forecast using the SARIMAX model
forecast_steps = len(X_energy_test)
forecast = model_fit.forecast(steps=forecast_steps)

# Inverse transform the scaled forecast to the original scale
forecast = scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()

# Create a time index for the forecast
forecast_dates = pd.date_range(
    start=dates.iloc[-1], periods=forecast_steps, closed="right"
)

# Plot the actual data and the SARIMAX forecast
plt.figure(figsize=(12, 6))
plt.plot(dates, X_energy[:-forecast_steps], label="Actual Energy Price", color="blue")
plt.plot(forecast_dates, forecast, label="SARIMAX Forecast", color="red")
plt.xlabel("Date")
plt.ylabel("Energy Price")
plt.title("Energy Price Forecast with SARIMAX")
plt.legend()
plt.grid(True)
plt.show()
