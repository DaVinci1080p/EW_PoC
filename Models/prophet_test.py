import warnings
import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from prophet import Prophet

# Suppress the FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

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

merged_df["Title"].fillna("-", inplace=True)
merged_df.dropna(subset=["Date"], inplace=True)

merged_df["ds"] = pd.to_datetime(merged_df["Date"])
merged_df["y"] = pd.to_datetime(merged_df["EnergyPrice"])

model = Prophet()
model.fit(merged_df)

# Make predictions
future_dates = model.make_future_dataframe(
    periods=365
)  # Generate future dates for prediction
predictions = model.predict(future_dates)
print(predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())
fig1 = model.plot(predictions)
fig1.show()
input("press a key: ")
