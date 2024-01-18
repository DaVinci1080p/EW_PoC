import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Embedding, Dense, Input, Concatenate, Masking
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your CSV files with the correct date parsing format
df_energy = pd.read_csv(
    "./Data/EnergyPrice_Clean.csv", parse_dates=["Date"], dayfirst=True
)
# Normalize the 'EnergyPrice' values
scaler = MinMaxScaler()
df_energy["EnergyPrice"] = scaler.fit_transform(df_energy[["EnergyPrice"]])

sequence_length = 10

dates = df_energy["Date"]
dates = dates[:-sequence_length]

# Convert datetime values to timestamps
df_energy["Date"] = df_energy["Date"].apply(lambda x: x.timestamp())

# Define X (input features) and y (target variable)
X_energy = []
y_energy = []

for i in range(len(df_energy) - sequence_length):
    X_sequence = df_energy.iloc[i : i + sequence_length]["EnergyPrice"].values
    y_target = df_energy.iloc[i + sequence_length]["EnergyPrice"]

    X_energy.append(X_sequence)
    y_energy.append(y_target)

X_energy = np.array(X_energy)
y_energy = np.array(y_energy)

# Split the data into training and testing sets
(
    X_energy_train,
    X_energy_test,
    y_energy_train,
    y_energy_test,
    X_date_train,
    X_date_test,
) = train_test_split(X_energy, y_energy, dates, test_size=0.2, random_state=42)

# Define the dimensionality of the title embeddings
embed_dim = 50  # Adjust as needed

# Create an LSTM model
model = Sequential()

# Add an LSTM layer for processing the energy data
model.add(LSTM(50, activation="relu", input_shape=(sequence_length, 1)))

# Add a Dense layer for regression
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mse")

'''
# Train the model
model.fit(X_energy_train, y_energy_train, epochs=50, batch_size=32, verbose=1)
model.save('./Models/LSTM_Baseline_Price.keras')
'''

loaded_model = load_model("./Models/LSTM_Baseline_Price.keras")

# Make predictions on the test set
predictions = loaded_model.predict(X_energy_test)

predictions = scaler.inverse_transform(predictions)
y_energy_test = y_energy_test.reshape(-1, 1)
y_energy_test = scaler.inverse_transform(y_energy_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_energy_test, predictions)
print(f"Mean Absolute Error (MAE): {mae}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_energy_test, predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
plt.scatter(
    X_date_test,
    y_energy_test,
    label="Actual Energy Price",
    marker="o",
    s=30,
    c="b",
    alpha=0.7,
)
plt.scatter(
    X_date_test,
    predictions,
    label="Predicted Energy Price",
    marker="x",
    s=30,
    c="r",
    alpha=0.7,
)
plt.xlabel("Date")
plt.ylabel("Energy Price")
plt.legend()
plt.title("Actual vs. Predicted Energy Prices (Trained on Price data)")
plt.xticks(rotation=45)
plt.tight_layout()

# Add MAE and RMSE as comments
plt.text(0.05, 0.8, f"MAE: {mae:.4f}", transform=plt.gca().transAxes)
plt.text(0.05, 0.75, f"RMSE: {rmse:.4f}", transform=plt.gca().transAxes)


# Show the scatterplot
plt.show()