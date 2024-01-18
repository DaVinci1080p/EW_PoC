import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler

# Load your CSV files with the correct date parsing format
df1 = pd.read_csv(
    "Data/baselineData_PreProcessed_Clean.csv",
    parse_dates=["Publication Date"],
    dayfirst=True,
)
df2 = pd.read_csv("Data/EnergyPrice_Clean.csv", parse_dates=["Date"], dayfirst=True)

# Check the structure of the DataFrames
print("DataFrame 1 (URL, Title, Publication Date):")
print(df1.head())

print("\nDataFrame 2 (Date, EnergyPrice):")
print(df2.head())

# Merge DataFrames based on a common column (e.g., Date)
merged_df = pd.merge(df1, df2, left_on="Publication Date", right_on="Date")

# Check the merged DataFrame
print("\nMerged DataFrame:")
print(merged_df.head())

# Select the "Title" and "EnergyPrice" columns
data = merged_df[["Title", "EnergyPrice"]].copy()

# Convert the "EnergyPrice" column to integers
data["EnergyPrice"] = data["EnergyPrice"].astype(int)

"""
# Check the data type of the "EnergyPrice" column
energy_price_data_type = data['EnergyPrice'].dtype

# Print the data type
print("Data type of 'EnergyPrice' column:", energy_price_data_type)
"""

# Tokenize the "Title" column
max_words = 100  # You can adjust this based on your vocabulary size
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data["Title"])
title_sequences = tokenizer.texts_to_sequences(data["Title"])

# Define a sequence length (e.g., 7 days) for predicting trends
sequence_length = 7
title_sequences = pad_sequences(title_sequences, maxlen=sequence_length)

# Normalize the 'EnergyPrice' values
scaler = MinMaxScaler()
data["EnergyPrice"] = scaler.fit_transform(data[["EnergyPrice"]])

# Combine the title and EnergyPrice data
X_title = title_sequences
X_energy = data[["EnergyPrice"]].values
X_combined = np.hstack((X_title, X_energy))
print("X_Combined........................", X_combined.shape)
# Create sequences of data with a rolling window
sequences = []
target = []

for i in range(len(X_combined) - sequence_length):
    sequences.append(X_combined[i : i + sequence_length])
    target.append(X_combined[i + sequence_length])

# Convert to numpy arrays
X = np.array(sequences)
y = np.array(target)
print("X and y: ", X, y)

# Split the data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train an LSTM model
model = Sequential()
model.add(
    LSTM(50, activation="relu", input_shape=(sequence_length, X_combined.shape[1]))
)
# Output dimension matches input dimension
model.add(Dense(X_combined.shape[1]))
model.compile(optimizer="adam", loss="mse")

model.fit(X_train, y_train, epochs=50, batch_size=100, verbose=1)

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the scaled test set to get actual EnergyPrice values
y_test_inverse = scaler.inverse_transform(y_test[:, -1].reshape(-1, 1)).flatten()

# Inverse transform the scaled predictions to get actual EnergyPrice values
predictions_inverse = scaler.inverse_transform(
    predictions[:, -1].reshape(-1, 1)
).flatten()

# Plot the actual and predicted values
plt.figure(figsize=(12, 6))
plt.plot(
    data.index[train_size + sequence_length :],
    y_test_inverse,
    label="Actual EnergyPrice",
    marker="o",
)
plt.plot(
    data.index[train_size + sequence_length :],
    predictions_inverse,
    label="Predicted EnergyPrice",
    marker="x",
)
plt.xlabel("Date")
plt.ylabel("EnergyPrice")
plt.legend()
plt.show()
