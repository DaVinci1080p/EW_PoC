import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import tensorflow as tf


class EnergyPricePredictor:
    def __init__(
        self,
        model_type="lstm",
        sequence_length_title=10,
        skip_Kfold=False,
        custom_learning_rate=0.0001,
    ):
        self.model_type = model_type
        self.sequence_length_title = sequence_length_title
        self.model = None
        self.tokenizer = None
        self.scaler = MinMaxScaler()
        self.predictions = None
        self.prediction_interval = None
        self.skip_Kfold = skip_Kfold
        self.custom_optimizer = tf.keras.optimizers.Adam(
            learning_rate=custom_learning_rate
        )

    def load_data(self, title_file, energy_file):
        df_titles = pd.read_csv(
            title_file, parse_dates=["Publication Date"], dayfirst=True
        )
        df_energy = pd.read_csv(energy_file, parse_dates=["Date"], dayfirst=True)

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

        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            title_sequences = merged_df["Title"].apply(
                lambda x: self.tokenizer.encode(x, add_special_tokens=True)
            )
            title_sequences = title_sequences.apply(
                lambda x: x[: self.sequence_length_title]
                + [0] * (self.sequence_length_title - len(x))
            )
        elif self.model_type == "lstm":
            self.max_words = 1000
            self.tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(merged_df["Title"])
            title_sequences = self.tokenizer.texts_to_sequences(merged_df["Title"])
            title_sequences = pad_sequences(
                title_sequences,
                maxlen=self.sequence_length_title,
                padding="post",
                truncating="post",
            )

        merged_df["EnergyPrice"] = self.scaler.fit_transform(merged_df[["EnergyPrice"]])

        # Define X (input features) and y (target variable)
        X_title = np.array(title_sequences.tolist())
        X_title = X_title[: -self.sequence_length_title]
        X_energy = []
        y_energy = []

        for i in range(len(merged_df) - self.sequence_length_title):
            X_sequence = merged_df.iloc[i : i + self.sequence_length_title][
                "EnergyPrice"
            ].values
            y_target = merged_df.iloc[i + self.sequence_length_title]["EnergyPrice"]

            X_energy.append(X_sequence)
            y_energy.append(y_target)
        dates = merged_df["Date"]
        dates = dates[: -self.sequence_length_title]

        X_energy = np.array(X_energy)
        y_energy = np.array(y_energy)
        (
            self.X_title_train,
            self.X_title_test,
            self.X_energy_train,
            self.X_energy_test,
            self.y_energy_train,
            self.y_energy_test,
            self.X_date_train,
            self.X_date_test,
        ) = train_test_split(
            X_title, X_energy, y_energy, dates, test_size=0.2, random_state=42
        )
        return (
            X_title,
            X_energy,
            y_energy,
            self.X_date_train,
            self.X_date_test,
        )

    def create_model(self):
        if self.model_type == "bert":
            self.create_bert()

        elif self.model_type == "lstm":
            self.create_lstm()

    def create_lstm(self):
        embed_dim = 50
        title_input = tf.keras.Input(shape=(self.sequence_length_title,))
        energy_input = tf.keras.Input(shape=(self.sequence_length_title, 1))
        title_embedding = Embedding(input_dim=self.max_words, output_dim=embed_dim)(
            title_input
        )
        title_lstm = tf.keras.layers.LSTM(50, activation="relu")(title_embedding)
        energy_lstm = tf.keras.layers.LSTM(50, activation="relu")(energy_input)

        merged = tf.keras.layers.Concatenate()([title_lstm, energy_lstm])
        output = tf.keras.layers.Dense(1)(merged)

        self.model = tf.keras.Model(inputs=[title_input, energy_input], outputs=output)
        self.model.compile(optimizer=self.custom_optimizer, loss="mse")

    def create_bert(self):
        embed_dim = self.X_title_train.shape[2]
        title_input = tf.keras.Input(
            shape=(self.sequence_length_title, embed_dim), dtype=tf.float32
        )
        energy_input = tf.keras.Input(shape=(self.sequence_length_title, 1))

        title_lstm = tf.keras.layers.LSTM(50, activation="relu")(title_input)

        energy_lstm = tf.keras.layers.LSTM(50, activation="relu")(energy_input)

        merged = tf.keras.layers.Concatenate()([title_lstm, energy_lstm])
        output = tf.keras.layers.Dense(1)(merged)

        self.model = tf.keras.Model(inputs=[title_input, energy_input], outputs=output)
        self.model.compile(optimizer="adam", loss="mse")

    def fit_model(
        self, epochs=50, batch_size=32, verbose=1, n_splits=5, random_state=42
    ):
        if self.model_type == "bert":
            model = TFBertModel.from_pretrained("bert-base-uncased")
            self.X_title_train = model.predict(self.X_title_train)[0]
            self.X_title_test = model.predict(self.X_title_test)[0]

        # Kfold cross validation if skip_Kfold is not True
        if not self.skip_Kfold:
            self.cross_validation(epochs, batch_size, verbose, n_splits, random_state)

        # Train model on training data for predictions
        self.create_model()
        self.model.fit(
            [self.X_title_train, self.X_energy_train],
            self.y_energy_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def cross_validation(
        self, epochs=50, batch_size=32, verbose=1, n_splits=5, random_state=42
    ):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_index, val_index in kf.split(self.X_title_train):
            X_title_train_fold, X_title_val_fold = (
                self.X_title_train[train_index],
                self.X_title_train[val_index],
            )
            X_energy_train_fold, X_energy_val_fold = (
                self.X_energy_train[train_index],
                self.X_energy_train[val_index],
            )
            y_energy_train_fold, y_energy_val_fold = (
                self.y_energy_train[train_index],
                self.y_energy_train[val_index],
            )

            self.create_model()
            self.model.fit(
                [X_title_train_fold, X_energy_train_fold],
                y_energy_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
            )

            # Evaluate on validation set
            val_predictions = self.model.predict([X_title_val_fold, X_energy_val_fold])
            val_predictions = self.scaler.inverse_transform(val_predictions)
            y_energy_val_fold = self.scaler.inverse_transform(
                y_energy_val_fold.reshape(-1, 1)
            )
            val_mae = mean_absolute_error(y_energy_val_fold, val_predictions)
            val_rmse = np.sqrt(mean_squared_error(y_energy_val_fold, val_predictions))
            print(f"Validation MAE: {val_mae:.4f}, Validation RMSE: {val_rmse:.4f}")

    def save_model(self, model_file):
        self.model.save(model_file, save_format="tf")

    def load_model(self, model_file):
        self.model = tf.keras.models.load_model(model_file)

    def predict(self):
        self.predictions = self.model.predict([self.X_title_test, self.X_energy_test])
        self.predictions = self.scaler.inverse_transform(self.predictions)
        self.y_energy_test = self.scaler.inverse_transform(
            self.y_energy_test.reshape(-1, 1)
        )
        mae = mean_absolute_error(self.y_energy_test, self.predictions)
        rmse = np.sqrt(mean_squared_error(self.y_energy_test, self.predictions))
        return self.y_energy_test, mae, rmse

    def plot_predictions(self, mae, rmse):
        # Sort the data by date to ensure chronological order
        sorted_indices = np.argsort(self.X_date_test)
        self.X_date_test = self.X_date_test.iloc[sorted_indices].values
        self.y_energy_test = self.y_energy_test[sorted_indices]
        self.predictions = self.predictions[sorted_indices]

        # Line plot of the actual and predicted energy prices with dates
        plt.figure(figsize=(12, 6))
        plt.plot(
            self.X_date_test,
            self.y_energy_test,
            label="Actual Energy Price",
            color="b",
            alpha=0.7,
            linestyle="-",
        )
        plt.plot(
            self.X_date_test,
            self.predictions,
            label="Predicted Energy Price",
            color="r",
            alpha=0.7,
            linestyle="-",
        )

        # Calculate a dynamic 95% prediction interval
        coverage_probability = self.calculate_coverage_probability()

        # Plot the dynamic 95% prediction interval as a shaded area
        plt.fill_between(
            self.X_date_test,
            self.lower_bound,
            self.upper_bound,
            color="gray",
            alpha=0.5,
            label="Dynamic 95% Prediction Interval",
        )

        plt.xlabel("Date")
        plt.ylabel("Energy Price")
        plt.legend(loc="upper left")
        self.plot_info(
            f"Actual vs. Predicted Energy Prices with Dynamic 95% Prediction Interval ( {self.model_type} Trained on Price and Article Title data)",
            mae,
            rmse,
            coverage_probability,
        )
        # Scatter plot of the actual and predicted energy prices with dates
        plt.figure(figsize=(12, 6))
        plt.scatter(
            self.X_date_test,
            self.y_energy_test,
            label="Actual Energy Price",
            marker="o",
            s=30,
            c="b",
            alpha=0.7,
        )
        plt.scatter(
            self.X_date_test,
            self.predictions,
            label="Predicted Energy Price",
            marker="x",
            s=30,
            c="r",
            alpha=0.7,
        )

        plt.xlabel("Date")
        plt.ylabel("Energy Price")
        plt.legend()
        self.plot_info(
            f"Actual vs. Predicted Energy Prices ({self.model_type} Trained on Price and Article Title data)",
            mae,
            rmse,
            coverage_probability,
        )

    def plot_info(self, title, mae, rmse, coverage_probability):
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Add MAE and RMSE as comments
        plt.text(0.05, 0.8, f"MAE: {mae:.4f}", transform=plt.gca().transAxes)
        plt.text(0.05, 0.75, f"RMSE: {rmse:.4f}", transform=plt.gca().transAxes)
        plt.text(
            0.05,
            0.70,
            f"Coverage Probability: {coverage_probability}",
            transform=plt.gca().transAxes,
        )

        # Show the line plot
        plt.show()

    def calculate_coverage_probability(self):
        rolling_std = pd.Series(self.predictions.squeeze()).rolling(window=5).std()
        rolling_predictions = (
            pd.Series(self.predictions.squeeze()).rolling(window=5).mean()
        )
        self.lower_bound = (rolling_predictions - 1.96 * rolling_std).values
        self.upper_bound = (rolling_predictions + 1.96 * rolling_std).values

        n = len(self.y_energy_test)
        observations_within_interval = sum(
            self.lower_bound[i] <= self.y_energy_test[i] <= self.upper_bound[i]
            for i in range(n)
        )
        coverage_probability = observations_within_interval / n * 100
        return "{:.2f}%".format(float(coverage_probability))
