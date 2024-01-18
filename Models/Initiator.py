from ML_Models import EnergyPricePredictor

# Instantiate the EnergyPricePredictor object
predictor = EnergyPricePredictor(
    model_type="bert", skip_Kfold=False, custom_learning_rate=0.00001
)

# Load the data
X_title, X_energy, y_energy, X_date_train, X_date_test = predictor.load_data(
    title_file="./Data/baselineData_PreProcessed_Clean.csv",
    energy_file="./Data/EnergyPrice_Clean.csv",
)

# Fit the model with k-fold cross-validation
predictor.fit_model(
    epochs=200,
    batch_size=32,
    verbose=1,
    n_splits=5,
    random_state=42,
)

# Save the model
model_name = f"./Models/{input('Input model name here: ')}.keras"
predictor.save_model(model_name)

# Load Model
predictor.load_model(model_name)

# Make predictions
y_energy_test, mae, rmse = predictor.predict()

# Plot the predictions
predictor.plot_predictions(mae, rmse)
