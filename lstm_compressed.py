import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

# Constants
CSV_FILES = [
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00037-tid-2487379836564725268-a2d53787-d3fe-42ed-a49e-19f061d3ee08-4211-2.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00038-tid-2487379836564725268-a2d53787-d3fe-42ed-a49e-19f061d3ee08-4235-3.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00053-tid-7300699832233812418-b4be234d-4133-435e-89d2-2263e14852f9-2937-4.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00054-tid-7300699832233812418-b4be234d-4133-435e-89d2-2263e14852f9-2962-2.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00055-tid-7300699832233812418-b4be234d-4133-435e-89d2-2263e14852f9-2961-2.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00056-tid-7300699832233812418-b4be234d-4133-435e-89d2-2263e14852f9-2966-2.c000.csv",
    "/home/stamou/rul/rul/battery-rul-estimation/data/datanew/part-00057-tid-7300699832233812418-b4be234d-4133-435e-89d2-2263e14852f9-2940-3.c000.csv",
]

# Load and preprocess data
def load_and_preprocess_data(files):
    dfs = []
    for csv_file in files:
        data = pd.read_csv(csv_file)
        selected_columns = [
            "MainEngineLubOilInletPressure_ControlAlarmMonitoringSystem_Instant_bar",
            "MainEngineAirSpringAirPressure_ControlAlarmMonitoringSystem_Instant_bar",
            "MainEngineTurbochargerLubOilInletPressure_ControlAlarmMonitoringSystem_Instant_bar",
            "RUL"
        ]
        data = data[selected_columns]
        data.dropna(inplace=True)
        dfs.append(data)
    all_data = pd.concat(dfs, ignore_index=True)
    X = all_data.drop(columns=["RUL"])
    y = all_data["RUL"]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled, scaler_X, scaler_y

#  LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, activation='relu', return_sequences=True),
        layers.LSTM(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

#  Autoencoder
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(1, 3)),  # Adjust the input shape to match your data
            layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.Conv1D(filters=16, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.Conv1D(filters=8, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(units=1 * 8, activation='relu'),
            layers.Reshape((1, 8)),
            layers.Conv1DTranspose(filters=8, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.Conv1DTranspose(filters=16, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Dropout(0.2),
            layers.Conv1DTranspose(filters=32, kernel_size=2, strides=1, activation='relu', padding='same'),
            layers.Conv1DTranspose(filters=3, kernel_size=2, strides=1, padding='same')
        ])
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Train and evaluate Autoencoder
def train_and_evaluate_autoencoder(train_x, val_x, test_x, latent_dim):
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                        loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    autoencoder.fit(train_x, train_x, epochs=50, batch_size=32, validation_data=(val_x, val_x))
    train_x_encoded = autoencoder.encoder(train_x).numpy()
    val_x_encoded = autoencoder.encoder(val_x).numpy()
    test_x_encoded = autoencoder.encoder(test_x).numpy()
    return train_x_encoded, val_x_encoded, test_x_encoded

# Train and evaluate LSTM Model on compressed data
def train_and_evaluate_lstm(train_x, val_x, test_x, train_y, val_y, test_y):
    input_shape = (train_x.shape[1], train_x.shape[2])  # Adjust input shape for LSTM
    lstm_model = build_lstm_model(input_shape)
    lstm_model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(val_x, val_y))
    test_predictions = lstm_model.predict(test_x).flatten()
    mse = mean_squared_error(test_y, test_predictions)
    mae = mean_absolute_error(test_y, test_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, test_predictions)
    return mse, mae, rmse, r2

def main():
    X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(CSV_FILES)
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    timesteps = 1  
    features = X_train.shape[1]

    X_train = X_train.reshape((X_train.shape[0], timesteps, features))
    X_val = X_val.reshape((X_val.shape[0], timesteps, features))
    X_test = X_test.reshape((X_test.shape[0], timesteps, features))

    # Train and evaluate Autoencoder
    latent_dim = 15
    train_x_encoded, val_x_encoded, test_x_encoded = train_and_evaluate_autoencoder(X_train, X_val, X_test, latent_dim)

    train_x_encoded = train_x_encoded.reshape((train_x_encoded.shape[0], 1, train_x_encoded.shape[1]))
    val_x_encoded = val_x_encoded.reshape((val_x_encoded.shape[0], 1, val_x_encoded.shape[1]))
    test_x_encoded = test_x_encoded.reshape((test_x_encoded.shape[0], 1, test_x_encoded.shape[1]))

    mse_comp, mae_comp, rmse_comp, r2_comp = train_and_evaluate_lstm(train_x_encoded, val_x_encoded, test_x_encoded, y_train, y_val, y_test)
    print(f"LSTM on Compressed Data - MSE: {mse_comp}, MAE: {mae_comp}, RMSE: {rmse_comp}, R2: {r2_comp}")

if __name__ == "__main__":
    main()
