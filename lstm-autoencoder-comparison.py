import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
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

MODEL_DIRECTORY = "./models/"
HISTORY_DIRECTORY = "./history/"
LATENT_DIM = 15

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

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, activation='relu', return_sequences=True),
        layers.LSTM(64, activation='relu'),
        layers.Dense(1)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_autoencoder(latent_dim, input_shape):
    class Autoencoder(Model):
        def __init__(self, latent_dim, input_shape):
            super(Autoencoder, self).__init__()
            self.encoder = tf.keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Dropout(0.2),
                layers.Conv1D(filters=16, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Dropout(0.2),
                layers.Conv1D(filters=8, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Flatten(),
                layers.Dense(latent_dim, activation='relu')
            ])
            self.decoder = tf.keras.Sequential([
                layers.Dense(units=input_shape[0] * 8, activation='relu'),
                layers.Reshape((input_shape[0], 8)),
                layers.Conv1DTranspose(filters=8, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Dropout(0.2),
                layers.Conv1DTranspose(filters=16, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Dropout(0.2),
                layers.Conv1DTranspose(filters=32, kernel_size=2, strides=1, activation='relu', padding='same'),
                layers.Conv1DTranspose(filters=1, kernel_size=2, strides=1, padding='same')
            ])

        def call(self, inputs):
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = Autoencoder(latent_dim, input_shape)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return autoencoder

def train_and_evaluate(model, X_train, X_test, y_train, y_test, use_early_stopping, model_type):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=MODEL_DIRECTORY + 'best_model', save_best_only=True, monitor='val_loss', save_format='tf')
    callbacks = [model_checkpoint]
    if use_early_stopping:
        callbacks.append(early_stopping)

    start_time = time.time()
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)
    training_time = time.time() - start_time

    results = model.evaluate(X_test, y_test, return_dict=True)
    test_predictions = model.predict(X_test)

    if model_type == 'LSTM':
        test_predictions = test_predictions.flatten()  # Flatten for LSTM
    else:
        test_predictions = test_predictions.reshape(-1)
        y_test = y_test.reshape(-1)

    r2_test = r2_score(y_test, test_predictions)

    return results, r2_test, training_time

def main():
    X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(CSV_FILES)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Reshape input data to include time steps
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train_reshaped.shape[1:]

    # Training and evaluating LSTM
    lstm_model = build_lstm_model(input_shape)
    lstm_results_list = []

    for use_early_stopping in [False, True]:
        lstm_results, lstm_r2_test, lstm_training_time = train_and_evaluate(
            lstm_model, X_train_reshaped, X_test_reshaped, y_train, y_test, use_early_stopping, 'LSTM')
        lstm_results_list.append({
            'Model': 'LSTM',
            'Early Stopping': 'Yes' if use_early_stopping else 'No',
            'MSE': lstm_results['mse'],
            'MAE': lstm_results['mae'],
            'RMSE': lstm_results['rmse'],
            'R2': lstm_r2_test,
            'Training Time (seconds)': lstm_training_time
        })

    # Training and evaluating Autoencoder
    autoencoder_model = build_autoencoder(LATENT_DIM, input_shape)
    autoencoder_results_list = []

    for use_early_stopping in [False, True]:
        autoencoder_results, autoencoder_r2_test, autoencoder_training_time = train_and_evaluate(
            autoencoder_model, X_train_reshaped, X_test_reshaped, X_train_reshaped, X_test_reshaped, use_early_stopping, 'Autoencoder')
        autoencoder_results_list.append({
            'Model': 'Autoencoder',
            'Early Stopping': 'Yes' if use_early_stopping else 'No',
            'MSE': autoencoder_results['mse'],
            'MAE': autoencoder_results['mae'],
            'RMSE': autoencoder_results['rmse'],
            'R2': autoencoder_r2_test,
            'Training Time (seconds)': autoencoder_training_time
        })

    # Combine results into a single DataFrame
    results_df = pd.DataFrame(lstm_results_list + autoencoder_results_list)
    print(results_df)

if __name__ == "__main__":
    main()
