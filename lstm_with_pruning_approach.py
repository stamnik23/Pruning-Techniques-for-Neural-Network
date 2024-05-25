import numpy as np
import pandas as pd
import os
import logging
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

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

MODEL_DIRECTORY = "/home/stamou/rul/rul/battery-rul-estimation/results/pm/model/"
HISTORY_DIRECTORY = "/home/stamou/rul/rul/battery-rul-estimation/results/pm/history/"
IS_TRAINING = True
LATENT_DIM = 15
LEARNING_RATE = 0.0001

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

    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def train_and_evaluate(use_early_stopping):
    X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(CSV_FILES)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train_reshaped.shape[1:]
    model = build_lstm_model(input_shape)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=MODEL_DIRECTORY + 'best_model.keras', save_best_only=True, monitor='val_loss')

    callbacks = [model_checkpoint]
    if use_early_stopping:
        callbacks.append(early_stopping)

    start_time = time.time()
    history = model.fit(X_train_reshaped, y_train,
                        epochs=50,
                        batch_size=32,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=callbacks)
    training_time = time.time() - start_time

    results = model.evaluate(X_test_reshaped, y_test, return_dict=True)
    test_predictions = model.predict(X_test_reshaped).flatten()
    r2_test = r2_score(y_test, test_predictions)

    return results, r2_test, training_time

# Run the experiments and collect results
results_list = []

early_stopping_options = [False, True]

for use_early_stopping in early_stopping_options:
    results, r2_test, training_time = train_and_evaluate(use_early_stopping)
    results_list.append({
        'Early Stopping': 'Yes' if use_early_stopping else 'No',
        'MSE': results['mse'],
        'MAE': results['mae'],
        'RMSE': results['rmse'],
        'R2': r2_test,
        'Training Time': training_time
    })

results_df = pd.DataFrame(results_list)
print(results_df)

# Plotting the results
fig, axs = plt.subplots(1, 3, figsize=(21, 7))


axs[0].bar(results_df['Early Stopping'], results_df['RMSE'])
axs[0].set_title('Comparison of RMSE with and without Early Stopping')
axs[0].set_xlabel('Early Stopping')
axs[0].set_ylabel('RMSE')


axs[1].bar(results_df['Early Stopping'], results_df['R2'])
axs[1].set_title('Comparison of R2 with and without Early Stopping')
axs[1].set_xlabel('Early Stopping')
axs[1].set_ylabel('R2')

axs[2].bar(results_df['Early Stopping'], results_df['Training Time'])
axs[2].set_title('Comparison of Training Time (seconds) with and without Early Stopping')
axs[2].set_xlabel('Early Stopping')
axs[2].set_ylabel('Training Time (seconds)')

plt.tight_layout()
plt.show()
