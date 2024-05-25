import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import nni
import tensorflow_model_optimization as tfmot
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

MODEL_DIRECTORY = "/home/stamou/rul/rul/battery-rul-estimation/results/pm/model/"
HISTORY_DIRECTORY = "/home/stamou/rul/rul/battery-rul-estimation/results/pm/history/"
IS_TRAINING = True

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


class PrunableConv1DTranspose(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation=None, **kwargs):
        super(PrunableConv1DTranspose, self).__init__(**kwargs)
        self.conv_transpose = layers.Conv1DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
    
    def call(self, inputs):
        return self.conv_transpose(inputs)

    def get_prunable_weights(self):
        return [self.conv_transpose.weights[0]]

#  Autoencoder model
def build_autoencoder(latent_dim, input_shape, pruning_method):
    pruning_params = {}
    if pruning_method == 'constant':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5, 0, 10000, frequency=100)
        }
    elif pruning_method == 'polynomial':
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                     final_sparsity=0.5,
                                                                     begin_step=0,
                                                                     end_step=10000)
        }

    x = layers.Input(shape=input_shape)
    encoded = layers.Conv1D(filters=32, kernel_size=2, strides=1, activation='relu', padding='same')(x)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Conv1D(filters=16, kernel_size=2, strides=1, activation='relu', padding='same')(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    encoded = layers.Conv1D(filters=8, kernel_size=2, strides=1, activation='relu', padding='same')(encoded)
    encoded = layers.Flatten()(encoded)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)

    decoded = layers.Dense(units=input_shape[0] * 8, activation='relu')(encoded)
    decoded = layers.Reshape((input_shape[0], 8))(decoded)
    decoded = PrunableConv1DTranspose(filters=8, kernel_size=2, strides=1, activation='relu', padding='same')(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = PrunableConv1DTranspose(filters=16, kernel_size=2, strides=1, activation='relu', padding='same')(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    decoded = PrunableConv1DTranspose(filters=32, kernel_size=2, strides=1, activation='relu', padding='same')(decoded)
    decoded = PrunableConv1DTranspose(filters=1, kernel_size=2, strides=1, padding='same')(decoded)

    autoencoder = Model(x, decoded)

    if pruning_method in ['constant', 'polynomial']:
        autoencoder = tfmot.sparsity.keras.prune_low_magnitude(autoencoder, **pruning_params)
    elif pruning_method == 'l1_regularization':
        for layer in autoencoder.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l1(0.01)
    elif pruning_method == 'l2_regularization':
        for layer in autoencoder.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = tf.keras.regularizers.l2(0.01)

    return autoencoder

class NniReporterCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(NniReporterCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and 'val_loss' in logs:
       
            val_predictions = self.model.predict(self.validation_data[0])
            val_true = self.validation_data[1]

            val_r2 = r2_score(val_true.flatten(), val_predictions.flatten())

            nni.report_intermediate_result({
                'default': logs['val_loss'],  # Primary metric for the assessor
                'val_mse': logs['val_mse'],
                'val_mae': logs['val_mae'],
                'val_rmse': np.sqrt(logs['val_mse']),  # RMSE is not directly available, so compute it
                'val_r2': val_r2  # Report R²
            })

def train_and_evaluate(params, use_early_stopping):
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    latent_dim = params['latent_dim']
    pruning_method = params['pruning_method']

    X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(CSV_FILES)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train_reshaped.shape[1:]

    autoencoder = build_autoencoder(latent_dim, input_shape, pruning_method)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

    callbacks = []
    if use_early_stopping:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks.append(early_stopping)
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    callbacks.append(pruning_callback)

    start_time = time.time()
    history = autoencoder.fit(X_train_reshaped, X_train_reshaped,
                              epochs=50,
                              batch_size=batch_size,
                              verbose=1,
                              validation_split=0.1,
                              callbacks=callbacks)
    end_time = time.time()
    training_time = end_time - start_time

    autoencoder.save_weights(MODEL_DIRECTORY + 'model.weights.h5')
    autoencoder.save(MODEL_DIRECTORY + 'model.keras')
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = HISTORY_DIRECTORY + 'history.csv'
    hist_df.to_csv(hist_csv_file, index=False)

    # Measure inference time
    inference_start_time = time.time()
    predictions = autoencoder.predict(X_test_reshaped)
    inference_end_time = time.time()
    inference_time = inference_end_time - inference_start_time

    mse = mean_squared_error(X_test_reshaped.flatten(), predictions.flatten())
    mae = mean_absolute_error(X_test_reshaped.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(X_test_reshaped.flatten(), predictions.flatten())

    return mse, mae, rmse, r2, training_time, inference_time

def main(params):
    results = []

    # No early stopping, no pruning
    mse, mae, rmse, r2, training_time, inference_time = train_and_evaluate(params, use_early_stopping=False)
    results.append(['No Pruning', 'No Early Stopping', mse, mae, rmse, r2, training_time, inference_time])

    # Early stopping, no pruning
    mse, mae, rmse, r2, training_time, inference_time = train_and_evaluate(params, use_early_stopping=True)
    results.append(['No Pruning', 'Early Stopping', mse, mae, rmse, r2, training_time, inference_time])

    # Early stopping, with pruning
    pruning_methods = ['constant', 'polynomial', 'l1_regularization', 'l2_regularization']
    for pruning_method in pruning_methods:
        params['pruning_method'] = pruning_method
        mse, mae, rmse, r2, training_time, inference_time = train_and_evaluate(params, use_early_stopping=True)
        results.append([pruning_method, 'Early Stopping', mse, mae, rmse, r2, training_time, inference_time])

    results_df = pd.DataFrame(results, columns=['Pruning Method', 'Early Stopping', 'MSE', 'MAE', 'RMSE', 'R2', 'Training Time (seconds)', 'Inference Time (seconds)'])
    print(results_df)

    # Plotting RMSE
    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(len(pruning_methods) + 2)  
    bar_width = 0.35

    no_pruning = results_df[results_df['Pruning Method'] == 'No Pruning']
    with_pruning = results_df[results_df['Pruning Method'] != 'No Pruning']

    bar1 = ax.bar(index[:2], no_pruning['RMSE'], bar_width, label='No Pruning')
    bar2 = ax.bar(index[2:], with_pruning['RMSE'], bar_width, label='Pruning')

    ax.set_xlabel('Pruning Method and Early Stopping')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE by Pruning Method and Early Stopping')
    ax.set_xticks(index)
    ax.set_xticklabels(['No Pruning\nNo Early Stopping', 'No Pruning\nEarly Stopping'] + list(with_pruning['Pruning Method'] + '\nEarly Stopping'))
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Plotting R2
    fig, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(index[:2], no_pruning['R2'], bar_width, label='No Pruning')
    bar2 = ax.bar(index[2:], with_pruning['R2'], bar_width, label='Pruning')

    ax.set_xlabel('Pruning Method and Early Stopping')
    ax.set_ylabel('R2 Score')
    ax.set_title('R2 Score by Pruning Method and Early Stopping')
    ax.set_xticks(index)
    ax.set_xticklabels(['No Pruning\nNo Early Stopping', 'No Pruning\nEarly Stopping'] + list(with_pruning['Pruning Method'] + '\nEarly Stopping'))
    ax.legend()

    plt.tight_layout()
    plt.show()


params = nni.get_next_parameter()
print(params)
main(params)
