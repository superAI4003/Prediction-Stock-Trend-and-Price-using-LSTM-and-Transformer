import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from itertools import product

# Load dataset
data = pd.read_csv('datasets/AMD_raw.csv')

# Preprocess data
def preprocess_data(data):
    # Convert datetime to pandas datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    
    # Select features and target
    features = ['open', 'high', 'low', 'close', 'returns', 'log_ma_7', 'log_ma_14', 'log_ma_28', 'ema_crossover', 'macdh']
    target = 'predict'
    
    # Encode target variable
    data[target] = data[target].apply(lambda x: 1 if x == 'high' else 0)
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    return scaled_features, data[target].values

# Prepare data for LSTM
def create_sequences(data, target, n_timesteps=14):
    X, y = [], []
    for i in range(len(data) - n_timesteps):
        X.append(data[i:i + n_timesteps])
        y.append(target[i + n_timesteps])
    return np.array(X), np.array(y)

# Load and preprocess data
scaled_features, target = preprocess_data(data)
X, y = create_sequences(scaled_features, target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ETL class to hold training and testing data
class ETL:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

etl = ETL(X_train, X_test, y_train, y_test)

# Build, compile, and fit LSTM model
def build_lstm(etl: ETL, n_timesteps, n_features, n_outputs, units, dropout_rate, epochs, batch_size) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Builds, compiles, and fits our LSTM baseline model.
    """
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_outputs, activation='sigmoid'))  # Use sigmoid for binary classification
    
    print('compiling baseline model...')
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    print('fitting model...')
    history = model.fit(etl.X_train, etl.y_train, batch_size=batch_size, epochs=epochs,
                        validation_data=(etl.X_test, etl.y_test), verbose=1, callbacks=callbacks)
    
    return model, history

# Define hyperparameter grid
param_grid = {
    'units': [50],
    'dropout_rate': [0.2],
    'epochs': [50],
    'batch_size': [32]
}

# Generate all combinations of hyperparameters
param_combinations = list(product(param_grid['units'], param_grid['dropout_rate'], param_grid['epochs'], param_grid['batch_size']))

# Initialize variables to store the best model and its performance
best_model = None
best_history = None
best_accuracy = 0

# Iterate over all combinations of hyperparameters
for units, dropout_rate, epochs, batch_size in param_combinations:
    print(f'Testing combination: units={units}, dropout_rate={dropout_rate}, epochs={epochs}, batch_size={batch_size}')
    model, history = build_lstm(etl, 14, 10, 1, units, dropout_rate, epochs, batch_size)
    
    # Check if this model has the best validation accuracy
    val_accuracy = max(history.history['val_accuracy'])
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model
        best_history = history

print(f'Best model found with validation accuracy: {best_accuracy}')