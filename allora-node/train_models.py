import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка данных
def load_data(token_name):
    try:
        data = pd.read_csv(f'data/{token_name}.csv')
        logging.info(f"Data for {token_name} loaded successfully.")
        return data['Close'].values
    except Exception as e:
        logging.error(f"Error loading data for {token_name}: {e}")
        return None

# Подготовка данных
def prepare_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    X, Y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back])
        Y.append(data_scaled[i+look_back])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, scaler

# Создание модели
def create_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Функция тренировки и сохранения модели
def train_and_save_model(token_name, look_back, horizon):
    try:
        data = load_data(token_name)
        if data is None:
            return

        X, Y, scaler = prepare_data(data, look_back)
        
        model = create_model(look_back)
        
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, Y, epochs=50, batch_size=32, verbose=2, callbacks=[early_stopping])
        
        model.save(f'models/{token_name}_{horizon}_model.h5')
        joblib.dump(scaler, f'models/{token_name}_{horizon}_scaler.pkl')
        
        logging.info(f"Model and scaler for {token_name} with {horizon} horizon saved successfully.")
    except Exception as e:
        logging.error(f"Error training model for {token_name} with {horizon} horizon: {e}")

# Пример использования
horizons = {
    '10min': 10,
    '20min': 20,
    '24h': 24 * 60
}

tokens = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD', 'ARBUSD']

for token in tokens:
    for horizon_name, look_back in horizons.items():
        train_and_save_model(token, look_back, horizon_name)
