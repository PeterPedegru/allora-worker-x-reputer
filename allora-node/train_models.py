import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from sklearn.model_selection import TimeSeriesSplit
from ta import add_all_ta_features

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Загрузка данных
def load_data(token_name):
    try:
        data = pd.read_csv(f'data/{token_name}.csv')
        logging.info(f"Data for {token_name} loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data for {token_name}: {e}")
        return None

# Подготовка данных
def prepare_data(data, look_back, feature_columns):
    # Добавление технических индикаторов
    data = add_all_ta_features(
        data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[feature_columns])
    
    X, Y = [], []
    for i in range(len(data_scaled) - look_back):
        X.append(data_scaled[i:i+look_back])
        Y.append(data_scaled[i+look_back, 0])  # Предсказываем цену закрытия
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, scaler

# Создание модели
def create_model(look_back, input_dim, model_type='LSTM'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(100, return_sequences=True, input_shape=(look_back, input_dim)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(100, return_sequences=True, input_shape=(look_back, input_dim)))
        model.add(Dropout(0.2))
        model.add(GRU(100, return_sequences=False))
    
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Функция тренировки и сохранения модели
def train_and_save_model(token_name, look_back, horizon, model_type='LSTM'):
    try:
        data = load_data(token_name)
        if data is None:
            return
        
        feature_columns = ['Close', 'Volume', 'volatility_bbm', 'trend_macd', 'momentum_rsi']  # Добавлены технические индикаторы
        X, Y, scaler = prepare_data(data, look_back, feature_columns)
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_loss = float('inf')
        
        for train_index, val_index in tscv.split(X):
            X_train, X_val = X[train_index], X[val_index]
            Y_train, Y_val = Y[train_index], Y[val_index]
            
            model = create_model(look_back, len(feature_columns), model_type)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=2, validation_data=(X_val, Y_val), callbacks=[early_stopping])
            
            val_loss = model.evaluate(X_val, Y_val, verbose=0)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        
        best_model.save(f'models/{token_name}_{horizon}_model.h5')
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
        train_and_save_model(token, look_back, horizon_name, model_type='GRU')
