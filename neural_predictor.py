import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf  # You'll need: pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from loguru import logger

class LSTMPredictor:
    """Deep Learning price predictor using LSTM"""
    
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler() 
        self.is_trained = False
        
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        logger.info("LSTM model built successfully")
        
    def prepare_data(self, data: pd.DataFrame):
        """Prepare sequences for LSTM"""
        prices = data['close'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_prices)):
            X.append(scaled_prices[i-self.sequence_length:i, 0])
            y.append(scaled_prices[i, 0])
        
        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def train(self, data: pd.DataFrame, epochs=50):
        """Train the LSTM model"""
        if len(data) < self.sequence_length + 100:
            logger.warning("Not enough data for LSTM training")
            return
        
        if self.model is None:
            self.build_model()
        
        X, y = self.prepare_data(data)
        
        # Split into train/test
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=0
        )
        
        self.is_trained = True
        logger.success(f"LSTM trained - Final loss: {history.history['loss'][-1]:.6f}")
        
    def predict_next(self, recent_data: pd.DataFrame) -> Dict:
        """Predict next price movement"""
        if not self.is_trained or len(recent_data) < self.sequence_length:
            return {'signal': 0, 'confidence': 0.5}
        
        # Prepare last sequence
        prices = recent_data['close'].values[-self.sequence_length:].reshape(-1, 1)
        scaled = self.scaler.transform(prices)
        X = np.array(scaled).reshape(1, self.sequence_length, 1)
        
        # Predict
        predicted_scaled = self.model.predict(X, verbose=0)[0, 0]
        
        # Inverse transform to get actual price
        predicted_price = self.scaler.inverse_transform([[predicted_scaled]])[0, 0]
        current_price = recent_data['close'].iloc[-1]
        
        price_change = (predicted_price - current_price) / current_price
        
        signal = 1 if price_change > 0.01 else -1 if price_change < -0.01 else 0
        confidence = min(1.0, abs(price_change) * 50)  # Scale confidence
        
        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_pct': price_change * 100
        }