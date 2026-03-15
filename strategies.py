import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import ta
from loguru import logger
from config import config

# ======================= ДОБАВЛЕННАЯ ФУНКЦИЯ ATR =======================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Рассчитывает Average True Range (ATR) для заданного DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame с колонками 'high', 'low', 'close'.
    period (int): Период для расчёта ATR (по умолчанию 14).
    
    Returns:
    pd.Series: Значения ATR.
    """
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr
# =======================================================================

@dataclass
class Signal:
    symbol: str
    signal: int
    confidence: float
    strength: float
    reasons: List[str]
    price: float
    timestamp: pd.Timestamp
    strategy_name: str

class TechnicalIndicatorCalculator:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        return df

class SmartIndicatorStrategy:
    def __init__(self, volatility_threshold: float = None):
        # если не передан, берём из config
        if volatility_threshold is None:
            volatility_threshold = config.VOLATILITY_THRESHOLD
        self.name = "Smart Indicator"
        self.calculator = TechnicalIndicatorCalculator()
        self.volatility_threshold = volatility_threshold

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        if len(data) < 50:
            return None
        
        df = self.calculator.calculate_all(data)
        latest = df.iloc[-1]
        
        # ========== НОВЫЙ ФИЛЬТР ПО ВОЛАТИЛЬНОСТИ ==========
        atr_series = calculate_atr(data, period=14)
        if atr_series is None or atr_series.empty:
            logger.debug(f"ATR calculation failed for {symbol}")
            return None
        
        latest_atr = atr_series.iloc[-1]
        current_price = latest['close']
        
        if latest_atr < current_price * self.volatility_threshold:
            logger.debug(f"{symbol} volatility too low (ATR={latest_atr:.4f} < {current_price*self.volatility_threshold:.4f}), skipping")
            return None
        
        vol_ratio = latest_atr / (current_price * self.volatility_threshold)
        vol_factor = min(vol_ratio, 2.0)
        # ===================================================
        
        signal_score = 0
        reasons = []
        
        # RSI signals
        if latest['rsi'] < 30:
            signal_score += 2
            reasons.append(f"RSI oversold: {latest['rsi']:.1f}")
        elif latest['rsi'] > 70:
            signal_score -= 2
            reasons.append(f"RSI overbought: {latest['rsi']:.1f}")
        
        # MACD signals
        if latest['macd'] > latest['macd_signal']:
            signal_score += 1
            reasons.append("MACD bullish")
        elif latest['macd'] < latest['macd_signal']:
            signal_score -= 1
            reasons.append("MACD bearish")
        
        # Bollinger Bands
        if latest['bb_position'] < 0.2:
            signal_score += 1
            reasons.append("Near lower Bollinger Band")
        elif latest['bb_position'] > 0.8:
            signal_score -= 1
            reasons.append("Near upper Bollinger Band")
        
        if abs(signal_score) < 1:
            return None
        
        final_signal = 1 if signal_score > 0 else -1
        base_confidence = min(1.0, abs(signal_score) / 5)
        confidence = min(base_confidence * vol_factor, 1.0)
        
        return Signal(
            symbol=symbol,
            signal=final_signal,
            confidence=confidence,
            strength=abs(signal_score),
            reasons=reasons[:3],
            price=latest['close'],
            timestamp=df.index[-1],
            strategy_name=self.name
        )

# ======================= ДЕТЕКТОР КАНАЛОВ =======================

@dataclass
class ChannelSignal:
    channel_type: str  # 'ascending', 'descending', 'horizontal'
    upper_line: List[float]
    lower_line: List[float]
    slope: float
    confidence: float
    price_position: float

class ChannelDetector:
    """
    Детектор ценовых каналов (нисходящих, восходящих, горизонтальных)
    """
    def __init__(self, lookback_period: int = 50, channel_width: float = 2.0):
        self.lookback_period = lookback_period
        self.channel_width = channel_width

    def detect_channel(self, df: pd.DataFrame) -> Optional[ChannelSignal]:
        if len(df) < self.lookback_period:
            return None

        recent = df.tail(self.lookback_period).copy()
        prices = recent['close'].values
        x = np.arange(len(prices))

        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        center = np.polyval([slope, intercept], x)

        deviations = prices - center
        std = np.std(deviations)
        upper = center + std * self.channel_width
        lower = center - std * self.channel_width

        inside = np.sum((prices >= lower) & (prices <= upper)) / len(prices)
        if inside < 0.9:
            return None

        if slope > 0.001 * np.mean(prices):
            channel_type = "ascending"
            confidence = min(1.0, slope * 100)
        elif slope < -0.001 * np.mean(prices):
            channel_type = "descending"
            confidence = min(1.0, abs(slope) * 100)
        else:
            channel_type = "horizontal"
            confidence = 0.5

        last_price = prices[-1]
        pos = (last_price - lower[-1]) / (upper[-1] - lower[-1])
        pos = np.clip(pos, 0, 1)

        return ChannelSignal(channel_type, upper.tolist(), lower.tolist(), slope, confidence, pos)

    def get_trading_signal(self, df: pd.DataFrame) -> Dict:
        channel = self.detect_channel(df)
        if not channel:
            return {'signal': 0, 'confidence': 0, 'action': 'none'}

        if channel.channel_type == "descending":
            if channel.price_position > 0.8:
                return {'signal': -1, 'confidence': channel.confidence * 0.9, 'action': 'short_at_resistance'}
            elif channel.price_position < 0.2:
                return {'signal': 1, 'confidence': channel.confidence * 0.8, 'action': 'cover_at_support'}
        elif channel.channel_type == "ascending":
            if channel.price_position < 0.2:
                return {'signal': 1, 'confidence': channel.confidence * 0.9, 'action': 'buy_at_support'}
            elif channel.price_position > 0.8:
                return {'signal': -1, 'confidence': channel.confidence * 0.8, 'action': 'sell_at_resistance'}

        return {'signal': 0, 'confidence': 0, 'action': 'hold'}