import numpy as np
import pandas as pd
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Detects market regimes without external ML libraries.
    Uses rolling volatility and simple trend strength.
    """
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.regime_names = {
            0: "Ranging / Low Volatility",
            1: "Trending Bull / High Volatility",
            2: "Trending Bear / High Volatility"
        }
        self.is_fitted = False

    def fit(self, data: pd.DataFrame):
        """No training needed – just store historical volatility for reference."""
        self.historical_volatility = data['close'].pct_change().std() * np.sqrt(252)
        self.is_fitted = True

    def predict_regime(self, data: pd.DataFrame) -> int:
        """Predict regime based on recent volatility and trend."""
        if len(data) < self.lookback:
            return 0

        close = data['close'].values
        returns = np.diff(close) / close[:-1]
        recent_returns = returns[-self.lookback:]

        # Volatility (annualized)
        vol = np.std(recent_returns) * np.sqrt(252)

        # Trend strength: linear regression slope of price over lookback window
        price_series = close[-self.lookback:]
        x = np.arange(len(price_series))
        slope = np.polyfit(x, price_series, 1)[0]
        trend_strength = slope / (price_series.mean() + 1e-8) * 100  # percent per bar

        # Thresholds (you can tune these)
        if abs(trend_strength) < 0.5:
            return 0  # ranging
        elif trend_strength > 0.5:
            return 1  # bullish
        else:
            return 2  # bearish

    def get_regime_name(self, regime: int) -> str:
        return self.regime_names.get(regime, f"Regime {regime}")

    def get_regime_characteristics(self, data: pd.DataFrame) -> Dict:
        regime = self.predict_regime(data)
        returns = data['close'].pct_change().dropna()
        recent_returns = returns.tail(20)
        trend = recent_returns.mean()
        volatility = recent_returns.std() * np.sqrt(252)

        return {
            'regime_id': regime,
            'regime_name': self.get_regime_name(regime),
            'volatility': volatility,
            'trend_strength': abs(trend) / (volatility + 1e-8) if volatility > 0 else 0,
            'recent_return': trend,
            'is_trending': abs(trend) > volatility * 0.5,
            'is_volatile': volatility > self.historical_volatility if hasattr(self, 'historical_volatility') else False
        }