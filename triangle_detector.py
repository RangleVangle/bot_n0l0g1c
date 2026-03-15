import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class TrianglePattern:
    """Структура обнаруженного треугольника"""
    type: str  # 'ascending', 'descending', 'symmetrical'
    start_idx: int
    end_idx: int
    upper_line: np.ndarray
    lower_line: np.ndarray
    resistance: float
    support: float
    height: float
    slope_upper: float
    slope_lower: float
    touches_upper: int
    touches_lower: int
    volume_ratio: float
    confidence: float

class TriangleDetector:
    """
    Обнаружение треугольных паттернов с более мягкими параметрами.
    """
    def __init__(self, lookback: int = 100, min_touches: int = 2,
                 volume_window: int = 20, max_deviation: float = 0.03):
        self.lookback = lookback
        self.min_touches = min_touches
        self.volume_window = volume_window
        self.max_deviation = max_deviation

    def _find_peaks(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        peaks_idx = []
        peaks_val = []
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks_idx.append(i)
                peaks_val.append(prices[i])
        return {'x': np.array(peaks_idx), 'y': np.array(peaks_val)}

    def _find_troughs(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        troughs_idx = []
        troughs_val = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs_idx.append(i)
                troughs_val.append(prices[i])
        return {'x': np.array(troughs_idx), 'y': np.array(troughs_val)}

    def find_triangle(self, df: pd.DataFrame) -> Optional[TrianglePattern]:
        if len(df) < self.lookback:
            return None

        data = df.iloc[-self.lookback:].copy()
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values
        x = np.arange(len(data))

        peaks = self._find_peaks(highs)
        troughs = self._find_troughs(lows)

        if len(peaks['x']) < self.min_touches or len(troughs['x']) < self.min_touches:
            logger.debug(f"Triangle: not enough touches (peaks {len(peaks['x'])}, troughs {len(troughs['x'])})")
            return None

        # Линии регрессии по пикам и впадинам
        coeffs_upper = np.polyfit(peaks['x'], peaks['y'], 1)
        coeffs_lower = np.polyfit(troughs['x'], troughs['y'], 1)
        upper_line = np.polyval(coeffs_upper, x)
        lower_line = np.polyval(coeffs_lower, x)

        # Процент свечей внутри канала
        inside = ((highs <= upper_line * (1 + self.max_deviation)) &
                  (lows >= lower_line * (1 - self.max_deviation)))
        inside_pct = inside.mean()
        if inside_pct < 0.6:  # снизили порог
            logger.debug(f"Triangle: only {inside_pct:.2%} candles inside")
            return None

        slope_upper = coeffs_upper[0]
        slope_lower = coeffs_lower[0]

        eps = 1e-5
        if abs(slope_upper) < eps and slope_lower > eps:
            triangle_type = "ascending"
            resistance = upper_line[0]
            support = None
        elif slope_upper < -eps and abs(slope_lower) < eps:
            triangle_type = "descending"
            support = lower_line[0]
            resistance = None
        elif slope_upper < -eps and slope_lower > eps:
            triangle_type = "symmetrical"
            resistance = upper_line[-1]
            support = lower_line[-1]
        else:
            logger.debug(f"Triangle: slopes not converging ({slope_upper:.4f}, {slope_lower:.4f})")
            return None

        avg_price = (upper_line + lower_line).mean() / 2
        height_pct = ((upper_line - lower_line) / avg_price).mean() * 100

        volume_inside = volumes[inside].mean()
        volume_outside = volumes[~inside].mean() if (~inside).any() else volume_inside
        vol_ratio = volume_inside / (volume_outside + 1e-8)

        touches_upper = np.sum(np.abs(highs - upper_line) / highs < self.max_deviation)
        touches_lower = np.sum(np.abs(lows - lower_line) / lows < self.max_deviation)

        confidence = (min(touches_upper / self.min_touches, 1.0) * 0.4 +
                      min(touches_lower / self.min_touches, 1.0) * 0.4 +
                      min(1.0, 1.5 / (vol_ratio + 1e-8)) * 0.2)
        confidence = min(confidence, 0.95)

        pattern = TrianglePattern(
            type=triangle_type,
            start_idx=len(df) - self.lookback,
            end_idx=len(df) - 1,
            upper_line=upper_line,
            lower_line=lower_line,
            resistance=resistance if resistance else upper_line[-1],
            support=support if support else lower_line[-1],
            height=height_pct,
            slope_upper=slope_upper,
            slope_lower=slope_lower,
            touches_upper=int(touches_upper),
            touches_lower=int(touches_lower),
            volume_ratio=vol_ratio,
            confidence=confidence
        )

        logger.debug(f"✅ Triangle found: {triangle_type}, confidence {confidence:.2f}")
        return pattern

    def get_breakout_signal(self, df: pd.DataFrame, pattern: TrianglePattern) -> Dict:
        last_idx = -1
        current_price = df['close'].iloc[last_idx]
        volume = df['volume'].iloc[last_idx]
        avg_volume = df['volume'].iloc[-self.volume_window:].mean()
        vol_ratio = volume / avg_volume

        upper_last = pattern.upper_line[-1]
        lower_last = pattern.lower_line[-1]

        if current_price > upper_last * 1.003:  # 0.3% пробой
            conf = pattern.confidence * min(1.0, vol_ratio / 1.2)
            return {
                'signal': 1,
                'confidence': conf,
                'target': current_price + (upper_last - lower_last),
                'stop': lower_last,
                'reasoning': f"Upside breakout, volume {vol_ratio:.2f}"
            }
        if current_price < lower_last * 0.997:
            conf = pattern.confidence * min(1.0, vol_ratio / 1.2)
            return {
                'signal': -1,
                'confidence': conf,
                'target': current_price - (upper_last - lower_last),
                'stop': upper_last,
                'reasoning': f"Downside breakout, volume {vol_ratio:.2f}"
            }
        return {'signal': 0, 'confidence': 0, 'target': None, 'stop': None}