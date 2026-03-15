import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class RetestSignal:
    direction: str
    breakout_price: float
    retest_price: float
    current_price: float
    confidence: float
    stop_loss: float
    take_profit: float
    reason: str

class RetestDetector:
    def __init__(self, lookback: int = 50, retest_tolerance: float = 0.02,
                 min_breakout_move: float = 0.003, volume_factor: float = 1.2):
        self.lookback = lookback
        self.retest_tolerance = retest_tolerance
        self.min_breakout_move = min_breakout_move
        self.volume_factor = volume_factor

    def _find_swing_highs(self, highs: np.ndarray) -> np.ndarray:
        peaks = []
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append(highs[i])
        return np.array(peaks)

    def _find_swing_lows(self, lows: np.ndarray) -> np.ndarray:
        troughs = []
        for i in range(1, len(lows)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append(lows[i])
        return np.array(troughs)

    def detect_retest(self, df: pd.DataFrame) -> Optional[RetestSignal]:
        if len(df) < self.lookback + 10:
            return None

        recent = df.tail(self.lookback).copy()
        highs = recent['high'].values
        lows = recent['low'].values
        closes = recent['close'].values
        volumes = recent['volume'].values

        swing_highs = self._find_swing_highs(highs)
        swing_lows = self._find_swing_lows(lows)

        current_price = df['close'].iloc[-1]

        # ----- Bullish breakout -----
        relevant_highs = swing_highs[swing_highs < current_price * (1 - self.min_breakout_move)]
        if len(relevant_highs) > 0:
            breakout_level = relevant_highs.max()
            for i in range(-5, 0):
                if abs(closes[i] - breakout_level) / breakout_level < self.retest_tolerance:
                    # Нашли свечу ретеста – теперь проверим объём на пробойной свече
                    break_idx = np.where(highs > breakout_level)[0]
                    if len(break_idx) > 0:
                        first_break = break_idx[0]
                        breakout_vol = volumes[first_break]
                        avg_vol = volumes[max(0, first_break-20):first_break].mean()
                        if breakout_vol > avg_vol * self.volume_factor:
                            confidence = min(0.9, 0.5 + (breakout_vol / avg_vol) * 0.2)
                            logger.debug(f"✅ Bullish retest at {closes[i]:.4f}, breakout level {breakout_level:.4f}")
                            return RetestSignal(
                                direction='long',
                                breakout_price=breakout_level,
                                retest_price=closes[i],
                                current_price=current_price,
                                confidence=confidence,
                                stop_loss=breakout_level * 0.995,
                                take_profit=current_price * 1.01,
                                reason=f"Bullish breakout above {breakout_level:.4f}, retested at {closes[i]:.4f}"
                            )
                    break

        # ----- Bearish breakdown -----
        relevant_lows = swing_lows[swing_lows > current_price * (1 + self.min_breakout_move)]
        if len(relevant_lows) > 0:
            breakout_level = relevant_lows.min()
            for i in range(-5, 0):
                if abs(closes[i] - breakout_level) / breakout_level < self.retest_tolerance:
                    break_idx = np.where(lows < breakout_level)[0]
                    if len(break_idx) > 0:
                        first_break = break_idx[0]
                        breakout_vol = volumes[first_break]
                        avg_vol = volumes[max(0, first_break-20):first_break].mean()
                        if breakout_vol > avg_vol * self.volume_factor:
                            confidence = min(0.9, 0.5 + (breakout_vol / avg_vol) * 0.2)
                            logger.debug(f"✅ Bearish retest at {closes[i]:.4f}, breakout level {breakout_level:.4f}")
                            return RetestSignal(
                                direction='short',
                                breakout_price=breakout_level,
                                retest_price=closes[i],
                                current_price=current_price,
                                confidence=confidence,
                                stop_loss=breakout_level * 1.005,
                                take_profit=current_price * 0.99,
                                reason=f"Bearish breakdown below {breakout_level:.4f}, retested at {closes[i]:.4f}"
                            )
                    break

        logger.debug("No retest pattern found")
        return None