"""Trend indicators: SMA, EMA, MACD, ADX."""

import numpy as np
import pandas as pd

from .base import TrendIndicator


class SMA(TrendIndicator):
    """Simple Moving Average."""

    def __init__(self, period: int = 20):
        super().__init__("SMA", period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["close"])
        return data["close"].rolling(window=self.period).mean()


class EMA(TrendIndicator):
    """Exponential Moving Average."""

    def __init__(self, period: int = 20):
        super().__init__("EMA", period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["close"])
        return data["close"].ewm(span=self.period, adjust=False).mean()


class MACD(TrendIndicator):
    """Moving Average Convergence Divergence."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("MACD", slow)
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate MACD line (fast EMA - slow EMA)."""
        self.validate_data(data, ["close"])
        fast_ema = data["close"].ewm(span=self.fast, adjust=False).mean()
        slow_ema = data["close"].ewm(span=self.slow, adjust=False).mean()
        return fast_ema - slow_ema

    def calculate_all(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD line, signal line, and histogram.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        macd_line = self.calculate(data)
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get MACD crossover signal."""
        macd_line, signal_line, _ = self.calculate_all(data)
        signals = pd.Series(0, index=data.index)

        # Bullish crossover
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        # Bearish crossover
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1

        return signals


class ADX(TrendIndicator):
    """Average Directional Index - measures trend strength."""

    def __init__(self, period: int = 14):
        super().__init__("ADX", period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX value."""
        self.validate_data(data, ["high", "low", "close"])

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm, index=data.index).ewm(span=self.period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=data.index).ewm(span=self.period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        return adx

    def calculate_all(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, and -DI.

        Returns:
            Tuple of (adx, plus_di, minus_di)
        """
        self.validate_data(data, ["high", "low", "close"])

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm, index=data.index).ewm(span=self.period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=data.index).ewm(span=self.period, adjust=False).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        return adx, plus_di, minus_di

    def is_trending(self, data: pd.DataFrame, threshold: float = 25) -> pd.Series:
        """Check if market is trending (ADX > threshold)."""
        adx = self.calculate(data)
        return adx > threshold
