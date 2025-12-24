"""Base class for technical indicators."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Indicator(ABC):
    """Abstract base class for all technical indicators."""

    def __init__(self, name: str, period: int = 14):
        self.name = name
        self.period = period

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the indicator values.

        Args:
            data: DataFrame with OHLCV columns (open, high, low, close, volume)

        Returns:
            Series with indicator values
        """
        pass

    def validate_data(self, data: pd.DataFrame, required_cols: list[str]) -> None:
        """Validate input data has required columns."""
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if len(data) < self.period:
            raise ValueError(f"Insufficient data: need {self.period} rows, got {len(data)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(period={self.period})"


class TrendIndicator(Indicator):
    """Base class for trend-following indicators."""

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get trend signal: 1 for bullish, -1 for bearish, 0 for neutral."""
        values = self.calculate(data)
        return pd.Series(np.where(values > 0, 1, np.where(values < 0, -1, 0)), index=data.index)


class MomentumIndicator(Indicator):
    """Base class for momentum indicators."""

    def __init__(self, name: str, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__(name, period)
        self.overbought = overbought
        self.oversold = oversold

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get momentum signal based on overbought/oversold levels."""
        values = self.calculate(data)
        signals = pd.Series(0, index=data.index)
        signals[values > self.overbought] = -1  # Overbought = potential sell
        signals[values < self.oversold] = 1  # Oversold = potential buy
        return signals


class VolatilityIndicator(Indicator):
    """Base class for volatility indicators."""

    def is_squeeze(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Detect low volatility squeeze conditions."""
        values = self.calculate(data)
        rolling_mean = values.rolling(window=50).mean()
        return values < (rolling_mean * threshold)


class VolumeIndicator(Indicator):
    """Base class for volume indicators."""

    def validate_data(self, data: pd.DataFrame, required_cols: list[str] = None) -> None:
        """Validate data includes volume column."""
        required = required_cols or ["close", "volume"]
        super().validate_data(data, required)
