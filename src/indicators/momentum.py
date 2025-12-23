"""Momentum indicators: RSI, Stochastic RSI, CCI, Williams %R."""

import numpy as np
import pandas as pd

from .base import MomentumIndicator


class RSI(MomentumIndicator):
    """Relative Strength Index."""

    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        super().__init__("RSI", period, overbought, oversold)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["close"])

        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_divergence(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Detect bullish/bearish divergence.

        Returns:
            Series: 1 for bullish divergence, -1 for bearish, 0 for none
        """
        rsi = self.calculate(data)
        close = data["close"]
        signals = pd.Series(0, index=data.index)

        for i in range(lookback, len(data)):
            window_close = close.iloc[i-lookback:i+1]
            window_rsi = rsi.iloc[i-lookback:i+1]

            # Bullish: price lower low, RSI higher low
            if window_close.iloc[-1] < window_close.min() * 1.01:
                if window_rsi.iloc[-1] > window_rsi.min() * 1.05:
                    signals.iloc[i] = 1

            # Bearish: price higher high, RSI lower high
            if window_close.iloc[-1] > window_close.max() * 0.99:
                if window_rsi.iloc[-1] < window_rsi.max() * 0.95:
                    signals.iloc[i] = -1

        return signals


class StochasticRSI(MomentumIndicator):
    """Stochastic RSI - RSI applied to RSI values."""

    def __init__(self, rsi_period: int = 14, stoch_period: int = 14,
                 k_period: int = 3, d_period: int = 3):
        super().__init__("StochRSI", rsi_period, 80, 20)
        self.stoch_period = stoch_period
        self.k_period = k_period
        self.d_period = d_period

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic RSI %K."""
        rsi = RSI(self.period).calculate(data)

        lowest_rsi = rsi.rolling(window=self.stoch_period).min()
        highest_rsi = rsi.rolling(window=self.stoch_period).max()

        stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        k = stoch_rsi.rolling(window=self.k_period).mean()

        return k

    def calculate_all(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calculate both %K and %D lines.

        Returns:
            Tuple of (k_line, d_line)
        """
        k = self.calculate(data)
        d = k.rolling(window=self.d_period).mean()
        return k, d

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get crossover signals."""
        k, d = self.calculate_all(data)
        signals = pd.Series(0, index=data.index)

        # Bullish crossover in oversold
        bullish = (k > d) & (k.shift(1) <= d.shift(1)) & (k < self.oversold)
        signals[bullish] = 1

        # Bearish crossover in overbought
        bearish = (k < d) & (k.shift(1) >= d.shift(1)) & (k > self.overbought)
        signals[bearish] = -1

        return signals


class CCI(MomentumIndicator):
    """Commodity Channel Index."""

    def __init__(self, period: int = 20, overbought: float = 100, oversold: float = -100):
        super().__init__("CCI", period, overbought, oversold)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["high", "low", "close"])

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        sma = typical_price.rolling(window=self.period).mean()
        mad = typical_price.rolling(window=self.period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )

        cci = (typical_price - sma) / (0.015 * mad)
        return cci


class WilliamsR(MomentumIndicator):
    """Williams %R - momentum indicator."""

    def __init__(self, period: int = 14):
        super().__init__("WilliamsR", period, -20, -80)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["high", "low", "close"])

        highest_high = data["high"].rolling(window=self.period).max()
        lowest_low = data["low"].rolling(window=self.period).min()

        williams_r = -100 * (highest_high - data["close"]) / (highest_high - lowest_low)
        return williams_r

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get signal based on Williams %R levels."""
        values = self.calculate(data)
        signals = pd.Series(0, index=data.index)
        signals[values > self.overbought] = -1  # Above -20 = overbought
        signals[values < self.oversold] = 1     # Below -80 = oversold
        return signals
