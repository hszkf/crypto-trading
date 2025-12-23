"""Volume indicators: OBV, VWAP, MFI."""

import numpy as np
import pandas as pd

from .base import VolumeIndicator


class OBV(VolumeIndicator):
    """On-Balance Volume - cumulative volume indicator."""

    def __init__(self):
        super().__init__("OBV", period=1)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["close", "volume"])

        close = data["close"]
        volume = data["volume"]

        direction = np.where(close > close.shift(1), 1,
                            np.where(close < close.shift(1), -1, 0))

        obv = (volume * direction).cumsum()
        return pd.Series(obv, index=data.index)

    def get_signal(self, data: pd.DataFrame, ma_period: int = 20) -> pd.Series:
        """Get OBV trend signal based on moving average."""
        obv = self.calculate(data)
        obv_ma = obv.rolling(window=ma_period).mean()

        signals = pd.Series(0, index=data.index)
        signals[obv > obv_ma] = 1   # Bullish accumulation
        signals[obv < obv_ma] = -1  # Bearish distribution

        return signals

    def detect_divergence(self, data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """Detect price/OBV divergence."""
        obv = self.calculate(data)
        close = data["close"]
        signals = pd.Series(0, index=data.index)

        for i in range(lookback, len(data)):
            price_window = close.iloc[i-lookback:i+1]
            obv_window = obv.iloc[i-lookback:i+1]

            # Bullish: price lower, OBV higher
            if price_window.iloc[-1] < price_window.iloc[0]:
                if obv_window.iloc[-1] > obv_window.iloc[0]:
                    signals.iloc[i] = 1

            # Bearish: price higher, OBV lower
            if price_window.iloc[-1] > price_window.iloc[0]:
                if obv_window.iloc[-1] < obv_window.iloc[0]:
                    signals.iloc[i] = -1

        return signals


class VWAP(VolumeIndicator):
    """Volume Weighted Average Price."""

    def __init__(self):
        super().__init__("VWAP", period=1)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate cumulative VWAP (typically reset daily)."""
        self.validate_data(data, ["high", "low", "close", "volume"])

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cumulative_tp_vol = (typical_price * data["volume"]).cumsum()
        cumulative_vol = data["volume"].cumsum()

        vwap = cumulative_tp_vol / cumulative_vol
        return vwap

    def calculate_with_bands(self, data: pd.DataFrame,
                             std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate VWAP with standard deviation bands.

        Returns:
            Tuple of (vwap, upper_band, lower_band)
        """
        self.validate_data(data, ["high", "low", "close", "volume"])

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        vwap = self.calculate(data)

        # Rolling variance for bands
        cumulative_vol = data["volume"].cumsum()
        sq_diff = ((typical_price - vwap) ** 2 * data["volume"]).cumsum()
        variance = sq_diff / cumulative_vol
        std = np.sqrt(variance)

        upper = vwap + (std * std_mult)
        lower = vwap - (std * std_mult)

        return vwap, upper, lower

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get signal based on price position relative to VWAP."""
        vwap = self.calculate(data)
        close = data["close"]

        signals = pd.Series(0, index=data.index)
        signals[close > vwap] = 1   # Above VWAP = bullish
        signals[close < vwap] = -1  # Below VWAP = bearish

        return signals


class MFI(VolumeIndicator):
    """Money Flow Index - volume-weighted RSI."""

    def __init__(self, period: int = 14, overbought: float = 80, oversold: float = 20):
        super().__init__("MFI", period)
        self.overbought = overbought
        self.oversold = oversold

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["high", "low", "close", "volume"])

        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        raw_money_flow = typical_price * data["volume"]

        # Positive/negative money flow
        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)

        positive_sum = positive_flow.rolling(window=self.period).sum()
        negative_sum = negative_flow.rolling(window=self.period).sum()

        money_ratio = positive_sum / negative_sum
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get signal based on overbought/oversold levels."""
        mfi = self.calculate(data)
        signals = pd.Series(0, index=data.index)

        signals[mfi > self.overbought] = -1  # Overbought
        signals[mfi < self.oversold] = 1     # Oversold

        return signals
