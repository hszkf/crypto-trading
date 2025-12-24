"""Volatility indicators: Bollinger Bands, ATR, Keltner Channels."""

import pandas as pd

from .base import VolatilityIndicator


class BollingerBands(VolatilityIndicator):
    """Bollinger Bands - volatility bands around moving average."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__("BollingerBands", period)
        self.std_dev = std_dev

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate middle band (SMA)."""
        self.validate_data(data, ["close"])
        return data["close"].rolling(window=self.period).mean()

    def calculate_all(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate upper, middle, and lower bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        self.validate_data(data, ["close"])

        middle = data["close"].rolling(window=self.period).mean()
        std = data["close"].rolling(window=self.period).std()

        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)

        return upper, middle, lower

    def bandwidth(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bandwidth (volatility measure)."""
        upper, middle, lower = self.calculate_all(data)
        return (upper - lower) / middle * 100

    def percent_b(self, data: pd.DataFrame) -> pd.Series:
        """Calculate %B - where price is relative to bands."""
        upper, middle, lower = self.calculate_all(data)
        return (data["close"] - lower) / (upper - lower)

    def is_squeeze(self, data: pd.DataFrame, lookback: int = 120) -> pd.Series:
        """Detect Bollinger Band squeeze (low volatility)."""
        bw = self.bandwidth(data)
        min_bw = bw.rolling(window=lookback).min()
        return bw <= min_bw * 1.05

    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """Get signals based on band touches/crosses."""
        upper, middle, lower = self.calculate_all(data)
        close = data["close"]
        signals = pd.Series(0, index=data.index)

        # Price crosses below lower band = potential buy
        signals[close < lower] = 1
        # Price crosses above upper band = potential sell
        signals[close > upper] = -1

        return signals


class ATR(VolatilityIndicator):
    """Average True Range - volatility indicator."""

    def __init__(self, period: int = 14):
        super().__init__("ATR", period)

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        self.validate_data(data, ["high", "low", "close"])

        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.period, adjust=False).mean()

        return atr

    def stop_loss_price(
        self, data: pd.DataFrame, multiplier: float = 2.0, direction: str = "long"
    ) -> pd.Series:
        """Calculate ATR-based stop loss price.

        Args:
            data: OHLCV DataFrame
            multiplier: ATR multiplier for stop distance
            direction: "long" or "short"
        """
        atr = self.calculate(data)
        close = data["close"]

        if direction == "long":
            return close - (atr * multiplier)
        else:
            return close + (atr * multiplier)

    def position_size(
        self,
        account_value: float,
        risk_percent: float,
        entry_price: float,
        atr_value: float,
        atr_multiplier: float = 2.0,
    ) -> float:
        """Calculate position size based on ATR stop.

        Args:
            account_value: Total account value
            risk_percent: Percentage of account to risk (e.g., 0.02 for 2%)
            entry_price: Entry price for the trade
            atr_value: Current ATR value
            atr_multiplier: Multiplier for stop distance

        Returns:
            Position size in units
        """
        risk_amount = account_value * risk_percent
        stop_distance = atr_value * atr_multiplier
        position_size = risk_amount / stop_distance
        return position_size


class KeltnerChannels(VolatilityIndicator):
    """Keltner Channels - ATR-based volatility bands."""

    def __init__(self, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0):
        super().__init__("KeltnerChannels", ema_period)
        self.atr_period = atr_period
        self.multiplier = multiplier

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate middle line (EMA)."""
        self.validate_data(data, ["close"])
        return data["close"].ewm(span=self.period, adjust=False).mean()

    def calculate_all(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate upper, middle, and lower channels.

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel)
        """
        self.validate_data(data, ["high", "low", "close"])

        middle = data["close"].ewm(span=self.period, adjust=False).mean()
        atr = ATR(self.atr_period).calculate(data)

        upper = middle + (atr * self.multiplier)
        lower = middle - (atr * self.multiplier)

        return upper, middle, lower

    def squeeze_with_bb(
        self, data: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0
    ) -> pd.Series:
        """Detect squeeze: BB inside Keltner Channels."""
        kc_upper, _, kc_lower = self.calculate_all(data)
        bb = BollingerBands(bb_period, bb_std)
        bb_upper, _, bb_lower = bb.calculate_all(data)

        # Squeeze when BB is inside KC
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        return squeeze
