"""Signal filters for additional validation."""

from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Optional

import pandas as pd

from ..strategies.base import Signal, Side
from ..indicators import ATR, RSI


class SignalFilter(ABC):
    """Abstract base class for signal filters."""

    @abstractmethod
    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Filter a signal.

        Args:
            signal: Signal to validate
            data: Market data

        Returns:
            True if signal passes filter
        """
        pass


class ConfluenceFilter(SignalFilter):
    """Filter signals based on multi-indicator confluence."""

    def __init__(self, min_indicators: int = 3):
        """Initialize confluence filter.

        Args:
            min_indicators: Minimum indicators that must agree
        """
        self.min_indicators = min_indicators

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Check if signal has sufficient confluence."""
        indicators_agreeing = 0

        # Check trend (EMA)
        ema_20 = data["close"].ewm(span=20, adjust=False).mean()
        ema_50 = data["close"].ewm(span=50, adjust=False).mean()

        if signal.side == Side.LONG:
            if ema_20.iloc[-1] > ema_50.iloc[-1]:
                indicators_agreeing += 1
        else:
            if ema_20.iloc[-1] < ema_50.iloc[-1]:
                indicators_agreeing += 1

        # Check RSI
        rsi = RSI(14).calculate(data)
        if signal.side == Side.LONG:
            if 40 < rsi.iloc[-1] < 70:
                indicators_agreeing += 1
        else:
            if 30 < rsi.iloc[-1] < 60:
                indicators_agreeing += 1

        # Check price vs VWAP (if volume available)
        if "volume" in data.columns:
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()

            if signal.side == Side.LONG:
                if data["close"].iloc[-1] > vwap.iloc[-1]:
                    indicators_agreeing += 1
            else:
                if data["close"].iloc[-1] < vwap.iloc[-1]:
                    indicators_agreeing += 1

        # Check momentum
        momentum = data["close"].pct_change(10).iloc[-1]
        if signal.side == Side.LONG and momentum > 0:
            indicators_agreeing += 1
        elif signal.side == Side.SHORT and momentum < 0:
            indicators_agreeing += 1

        return indicators_agreeing >= self.min_indicators


class TimeFilter(SignalFilter):
    """Filter signals based on time of day."""

    def __init__(self, allowed_hours: list[tuple[int, int]] = None,
                 blocked_hours: list[tuple[int, int]] = None):
        """Initialize time filter.

        Args:
            allowed_hours: List of (start_hour, end_hour) tuples for allowed trading
            blocked_hours: List of (start_hour, end_hour) tuples for blocked trading
        """
        self.allowed_hours = allowed_hours or [(9, 17)]  # Default: 9 AM - 5 PM
        self.blocked_hours = blocked_hours or []

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Check if current time is within allowed trading hours."""
        current_hour = datetime.now().hour

        # Check blocked hours first
        for start, end in self.blocked_hours:
            if start <= current_hour < end:
                return False

        # Check allowed hours
        for start, end in self.allowed_hours:
            if start <= current_hour < end:
                return True

        return False


class VolatilityFilter(SignalFilter):
    """Filter signals based on market volatility."""

    def __init__(self, min_atr_pct: float = 0.5, max_atr_pct: float = 5.0,
                 atr_period: int = 14):
        """Initialize volatility filter.

        Args:
            min_atr_pct: Minimum ATR as % of price
            max_atr_pct: Maximum ATR as % of price
            atr_period: ATR calculation period
        """
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.atr = ATR(atr_period)

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Check if volatility is within acceptable range."""
        atr_value = self.atr.calculate(data).iloc[-1]
        price = data["close"].iloc[-1]

        atr_pct = (atr_value / price) * 100

        return self.min_atr_pct <= atr_pct <= self.max_atr_pct


class RiskRewardFilter(SignalFilter):
    """Filter signals based on risk/reward ratio."""

    def __init__(self, min_ratio: float = 2.0):
        """Initialize R:R filter.

        Args:
            min_ratio: Minimum acceptable risk/reward ratio
        """
        self.min_ratio = min_ratio

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Check if signal meets minimum R:R ratio."""
        if not signal.stop_loss or not signal.take_profit:
            return True  # Can't evaluate without levels

        rr = signal.risk_reward_ratio
        return rr is not None and rr >= self.min_ratio


class VolumeFilter(SignalFilter):
    """Filter signals based on volume conditions."""

    def __init__(self, min_volume_mult: float = 1.0, lookback: int = 20):
        """Initialize volume filter.

        Args:
            min_volume_mult: Minimum volume as multiple of average
            lookback: Period for average volume calculation
        """
        self.min_volume_mult = min_volume_mult
        self.lookback = lookback

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Check if current volume is sufficient."""
        if "volume" not in data.columns:
            return True

        avg_volume = data["volume"].rolling(window=self.lookback).mean().iloc[-1]
        current_volume = data["volume"].iloc[-1]

        return current_volume >= avg_volume * self.min_volume_mult


class CompositeFilter(SignalFilter):
    """Combines multiple filters with AND logic."""

    def __init__(self, filters: list[SignalFilter] = None):
        self.filters = filters or []

    def add_filter(self, filter: SignalFilter) -> None:
        """Add a filter to the composite."""
        self.filters.append(filter)

    def filter(self, signal: Signal, data: pd.DataFrame) -> bool:
        """Signal must pass all filters."""
        return all(f.filter(signal, data) for f in self.filters)
