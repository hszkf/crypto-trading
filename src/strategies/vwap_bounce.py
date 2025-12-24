"""VWAP Bounce Strategy for Intraday Trading."""

from datetime import datetime

import pandas as pd

from ..indicators import ATR, RSI, VWAP
from .base import Side, Signal, SignalType, Strategy, StrategyResult


class VWAPBounceStrategy(Strategy):
    """VWAP bounce strategy for intraday trading.

    Trades pullbacks to VWAP with confirmation.

    Entry Long:
        - Price pulls back to VWAP from above
        - Bullish candle pattern at VWAP
        - RSI not oversold (bounce potential)

    Entry Short:
        - Price rallies to VWAP from below
        - Bearish candle pattern at VWAP
        - RSI not overbought (rejection potential)
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "15m",
        vwap_tolerance: float = 0.002,  # 0.2% tolerance
        rsi_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        super().__init__("VWAP_Bounce", symbol, timeframe)
        self.vwap = VWAP()
        self.rsi = RSI(rsi_period)
        self.atr = ATR(14)
        self.vwap_tolerance = vwap_tolerance
        self.atr_multiplier = atr_multiplier

    def _is_bullish_candle(self, data: pd.DataFrame, idx: int) -> bool:
        """Check for bullish candle pattern."""
        o, h, low, c = (
            data["open"].iloc[idx],
            data["high"].iloc[idx],
            data["low"].iloc[idx],
            data["close"].iloc[idx],
        )

        body = abs(c - o)
        lower_wick = min(o, c) - low
        upper_wick = h - max(o, c)

        # Bullish engulfing or hammer-like
        if c > o:
            if body > 0:
                # Hammer: lower wick > 2x body
                if lower_wick > body * 2 and upper_wick < body * 0.5:
                    return True
                # Strong bullish: body > 60% of range
                if body / (h - low) > 0.6:
                    return True
        return False

    def _is_bearish_candle(self, data: pd.DataFrame, idx: int) -> bool:
        """Check for bearish candle pattern."""
        o, h, low, c = (
            data["open"].iloc[idx],
            data["high"].iloc[idx],
            data["low"].iloc[idx],
            data["close"].iloc[idx],
        )

        body = abs(c - o)
        lower_wick = min(o, c) - low
        upper_wick = h - max(o, c)

        # Bearish engulfing or shooting star-like
        if c < o:
            if body > 0:
                # Shooting star: upper wick > 2x body
                if upper_wick > body * 2 and lower_wick < body * 0.5:
                    return True
                # Strong bearish: body > 60% of range
                if body / (h - low) > 0.6:
                    return True
        return False

    def _near_vwap(self, price: float, vwap_value: float) -> bool:
        """Check if price is within tolerance of VWAP."""
        return abs(price - vwap_value) / vwap_value < self.vwap_tolerance

    def _was_above_vwap(self, data: pd.DataFrame, vwap: pd.Series, lookback: int = 3) -> bool:
        """Check if price was above VWAP recently."""
        for i in range(1, lookback + 1):
            if data["close"].iloc[-i - 1] > vwap.iloc[-i - 1]:
                return True
        return False

    def _was_below_vwap(self, data: pd.DataFrame, vwap: pd.Series, lookback: int = 3) -> bool:
        """Check if price was below VWAP recently."""
        for i in range(1, lookback + 1):
            if data["close"].iloc[-i - 1] < vwap.iloc[-i - 1]:
                return True
        return False

    def evaluate(self, data: pd.DataFrame) -> StrategyResult:
        """Evaluate strategy across all data."""
        if not self.validate_data(data, min_rows=50):
            return StrategyResult(signals=[], indicators={})

        vwap = self.vwap.calculate(data)
        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)
        close = data["close"]

        signals = []

        for i in range(5, len(data)):
            timestamp = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()

            if not self._near_vwap(close.iloc[i], vwap.iloc[i]):
                continue

            # Long: pullback to VWAP from above
            was_above = any(close.iloc[i - j] > vwap.iloc[i - j] for j in range(1, 4))
            if was_above and self._is_bullish_candle(data, i) and rsi.iloc[i] > 30:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.LONG,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=close.iloc[i] - (atr.iloc[i] * self.atr_multiplier),
                        take_profit=close.iloc[i] + (atr.iloc[i] * self.atr_multiplier * 2),
                        metadata={"rsi": rsi.iloc[i], "vwap": vwap.iloc[i]},
                    )
                )

            # Short: rally to VWAP from below
            was_below = any(close.iloc[i - j] < vwap.iloc[i - j] for j in range(1, 4))
            if was_below and self._is_bearish_candle(data, i) and rsi.iloc[i] < 70:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.SHORT,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=close.iloc[i] + (atr.iloc[i] * self.atr_multiplier),
                        take_profit=close.iloc[i] - (atr.iloc[i] * self.atr_multiplier * 2),
                        metadata={"rsi": rsi.iloc[i], "vwap": vwap.iloc[i]},
                    )
                )

        return StrategyResult(signals=signals, indicators={"vwap": vwap, "rsi": rsi})

    def get_entry_signal(self, data: pd.DataFrame) -> Signal | None:
        """Check for entry signal on latest bar."""
        if not self.validate_data(data, min_rows=50):
            return None

        vwap = self.vwap.calculate(data)
        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)
        close = data["close"]

        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        if not self._near_vwap(close.iloc[-1], vwap.iloc[-1]):
            return None

        # Long: pullback from above
        if (
            self._was_above_vwap(data, vwap)
            and self._is_bullish_candle(data, -1)
            and rsi.iloc[-1] > 30
        ):
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=close.iloc[-1] - (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close.iloc[-1] + (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "vwap": vwap.iloc[-1]},
            )

        # Short: rally from below
        if (
            self._was_below_vwap(data, vwap)
            and self._is_bearish_candle(data, -1)
            and rsi.iloc[-1] < 70
        ):
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SHORT,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=close.iloc[-1] + (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close.iloc[-1] - (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "vwap": vwap.iloc[-1]},
            )

        return None

    def get_exit_signal(self, data: pd.DataFrame, position_side: Side) -> Signal | None:
        """Check for exit signal - price moves away from VWAP."""
        vwap = self.vwap.calculate(data)
        close = data["close"].iloc[-1]
        atr = self.atr.calculate(data)
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        _distance = abs(close - vwap.iloc[-1])  # Keep for potential future use

        if position_side == Side.LONG:
            # Exit if price drops significantly below VWAP
            if close < vwap.iloc[-1] - atr.iloc[-1]:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.LONG,
                    signal_type=SignalType.EXIT,
                    price=close,
                )

        elif position_side == Side.SHORT:
            # Exit if price rises significantly above VWAP
            if close > vwap.iloc[-1] + atr.iloc[-1]:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.SHORT,
                    signal_type=SignalType.EXIT,
                    price=close,
                )

        return None
