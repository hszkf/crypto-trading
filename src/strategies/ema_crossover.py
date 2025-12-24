"""EMA Crossover Strategy with RSI Filter."""

from datetime import datetime

import pandas as pd

from ..indicators import ATR, EMA, RSI
from .base import Side, Signal, SignalType, Strategy, StrategyResult


class EMACrossoverStrategy(Strategy):
    """EMA crossover strategy with RSI confirmation and trend filter.

    Entry Long:
        - Fast EMA crosses above slow EMA
        - RSI > 50 and < 70
        - Price above trend EMA (200)

    Entry Short:
        - Fast EMA crosses below slow EMA
        - RSI < 50 and > 30
        - Price below trend EMA (200)

    Exit:
        - Opposite crossover or RSI extreme levels
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        fast_period: int = 9,
        slow_period: int = 21,
        trend_period: int = 200,
        rsi_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        super().__init__("EMA_Crossover", symbol, timeframe)
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.trend_ema = EMA(trend_period)
        self.rsi = RSI(rsi_period)
        self.atr = ATR(14)
        self.atr_multiplier = atr_multiplier

    def evaluate(self, data: pd.DataFrame) -> StrategyResult:
        """Evaluate strategy across all data."""
        if not self.validate_data(data):
            return StrategyResult(signals=[], indicators={})

        # Calculate indicators
        fast = self.fast_ema.calculate(data)
        slow = self.slow_ema.calculate(data)
        trend = self.trend_ema.calculate(data)
        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)

        signals = []
        close = data["close"]

        for i in range(1, len(data)):
            timestamp = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()

            # Long entry
            if (
                fast.iloc[i] > slow.iloc[i]
                and fast.iloc[i - 1] <= slow.iloc[i - 1]
                and close.iloc[i] > trend.iloc[i]
                and 50 < rsi.iloc[i] < 70
            ):
                stop_loss = close.iloc[i] - (atr.iloc[i] * self.atr_multiplier)
                take_profit = close.iloc[i] + (atr.iloc[i] * self.atr_multiplier * 2)

                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.LONG,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={"rsi": rsi.iloc[i], "atr": atr.iloc[i]},
                    )
                )

            # Short entry
            elif (
                fast.iloc[i] < slow.iloc[i]
                and fast.iloc[i - 1] >= slow.iloc[i - 1]
                and close.iloc[i] < trend.iloc[i]
                and 30 < rsi.iloc[i] < 50
            ):
                stop_loss = close.iloc[i] + (atr.iloc[i] * self.atr_multiplier)
                take_profit = close.iloc[i] - (atr.iloc[i] * self.atr_multiplier * 2)

                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.SHORT,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        metadata={"rsi": rsi.iloc[i], "atr": atr.iloc[i]},
                    )
                )

        return StrategyResult(
            signals=signals,
            indicators={
                "fast_ema": fast,
                "slow_ema": slow,
                "trend_ema": trend,
                "rsi": rsi,
                "atr": atr,
            },
        )

    def get_entry_signal(self, data: pd.DataFrame) -> Signal | None:
        """Check for entry signal on latest bar."""
        if not self.validate_data(data, min_rows=self.trend_ema.period + 10):
            return None

        fast = self.fast_ema.calculate(data)
        slow = self.slow_ema.calculate(data)
        trend = self.trend_ema.calculate(data)
        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)

        close = data["close"].iloc[-1]
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        # Long crossover
        if (
            fast.iloc[-1] > slow.iloc[-1]
            and fast.iloc[-2] <= slow.iloc[-2]
            and close > trend.iloc[-1]
            and 50 < rsi.iloc[-1] < 70
        ):
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=close,
                stop_loss=close - (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close + (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "atr": atr.iloc[-1]},
            )

        # Short crossover
        if (
            fast.iloc[-1] < slow.iloc[-1]
            and fast.iloc[-2] >= slow.iloc[-2]
            and close < trend.iloc[-1]
            and 30 < rsi.iloc[-1] < 50
        ):
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SHORT,
                signal_type=SignalType.ENTRY,
                price=close,
                stop_loss=close + (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close - (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "atr": atr.iloc[-1]},
            )

        return None

    def get_exit_signal(self, data: pd.DataFrame, position_side: Side) -> Signal | None:
        """Check for exit signal."""
        fast = self.fast_ema.calculate(data)
        slow = self.slow_ema.calculate(data)
        rsi = self.rsi.calculate(data)

        close = data["close"].iloc[-1]
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        if position_side == Side.LONG:
            # Exit long on bearish crossover or overbought RSI
            if (fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]) or rsi.iloc[
                -1
            ] > 80:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.LONG,
                    signal_type=SignalType.EXIT,
                    price=close,
                )

        elif position_side == Side.SHORT:
            # Exit short on bullish crossover or oversold RSI
            if (fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]) or rsi.iloc[
                -1
            ] < 20:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.SHORT,
                    signal_type=SignalType.EXIT,
                    price=close,
                )

        return None
