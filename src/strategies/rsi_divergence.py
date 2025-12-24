"""RSI Divergence Strategy."""

from datetime import datetime

import numpy as np
import pandas as pd

from ..indicators import ATR, RSI
from .base import Side, Signal, SignalType, Strategy, StrategyResult


class RSIDivergenceStrategy(Strategy):
    """RSI divergence strategy.

    Detects price/RSI divergences for potential reversals.

    Bullish Divergence:
        - Price makes lower low
        - RSI makes higher low
        - Enter on RSI crossing above 30

    Bearish Divergence:
        - Price makes higher high
        - RSI makes lower high
        - Enter on RSI crossing below 70
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        rsi_period: int = 14,
        lookback: int = 20,
        atr_multiplier: float = 2.0,
    ):
        super().__init__("RSI_Divergence", symbol, timeframe)
        self.rsi = RSI(rsi_period)
        self.atr = ATR(14)
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier

    def _find_swing_points(self, series: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
        """Find swing high and low points efficiently using rolling windows."""
        # Use rolling min/max for efficient swing detection
        roll_min = series.rolling(window=2 * window + 1, center=True).min()
        roll_max = series.rolling(window=2 * window + 1, center=True).max()

        swing_lows = series == roll_min
        swing_highs = series == roll_max

        return swing_lows, swing_highs

    def _detect_divergences(
        self, price: pd.Series, rsi: pd.Series, lookback: int
    ) -> tuple[pd.Series, pd.Series]:
        """Detect bullish and bearish divergences efficiently."""
        _n = len(price)  # Keep for potential debugging
        bull_div = pd.Series(False, index=price.index)
        bear_div = pd.Series(False, index=price.index)

        # Precompute swing points once for entire series
        price_swing_lows, price_swing_highs = self._find_swing_points(price)
        rsi_swing_lows, rsi_swing_highs = self._find_swing_points(rsi)

        # Get indices of swing points
        price_low_indices = np.where(price_swing_lows)[0]
        price_high_indices = np.where(price_swing_highs)[0]
        rsi_low_indices = np.where(rsi_swing_lows)[0]
        rsi_high_indices = np.where(rsi_swing_highs)[0]

        # Detect bullish divergence: price lower low, RSI higher low
        for i in range(1, len(price_low_indices)):
            curr_idx = price_low_indices[i]
            prev_idx = price_low_indices[i - 1]

            # Check if within lookback window
            if curr_idx - prev_idx > lookback:
                continue

            # Price makes lower low
            if price.iloc[curr_idx] < price.iloc[prev_idx]:
                # Find corresponding RSI lows
                rsi_lows_in_range = rsi_low_indices[
                    (rsi_low_indices >= prev_idx - 3) & (rsi_low_indices <= curr_idx + 3)
                ]
                if len(rsi_lows_in_range) >= 2:
                    # RSI makes higher low
                    if rsi.iloc[rsi_lows_in_range[-1]] > rsi.iloc[rsi_lows_in_range[-2]]:
                        bull_div.iloc[curr_idx] = True

        # Detect bearish divergence: price higher high, RSI lower high
        for i in range(1, len(price_high_indices)):
            curr_idx = price_high_indices[i]
            prev_idx = price_high_indices[i - 1]

            # Check if within lookback window
            if curr_idx - prev_idx > lookback:
                continue

            # Price makes higher high
            if price.iloc[curr_idx] > price.iloc[prev_idx]:
                # Find corresponding RSI highs
                rsi_highs_in_range = rsi_high_indices[
                    (rsi_high_indices >= prev_idx - 3) & (rsi_high_indices <= curr_idx + 3)
                ]
                if len(rsi_highs_in_range) >= 2:
                    # RSI makes lower high
                    if rsi.iloc[rsi_highs_in_range[-1]] < rsi.iloc[rsi_highs_in_range[-2]]:
                        bear_div.iloc[curr_idx] = True

        return bull_div, bear_div

    def evaluate(self, data: pd.DataFrame) -> StrategyResult:
        """Evaluate strategy across all data."""
        if not self.validate_data(data):
            return StrategyResult(signals=[], indicators={})

        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)
        close = data["close"]

        bull_div, bear_div = self._detect_divergences(close, rsi, self.lookback)

        signals = []

        for i in range(1, len(data)):
            timestamp = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()

            # Bullish: divergence + RSI crossing above 30
            if bull_div.iloc[i] and rsi.iloc[i] > 30 and rsi.iloc[i - 1] <= 30:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.LONG,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=close.iloc[i] - (atr.iloc[i] * self.atr_multiplier),
                        take_profit=close.iloc[i] + (atr.iloc[i] * self.atr_multiplier * 2),
                        metadata={"rsi": rsi.iloc[i], "divergence_type": "bullish"},
                    )
                )

            # Bearish: divergence + RSI crossing below 70
            if bear_div.iloc[i] and rsi.iloc[i] < 70 and rsi.iloc[i - 1] >= 70:
                signals.append(
                    Signal(
                        timestamp=timestamp,
                        symbol=self.symbol,
                        side=Side.SHORT,
                        signal_type=SignalType.ENTRY,
                        price=close.iloc[i],
                        stop_loss=close.iloc[i] + (atr.iloc[i] * self.atr_multiplier),
                        take_profit=close.iloc[i] - (atr.iloc[i] * self.atr_multiplier * 2),
                        metadata={"rsi": rsi.iloc[i], "divergence_type": "bearish"},
                    )
                )

        return StrategyResult(
            signals=signals,
            indicators={"rsi": rsi, "bullish_divergence": bull_div, "bearish_divergence": bear_div},
        )

    def get_entry_signal(self, data: pd.DataFrame) -> Signal | None:
        """Check for entry signal on latest bar."""
        if not self.validate_data(data, min_rows=self.lookback + 20):
            return None

        rsi = self.rsi.calculate(data)
        atr = self.atr.calculate(data)
        close = data["close"]

        bull_div, bear_div = self._detect_divergences(close, rsi, self.lookback)

        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        # Bullish entry
        if bull_div.iloc[-1] and rsi.iloc[-1] > 30 and rsi.iloc[-2] <= 30:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=close.iloc[-1] - (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close.iloc[-1] + (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "divergence_type": "bullish"},
            )

        # Bearish entry
        if bear_div.iloc[-1] and rsi.iloc[-1] < 70 and rsi.iloc[-2] >= 70:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SHORT,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=close.iloc[-1] + (atr.iloc[-1] * self.atr_multiplier),
                take_profit=close.iloc[-1] - (atr.iloc[-1] * self.atr_multiplier * 2),
                metadata={"rsi": rsi.iloc[-1], "divergence_type": "bearish"},
            )

        return None

    def get_exit_signal(self, data: pd.DataFrame, position_side: Side) -> Signal | None:
        """Check for exit signal based on RSI extreme levels."""
        rsi = self.rsi.calculate(data)
        close = data["close"].iloc[-1]
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        if position_side == Side.LONG and rsi.iloc[-1] > 70:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.LONG,
                signal_type=SignalType.EXIT,
                price=close,
            )

        if position_side == Side.SHORT and rsi.iloc[-1] < 30:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SHORT,
                signal_type=SignalType.EXIT,
                price=close,
            )

        return None
