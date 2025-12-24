"""Bollinger Band Squeeze Breakout Strategy."""

from datetime import datetime

import pandas as pd

from ..indicators import ATR, BollingerBands, KeltnerChannels
from .base import Side, Signal, SignalType, Strategy, StrategyResult


class BollingerSqueezeStrategy(Strategy):
    """Bollinger Band squeeze breakout strategy.

    Detects low volatility periods (squeeze) where Bollinger Bands
    contract inside Keltner Channels, then trades the breakout.

    Entry:
        - BB inside KC (squeeze active)
        - Candle closes outside BB with volume spike
        - Direction determines long/short

    Exit:
        - Price returns to middle band
        - Or 2x band width from entry
    """

    def __init__(self, symbol: str, timeframe: str = "1h",
                 bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_multiplier: float = 1.5,
                 volume_threshold: float = 1.5):
        super().__init__("BB_Squeeze", symbol, timeframe)
        self.bb = BollingerBands(bb_period, bb_std)
        self.kc = KeltnerChannels(kc_period, 10, kc_multiplier)
        self.atr = ATR(14)
        self.volume_threshold = volume_threshold
        self.squeeze_bars = 6  # Minimum bars in squeeze

    def _detect_squeeze(self, data: pd.DataFrame) -> pd.Series:
        """Detect squeeze condition."""
        return self.kc.squeeze_with_bb(data, self.bb.period, self.bb.std_dev)

    def _volume_spike(self, data: pd.DataFrame) -> pd.Series:
        """Detect volume spike above threshold."""
        avg_vol = data["volume"].rolling(window=20).mean()
        return data["volume"] > (avg_vol * self.volume_threshold)

    def evaluate(self, data: pd.DataFrame) -> StrategyResult:
        """Evaluate strategy across all data."""
        if not self.validate_data(data):
            return StrategyResult(signals=[], indicators={})

        bb_upper, bb_middle, bb_lower = self.bb.calculate_all(data)
        squeeze = self._detect_squeeze(data)
        vol_spike = self._volume_spike(data)
        _atr = self.atr.calculate(data)  # Keep for potential future use

        signals = []
        close = data["close"]

        # Track consecutive squeeze bars
        squeeze_count = 0

        for i in range(1, len(data)):
            if squeeze.iloc[i]:
                squeeze_count += 1
            else:
                # Check for breakout after sufficient squeeze
                if squeeze_count >= self.squeeze_bars:
                    timestamp = data.index[i] if isinstance(data.index[i], datetime) else datetime.now()

                    # Bullish breakout
                    if close.iloc[i] > bb_upper.iloc[i] and vol_spike.iloc[i]:
                        band_width = bb_upper.iloc[i] - bb_lower.iloc[i]
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=self.symbol,
                            side=Side.LONG,
                            signal_type=SignalType.ENTRY,
                            price=close.iloc[i],
                            stop_loss=bb_middle.iloc[i],
                            take_profit=close.iloc[i] + (band_width * 2),
                            metadata={"squeeze_bars": squeeze_count, "band_width": band_width}
                        ))

                    # Bearish breakout
                    elif close.iloc[i] < bb_lower.iloc[i] and vol_spike.iloc[i]:
                        band_width = bb_upper.iloc[i] - bb_lower.iloc[i]
                        signals.append(Signal(
                            timestamp=timestamp,
                            symbol=self.symbol,
                            side=Side.SHORT,
                            signal_type=SignalType.ENTRY,
                            price=close.iloc[i],
                            stop_loss=bb_middle.iloc[i],
                            take_profit=close.iloc[i] - (band_width * 2),
                            metadata={"squeeze_bars": squeeze_count, "band_width": band_width}
                        ))

                squeeze_count = 0

        return StrategyResult(
            signals=signals,
            indicators={
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "squeeze": squeeze,
                "volume_spike": vol_spike
            }
        )

    def get_entry_signal(self, data: pd.DataFrame) -> Signal | None:
        """Check for entry signal on latest bar."""
        if not self.validate_data(data, min_rows=50):
            return None

        bb_upper, bb_middle, bb_lower = self.bb.calculate_all(data)
        squeeze = self._detect_squeeze(data)
        vol_spike = self._volume_spike(data)

        close = data["close"]
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        # Count recent squeeze bars
        squeeze_count = squeeze.iloc[-self.squeeze_bars-1:-1].sum()

        # Need sufficient squeeze before breakout
        if squeeze_count < self.squeeze_bars:
            return None

        # Currently not in squeeze (breakout happening)
        if squeeze.iloc[-1]:
            return None

        band_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]

        # Bullish breakout
        if close.iloc[-1] > bb_upper.iloc[-1] and vol_spike.iloc[-1]:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=bb_middle.iloc[-1],
                take_profit=close.iloc[-1] + (band_width * 2),
                metadata={"squeeze_bars": squeeze_count, "band_width": band_width}
            )

        # Bearish breakout
        if close.iloc[-1] < bb_lower.iloc[-1] and vol_spike.iloc[-1]:
            return Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SHORT,
                signal_type=SignalType.ENTRY,
                price=close.iloc[-1],
                stop_loss=bb_middle.iloc[-1],
                take_profit=close.iloc[-1] - (band_width * 2),
                metadata={"squeeze_bars": squeeze_count, "band_width": band_width}
            )

        return None

    def get_exit_signal(self, data: pd.DataFrame, position_side: Side) -> Signal | None:
        """Check for exit signal - price returns to middle band."""
        bb_upper, bb_middle, bb_lower = self.bb.calculate_all(data)
        close = data["close"].iloc[-1]
        timestamp = data.index[-1] if isinstance(data.index[-1], datetime) else datetime.now()

        if position_side == Side.LONG:
            if close < bb_middle.iloc[-1]:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.LONG,
                    signal_type=SignalType.EXIT,
                    price=close
                )

        elif position_side == Side.SHORT:
            if close > bb_middle.iloc[-1]:
                return Signal(
                    timestamp=timestamp,
                    symbol=self.symbol,
                    side=Side.SHORT,
                    signal_type=SignalType.EXIT,
                    price=close
                )

        return None
