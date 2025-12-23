"""Unit tests for trading strategies."""

import pytest
import pandas as pd

from src.strategies import (
    Strategy, StrategyResult, Signal, Side, SignalType,
    EMACrossoverStrategy,
    BollingerSqueezeStrategy,
    RSIDivergenceStrategy,
    VWAPBounceStrategy
)


class TestStrategyBase:
    """Tests for base strategy functionality."""

    def test_strategy_repr(self):
        """Test strategy string representation."""
        strategy = EMACrossoverStrategy("BTC/USDT", "1h")
        repr_str = repr(strategy)

        assert "EMACrossoverStrategy" in repr_str
        assert "BTC/USDT" in repr_str

    def test_position_size_calculation(self, sample_ohlcv_data):
        """Test position sizing logic."""
        strategy = EMACrossoverStrategy("BTC/USDT")

        size = strategy.calculate_position_size(
            account_value=10000,
            risk_percent=0.02,
            entry=100,
            stop_loss=95
        )

        # Risk = $200 (2% of 10000)
        # Stop distance = $5 (100 - 95)
        # Position = 200 / 5 = 40 units
        assert size == pytest.approx(40, rel=0.01)

    def test_validate_data(self, sample_ohlcv_data):
        """Test data validation."""
        strategy = EMACrossoverStrategy("BTC/USDT")

        assert strategy.validate_data(sample_ohlcv_data)
        assert not strategy.validate_data(sample_ohlcv_data.head(10))  # Too little data


class TestEMACrossoverStrategy:
    """Tests for EMA Crossover strategy."""

    def test_evaluate_returns_result(self, sample_ohlcv_data):
        """Test evaluate returns StrategyResult."""
        strategy = EMACrossoverStrategy("BTC/USDT", "1h")
        result = strategy.evaluate(sample_ohlcv_data)

        assert isinstance(result, StrategyResult)
        assert "fast_ema" in result.indicators
        assert "slow_ema" in result.indicators
        assert "rsi" in result.indicators

    def test_signals_have_stops(self, sample_ohlcv_data):
        """Test signals include stop loss and take profit."""
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = strategy.evaluate(sample_ohlcv_data)

        for signal in result.signals:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None

    def test_long_signals_in_uptrend(self, trending_up_data):
        """Test strategy generates long signals in uptrend."""
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = strategy.evaluate(trending_up_data)

        long_signals = [s for s in result.signals if s.side == Side.LONG]
        short_signals = [s for s in result.signals if s.side == Side.SHORT]

        # Should have more longs than shorts in uptrend
        assert len(long_signals) >= len(short_signals)

    def test_short_signals_in_downtrend(self, trending_down_data):
        """Test strategy generates short signals in downtrend."""
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = strategy.evaluate(trending_down_data)

        long_signals = [s for s in result.signals if s.side == Side.LONG]
        short_signals = [s for s in result.signals if s.side == Side.SHORT]

        # Should have more shorts than longs in downtrend
        assert len(short_signals) >= len(long_signals)

    def test_exit_signal_generation(self, sample_ohlcv_data):
        """Test exit signal generation."""
        strategy = EMACrossoverStrategy("BTC/USDT")

        # Check for long exit
        exit_signal = strategy.get_exit_signal(sample_ohlcv_data, Side.LONG)
        # May or may not have exit signal depending on data
        if exit_signal:
            assert exit_signal.signal_type == SignalType.EXIT


class TestBollingerSqueezeStrategy:
    """Tests for Bollinger Squeeze strategy."""

    def test_evaluate_returns_result(self, sample_ohlcv_data):
        """Test evaluate returns proper result."""
        strategy = BollingerSqueezeStrategy("BTC/USDT")
        result = strategy.evaluate(sample_ohlcv_data)

        assert isinstance(result, StrategyResult)
        assert "bb_upper" in result.indicators
        assert "squeeze" in result.indicators

    def test_squeeze_detection(self, ranging_data):
        """Test squeeze detection in low volatility."""
        strategy = BollingerSqueezeStrategy("BTC/USDT")
        result = strategy.evaluate(ranging_data)

        # Check squeeze indicator exists
        assert "squeeze" in result.indicators
        squeeze = result.indicators["squeeze"]
        assert squeeze.sum() > 0  # Should detect some squeeze periods


class TestRSIDivergenceStrategy:
    """Tests for RSI Divergence strategy."""

    def test_evaluate_returns_result(self, sample_ohlcv_data):
        """Test evaluate returns proper result."""
        strategy = RSIDivergenceStrategy("BTC/USDT")
        result = strategy.evaluate(sample_ohlcv_data)

        assert isinstance(result, StrategyResult)
        assert "rsi" in result.indicators
        assert "bullish_divergence" in result.indicators
        assert "bearish_divergence" in result.indicators

    def test_divergence_signals_have_metadata(self, sample_ohlcv_data):
        """Test divergence signals include type metadata."""
        strategy = RSIDivergenceStrategy("BTC/USDT")
        result = strategy.evaluate(sample_ohlcv_data)

        for signal in result.signals:
            assert "divergence_type" in signal.metadata


class TestVWAPBounceStrategy:
    """Tests for VWAP Bounce strategy."""

    def test_evaluate_returns_result(self, sample_ohlcv_data):
        """Test evaluate returns proper result."""
        strategy = VWAPBounceStrategy("BTC/USDT", "15m")
        result = strategy.evaluate(sample_ohlcv_data)

        assert isinstance(result, StrategyResult)
        assert "vwap" in result.indicators
        assert "rsi" in result.indicators

    def test_signals_near_vwap(self, sample_ohlcv_data):
        """Test signals occur near VWAP."""
        strategy = VWAPBounceStrategy("BTC/USDT", vwap_tolerance=0.01)
        result = strategy.evaluate(sample_ohlcv_data)

        vwap = result.indicators["vwap"]

        for signal in result.signals:
            # Find the bar where signal occurred
            idx = sample_ohlcv_data.index.get_loc(signal.timestamp)
            price = sample_ohlcv_data["close"].iloc[idx]
            vwap_val = vwap.iloc[idx]

            # Price should be within tolerance of VWAP
            assert abs(price - vwap_val) / vwap_val < strategy.vwap_tolerance * 2


class TestSignalProperties:
    """Tests for Signal dataclass properties."""

    def test_risk_reward_ratio(self):
        """Test R:R calculation."""
        signal = Signal(
            timestamp=pd.Timestamp.now(),
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=100,
            stop_loss=95,
            take_profit=115
        )

        # Risk = 5, Reward = 15, R:R = 3
        assert signal.risk_reward_ratio == pytest.approx(3.0)

    def test_risk_reward_none_without_levels(self):
        """Test R:R is None without SL/TP."""
        signal = Signal(
            timestamp=pd.Timestamp.now(),
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=100
        )

        assert signal.risk_reward_ratio is None
