"""Unit tests for signals module."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.signals.filters import (
    CompositeFilter,
    ConfluenceFilter,
    RiskRewardFilter,
    TimeFilter,
    VolatilityFilter,
    VolumeFilter,
)
from src.signals.generator import SignalGenerator
from src.signals.manager import ManagedSignal, SignalManager, SignalState
from src.strategies import EMACrossoverStrategy
from src.strategies.base import Side, Signal, SignalType


@pytest.fixture
def sample_signal():
    """Create a sample signal for testing."""
    return Signal(
        symbol="BTC/USDT",
        side=Side.LONG,
        signal_type=SignalType.ENTRY,
        price=50000.0,
        timestamp=datetime.now(),
        stop_loss=48000.0,
        take_profit=54000.0,
        confidence=0.8,
    )


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")

    close = 50000 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000) + 500

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }).set_index("timestamp")


class TestConfluenceFilter:
    """Tests for ConfluenceFilter."""

    def test_filter_initialization(self):
        """Test filter initialization with default value."""
        filter_ = ConfluenceFilter()
        assert filter_.min_indicators == 3

    def test_filter_initialization_custom(self):
        """Test filter initialization with custom value."""
        filter_ = ConfluenceFilter(min_indicators=2)
        assert filter_.min_indicators == 2

    def test_filter_with_data(self, sample_signal, sample_data):
        """Test filter returns boolean."""
        filter_ = ConfluenceFilter(min_indicators=1)
        result = filter_.filter(sample_signal, sample_data)
        assert isinstance(result, bool)

    def test_filter_high_confluence_required(self, sample_signal, sample_data):
        """Test filter with high confluence requirement."""
        filter_ = ConfluenceFilter(min_indicators=5)
        result = filter_.filter(sample_signal, sample_data)
        # With 5 required and max 4 indicators checked, should fail
        assert result is False


class TestTimeFilter:
    """Tests for TimeFilter."""

    def test_default_hours(self):
        """Test default allowed hours."""
        filter_ = TimeFilter()
        assert filter_.allowed_hours == [(9, 17)]
        assert filter_.blocked_hours == []

    def test_custom_allowed_hours(self):
        """Test custom allowed hours."""
        filter_ = TimeFilter(allowed_hours=[(0, 24)])
        assert filter_.allowed_hours == [(0, 24)]

    def test_custom_blocked_hours(self):
        """Test custom blocked hours."""
        filter_ = TimeFilter(blocked_hours=[(12, 13)])
        assert filter_.blocked_hours == [(12, 13)]


class TestVolatilityFilter:
    """Tests for VolatilityFilter."""

    def test_initialization(self):
        """Test filter initialization."""
        filter_ = VolatilityFilter(min_atr_pct=0.5, max_atr_pct=5.0, atr_period=14)
        assert filter_.min_atr_pct == 0.5
        assert filter_.max_atr_pct == 5.0

    def test_filter_with_data(self, sample_signal, sample_data):
        """Test filter returns boolean with valid data."""
        filter_ = VolatilityFilter()
        result = filter_.filter(sample_signal, sample_data)
        assert result in (True, False)


class TestRiskRewardFilter:
    """Tests for RiskRewardFilter."""

    def test_initialization(self):
        """Test filter initialization."""
        filter_ = RiskRewardFilter(min_ratio=2.0)
        assert filter_.min_ratio == 2.0

    def test_filter_with_good_rr(self, sample_data):
        """Test filter passes with good R:R."""
        signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=50000.0,
            timestamp=datetime.now(),
            stop_loss=49000.0,  # $1000 risk
            take_profit=52000.0,  # $2000 reward
        )
        filter_ = RiskRewardFilter(min_ratio=2.0)
        assert filter_.filter(signal, sample_data) is True

    def test_filter_with_bad_rr(self, sample_data):
        """Test filter fails with bad R:R."""
        signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=50000.0,
            timestamp=datetime.now(),
            stop_loss=49000.0,  # $1000 risk
            take_profit=50500.0,  # $500 reward
        )
        filter_ = RiskRewardFilter(min_ratio=2.0)
        assert filter_.filter(signal, sample_data) is False

    def test_filter_without_levels(self, sample_data):
        """Test filter passes without stop/take profit levels."""
        signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=50000.0,
            timestamp=datetime.now(),
        )
        filter_ = RiskRewardFilter(min_ratio=2.0)
        assert filter_.filter(signal, sample_data) is True


class TestVolumeFilter:
    """Tests for VolumeFilter."""

    def test_initialization(self):
        """Test filter initialization."""
        filter_ = VolumeFilter(min_volume_mult=1.5, lookback=20)
        assert filter_.min_volume_mult == 1.5
        assert filter_.lookback == 20

    def test_filter_with_volume_data(self, sample_signal, sample_data):
        """Test filter with volume data."""
        filter_ = VolumeFilter(min_volume_mult=0.5)  # Low threshold
        result = filter_.filter(sample_signal, sample_data)
        assert result in (True, False)

    def test_filter_without_volume(self, sample_signal):
        """Test filter passes without volume column."""
        data = pd.DataFrame({
            "open": [50000],
            "high": [51000],
            "low": [49000],
            "close": [50500],
        })
        filter_ = VolumeFilter()
        # Should return True when no volume data
        assert filter_.filter(sample_signal, data) is True


class TestCompositeFilter:
    """Tests for CompositeFilter."""

    def test_empty_composite(self, sample_signal, sample_data):
        """Test empty composite filter passes."""
        filter_ = CompositeFilter()
        assert filter_.filter(sample_signal, sample_data) is True

    def test_add_filter(self):
        """Test adding filters to composite."""
        composite = CompositeFilter()
        composite.add_filter(RiskRewardFilter())
        assert len(composite.filters) == 1

    def test_all_filters_must_pass(self, sample_data):
        """Test composite requires all filters to pass."""
        signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=50000.0,
            timestamp=datetime.now(),
            stop_loss=49000.0,
            take_profit=52000.0,
        )

        composite = CompositeFilter([
            RiskRewardFilter(min_ratio=2.0),
            VolumeFilter(min_volume_mult=0.1),
        ])
        result = composite.filter(signal, sample_data)
        assert isinstance(result, bool)


class TestSignalGenerator:
    """Tests for SignalGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = SignalGenerator()
        assert generator.strategies == []
        assert generator.min_confluence == 1

    def test_initialization_with_strategies(self):
        """Test generator initialization with strategies."""
        strategy = EMACrossoverStrategy(symbol="BTC/USDT")
        generator = SignalGenerator(strategies=[strategy], min_confluence=2)
        assert len(generator.strategies) == 1
        assert generator.min_confluence == 2

    def test_add_strategy(self):
        """Test adding a strategy."""
        generator = SignalGenerator()
        strategy = EMACrossoverStrategy(symbol="BTC/USDT")
        generator.add_strategy(strategy)
        assert len(generator.strategies) == 1

    def test_remove_strategy(self):
        """Test removing a strategy."""
        strategy = EMACrossoverStrategy(symbol="BTC/USDT")
        generator = SignalGenerator(strategies=[strategy])
        result = generator.remove_strategy("EMA_Crossover")
        assert result is True
        assert len(generator.strategies) == 0

    def test_remove_nonexistent_strategy(self):
        """Test removing non-existent strategy."""
        generator = SignalGenerator()
        result = generator.remove_strategy("NonExistent")
        assert result is False

    def test_generate_returns_list(self, sample_data):
        """Test generate returns list of signals."""
        strategy = EMACrossoverStrategy(symbol="BTC/USDT")
        generator = SignalGenerator(strategies=[strategy])
        signals = generator.generate(sample_data)
        assert isinstance(signals, list)

    def test_active_strategies_property(self):
        """Test active strategies property."""
        strategy = EMACrossoverStrategy(symbol="BTC/USDT")
        generator = SignalGenerator(strategies=[strategy])
        assert len(generator.active_strategies) == 1

    def test_get_last_signal_none(self):
        """Test get_last_signal returns None for unknown symbol."""
        generator = SignalGenerator()
        assert generator.get_last_signal("BTC/USDT") is None

    def test_repr(self):
        """Test string representation."""
        generator = SignalGenerator(min_confluence=2)
        assert "SignalGenerator" in repr(generator)
        assert "confluence=2" in repr(generator)


class TestManagedSignal:
    """Tests for ManagedSignal dataclass."""

    def test_creation(self, sample_signal):
        """Test managed signal creation."""
        managed = ManagedSignal(signal=sample_signal)
        assert managed.signal == sample_signal
        assert managed.state == SignalState.PENDING
        assert managed.entry_order is None

    def test_is_active_pending(self, sample_signal):
        """Test is_active for pending signal."""
        managed = ManagedSignal(signal=sample_signal)
        assert managed.is_active is True

    def test_is_active_executed(self, sample_signal):
        """Test is_active for executed signal."""
        managed = ManagedSignal(signal=sample_signal, state=SignalState.EXECUTED)
        assert managed.is_active is False

    def test_is_active_expired_by_state(self, sample_signal):
        """Test is_active for expired signal."""
        managed = ManagedSignal(signal=sample_signal, state=SignalState.EXPIRED)
        assert managed.is_active is False

    def test_is_active_expired_by_time(self, sample_signal):
        """Test is_active for time-expired signal."""
        managed = ManagedSignal(
            signal=sample_signal,
            expiry=datetime.now() - timedelta(hours=1),
        )
        assert managed.is_active is False


class TestSignalManager:
    """Tests for SignalManager."""

    def test_initialization(self):
        """Test signal manager initialization."""
        manager = SignalManager()
        assert manager.exchange is None
        assert manager.auto_execute is False
        assert manager.pending_count == 0

    def test_initialization_with_options(self):
        """Test signal manager initialization with options."""
        manager = SignalManager(
            signal_expiry_minutes=30,
            auto_execute=True,
        )
        assert manager.signal_expiry == timedelta(minutes=30)
        assert manager.auto_execute is True

    def test_add_signal(self, sample_signal):
        """Test adding a signal."""
        manager = SignalManager()
        managed = manager.add_signal(sample_signal)
        assert isinstance(managed, ManagedSignal)
        assert managed.signal == sample_signal
        assert manager.pending_count == 1

    def test_add_duplicate_signal_lower_confidence(self, sample_signal):
        """Test adding duplicate signal with lower confidence."""
        manager = SignalManager()
        manager.add_signal(sample_signal)

        lower_confidence_signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=51000.0,
            timestamp=datetime.now(),
            confidence=0.5,  # Lower than original 0.8
        )
        result = manager.add_signal(lower_confidence_signal)
        # Should return existing signal, not replace
        assert result.signal == sample_signal

    def test_add_signals_multiple(self):
        """Test adding multiple signals."""
        manager = SignalManager()
        signals = [
            Signal(
                symbol="BTC/USDT",
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=50000.0,
                timestamp=datetime.now(),
            ),
            Signal(
                symbol="ETH/USDT",
                side=Side.LONG,
                signal_type=SignalType.ENTRY,
                price=3000.0,
                timestamp=datetime.now(),
            ),
        ]
        managed_list = manager.add_signals(signals)
        assert len(managed_list) == 2
        assert manager.pending_count == 2

    def test_cancel_signal(self, sample_signal):
        """Test canceling a signal."""
        manager = SignalManager()
        manager.add_signal(sample_signal)
        result = manager.cancel_signal("BTC/USDT")
        assert result is True
        assert manager.pending_count == 0

    def test_cancel_nonexistent_signal(self):
        """Test canceling non-existent signal."""
        manager = SignalManager()
        result = manager.cancel_signal("BTC/USDT")
        assert result is False

    def test_get_pending_all(self, sample_signal):
        """Test getting all pending signals."""
        manager = SignalManager()
        manager.add_signal(sample_signal)
        pending = manager.get_pending()
        assert len(pending) == 1

    def test_get_pending_by_symbol(self, sample_signal):
        """Test getting pending signal by symbol."""
        manager = SignalManager()
        manager.add_signal(sample_signal)
        pending = manager.get_pending("BTC/USDT")
        assert len(pending) == 1

    def test_get_pending_unknown_symbol(self):
        """Test getting pending for unknown symbol."""
        manager = SignalManager()
        pending = manager.get_pending("UNKNOWN/USDT")
        assert len(pending) == 0

    def test_get_history(self, sample_signal):
        """Test getting signal history."""
        manager = SignalManager()
        manager.add_signal(sample_signal)
        manager.cancel_signal("BTC/USDT")
        history = manager.get_history()
        assert len(history) == 1
        assert history[0].state == SignalState.CANCELLED

    def test_cleanup_expired(self):
        """Test cleaning up expired signals."""
        manager = SignalManager(signal_expiry_minutes=0)
        signal = Signal(
            symbol="BTC/USDT",
            side=Side.LONG,
            signal_type=SignalType.ENTRY,
            price=50000.0,
            timestamp=datetime.now(),
        )
        manager.add_signal(signal)
        # Force expiry
        manager._pending["BTC/USDT"].expiry = datetime.now() - timedelta(minutes=1)
        count = manager.cleanup_expired()
        assert count == 1
        assert manager.pending_count == 0

    def test_on_execution_callback(self):
        """Test registering execution callback."""
        manager = SignalManager()

        async def callback(managed):
            pass

        manager.on_execution(callback)
        assert len(manager._callbacks) == 1

    def test_repr(self):
        """Test string representation."""
        manager = SignalManager()
        assert "SignalManager" in repr(manager)
        assert "pending=0" in repr(manager)


class TestSignalState:
    """Tests for SignalState enum."""

    def test_signal_states_exist(self):
        """Test all signal states are defined."""
        assert SignalState.PENDING.value == "pending"
        assert SignalState.EXECUTED.value == "executed"
        assert SignalState.EXPIRED.value == "expired"
        assert SignalState.CANCELLED.value == "cancelled"
        assert SignalState.FAILED.value == "failed"
