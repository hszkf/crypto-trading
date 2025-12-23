"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Generate realistic price movement
    returns = np.random.normal(0.0001, 0.02, n_bars)
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.uniform(1000, 10000, n_bars)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)

    return df


@pytest.fixture
def trending_up_data() -> pd.DataFrame:
    """Generate uptrending price data."""
    np.random.seed(123)
    n_bars = 300

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Strong uptrend with some noise
    trend = np.linspace(0, 0.5, n_bars)
    noise = np.random.normal(0, 0.01, n_bars)
    close = 100 * np.exp(trend + np.cumsum(noise))

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.uniform(5000, 15000, n_bars)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)


@pytest.fixture
def trending_down_data() -> pd.DataFrame:
    """Generate downtrending price data."""
    np.random.seed(456)
    n_bars = 300

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Strong downtrend
    trend = np.linspace(0, -0.4, n_bars)
    noise = np.random.normal(0, 0.01, n_bars)
    close = 100 * np.exp(trend + np.cumsum(noise))

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.uniform(5000, 15000, n_bars)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)


@pytest.fixture
def ranging_data() -> pd.DataFrame:
    """Generate ranging/sideways price data."""
    np.random.seed(789)
    n_bars = 300

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Sideways movement around 100
    close = 100 + np.cumsum(np.random.normal(0, 0.5, n_bars))
    close = np.clip(close, 90, 110)  # Keep in range

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.uniform(3000, 8000, n_bars)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)
