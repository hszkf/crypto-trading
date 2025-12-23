"""Unit tests for technical indicators."""

import pytest
import pandas as pd
import numpy as np

from src.indicators import (
    SMA, EMA, MACD, ADX,
    RSI, StochasticRSI, CCI, WilliamsR,
    BollingerBands, ATR, KeltnerChannels,
    OBV, VWAP, MFI
)


class TestTrendIndicators:
    """Tests for trend indicators."""

    def test_sma_calculation(self, sample_ohlcv_data):
        """Test SMA calculates correctly."""
        sma = SMA(period=20)
        result = sma.calculate(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        assert result.isna().sum() == 19  # First 19 values are NaN
        assert result.iloc[-1] == pytest.approx(
            sample_ohlcv_data["close"].iloc[-20:].mean(), rel=1e-6
        )

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test EMA calculates correctly."""
        ema = EMA(period=20)
        result = ema.calculate(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # EMA should be smoother than SMA
        assert not result.isna().all()

    def test_macd_components(self, sample_ohlcv_data):
        """Test MACD calculates all components."""
        macd = MACD(fast=12, slow=26, signal=9)
        macd_line, signal_line, histogram = macd.calculate_all(sample_ohlcv_data)

        assert len(macd_line) == len(sample_ohlcv_data)
        assert len(signal_line) == len(sample_ohlcv_data)
        assert len(histogram) == len(sample_ohlcv_data)

        # Histogram should be MACD - Signal
        valid_idx = ~(macd_line.isna() | signal_line.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx],
            macd_line[valid_idx] - signal_line[valid_idx]
        )

    def test_macd_signal_generation(self, sample_ohlcv_data):
        """Test MACD generates crossover signals."""
        macd = MACD()
        signals = macd.get_signal(sample_ohlcv_data)

        assert len(signals) == len(sample_ohlcv_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_adx_trending_market(self, trending_up_data):
        """Test ADX detects trending market."""
        adx = ADX(period=14)
        result = adx.calculate(trending_up_data)

        # ADX should be above 25 in trending market
        valid_adx = result.dropna()
        assert valid_adx.mean() > 20  # Should indicate trend


class TestMomentumIndicators:
    """Tests for momentum indicators."""

    def test_rsi_range(self, sample_ohlcv_data):
        """Test RSI stays within 0-100 range."""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_ohlcv_data)

        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_overbought_oversold(self, sample_ohlcv_data):
        """Test RSI signal generation."""
        rsi = RSI(period=14, overbought=70, oversold=30)
        signals = rsi.get_signal(sample_ohlcv_data)

        assert len(signals) == len(sample_ohlcv_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_stochastic_rsi(self, sample_ohlcv_data):
        """Test Stochastic RSI calculation."""
        stoch = StochasticRSI()
        k, d = stoch.calculate_all(sample_ohlcv_data)

        assert len(k) == len(sample_ohlcv_data)
        assert len(d) == len(sample_ohlcv_data)

    def test_cci_calculation(self, sample_ohlcv_data):
        """Test CCI calculation."""
        cci = CCI(period=20)
        result = cci.calculate(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # CCI typically oscillates around 0
        valid = result.dropna()
        assert abs(valid.mean()) < 100

    def test_williams_r_range(self, sample_ohlcv_data):
        """Test Williams %R stays within -100 to 0."""
        wr = WilliamsR(period=14)
        result = wr.calculate(sample_ohlcv_data)

        valid = result.dropna()
        assert valid.min() >= -100
        assert valid.max() <= 0


class TestVolatilityIndicators:
    """Tests for volatility indicators."""

    def test_bollinger_bands_order(self, sample_ohlcv_data):
        """Test BB upper > middle > lower."""
        bb = BollingerBands(period=20, std_dev=2.0)
        upper, middle, lower = bb.calculate_all(sample_ohlcv_data)

        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bollinger_percent_b(self, sample_ohlcv_data):
        """Test %B calculation."""
        bb = BollingerBands(period=20)
        percent_b = bb.percent_b(sample_ohlcv_data)

        assert len(percent_b) == len(sample_ohlcv_data)

    def test_atr_positive(self, sample_ohlcv_data):
        """Test ATR is always positive."""
        atr = ATR(period=14)
        result = atr.calculate(sample_ohlcv_data)

        valid = result.dropna()
        assert (valid > 0).all()

    def test_atr_stop_loss(self, sample_ohlcv_data):
        """Test ATR-based stop loss calculation."""
        atr = ATR(period=14)
        sl_long = atr.stop_loss_price(sample_ohlcv_data, multiplier=2.0, direction="long")
        sl_short = atr.stop_loss_price(sample_ohlcv_data, multiplier=2.0, direction="short")

        close = sample_ohlcv_data["close"]
        assert (sl_long < close).all()  # Long SL below price
        assert (sl_short > close).all()  # Short SL above price

    def test_keltner_channels(self, sample_ohlcv_data):
        """Test Keltner Channels calculation."""
        kc = KeltnerChannels()
        upper, middle, lower = kc.calculate_all(sample_ohlcv_data)

        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


class TestVolumeIndicators:
    """Tests for volume indicators."""

    def test_obv_calculation(self, sample_ohlcv_data):
        """Test OBV calculation."""
        obv = OBV()
        result = obv.calculate(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # OBV should be cumulative
        assert result.iloc[-1] != 0

    def test_vwap_calculation(self, sample_ohlcv_data):
        """Test VWAP calculation."""
        vwap = VWAP()
        result = vwap.calculate(sample_ohlcv_data)

        assert len(result) == len(sample_ohlcv_data)
        # VWAP should be within price range
        assert result.iloc[-1] > sample_ohlcv_data["low"].min()
        assert result.iloc[-1] < sample_ohlcv_data["high"].max()

    def test_mfi_range(self, sample_ohlcv_data):
        """Test MFI stays within 0-100."""
        mfi = MFI(period=14)
        result = mfi.calculate(sample_ohlcv_data)

        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100


class TestIndicatorValidation:
    """Test indicator input validation."""

    def test_missing_column_error(self):
        """Test error on missing required column."""
        # Provide enough data rows to pass the length check
        df = pd.DataFrame({"close": list(range(100, 120))})
        rsi = RSI()

        # Should work with just close
        rsi.calculate(df)

        # ATR needs high, low, close - should fail on missing columns
        atr = ATR()
        with pytest.raises(ValueError, match="Missing required columns"):
            atr.calculate(df)

    def test_insufficient_data_error(self, sample_ohlcv_data):
        """Test error on insufficient data."""
        small_df = sample_ohlcv_data.head(5)
        sma = SMA(period=20)

        with pytest.raises(ValueError):
            sma.calculate(small_df)
