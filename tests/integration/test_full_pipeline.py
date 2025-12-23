"""Integration tests for full trading pipeline."""

import pytest
import pandas as pd

from src.indicators import RSI, EMA, BollingerBands
from src.strategies import EMACrossoverStrategy, BollingerSqueezeStrategy
from src.signals import SignalGenerator, SignalManager
from src.backtesting import BacktestEngine, BacktestReport


class TestIndicatorChaining:
    """Test indicators can be chained together."""

    def test_multiple_indicators_on_same_data(self, sample_ohlcv_data):
        """Test multiple indicators process same data."""
        rsi = RSI(14)
        ema = EMA(20)
        bb = BollingerBands(20)

        rsi_values = rsi.calculate(sample_ohlcv_data)
        ema_values = ema.calculate(sample_ohlcv_data)
        bb_upper, bb_mid, bb_lower = bb.calculate_all(sample_ohlcv_data)

        # All should have same length
        assert len(rsi_values) == len(sample_ohlcv_data)
        assert len(ema_values) == len(sample_ohlcv_data)
        assert len(bb_upper) == len(sample_ohlcv_data)


class TestSignalPipeline:
    """Test signal generation pipeline."""

    def test_generator_with_multiple_strategies(self, sample_ohlcv_data):
        """Test signal generator combines multiple strategies."""
        strategies = [
            EMACrossoverStrategy("BTC/USDT", "1h"),
            BollingerSqueezeStrategy("BTC/USDT", "1h"),
        ]

        generator = SignalGenerator(strategies, min_confluence=1)
        signals = generator.generate(sample_ohlcv_data)

        # Should get signals from at least one strategy
        assert isinstance(signals, list)

    def test_confluence_filtering(self, sample_ohlcv_data):
        """Test confluence filter reduces signals."""
        strategies = [
            EMACrossoverStrategy("BTC/USDT"),
            BollingerSqueezeStrategy("BTC/USDT"),
        ]

        gen_loose = SignalGenerator(strategies, min_confluence=1)
        gen_strict = SignalGenerator(strategies, min_confluence=2)

        signals_loose = gen_loose.generate(sample_ohlcv_data)
        signals_strict = gen_strict.generate(sample_ohlcv_data)

        # Strict confluence should have fewer or equal signals
        assert len(signals_strict) <= len(signals_loose)

    def test_signal_manager_queuing(self, sample_ohlcv_data):
        """Test signal manager queues signals."""
        strategy = EMACrossoverStrategy("BTC/USDT")
        generator = SignalGenerator([strategy])

        signals = generator.generate(sample_ohlcv_data)

        manager = SignalManager(signal_expiry_minutes=60)

        for signal in signals:
            managed = manager.add_signal(signal)
            assert managed is not None

        # Check pending signals
        pending = manager.get_pending()
        assert len(pending) <= len(signals)  # May dedupe by symbol


class TestBacktestPipeline:
    """Test complete backtest pipeline."""

    def test_full_backtest_flow(self, sample_ohlcv_data):
        """Test complete backtest from strategy to report."""
        # 1. Create strategy
        strategy = EMACrossoverStrategy("BTC/USDT", "1h")

        # 2. Run backtest
        engine = BacktestEngine(
            initial_capital=10000,
            commission_pct=0.1,
            slippage_pct=0.05
        )
        result = engine.run(strategy, sample_ohlcv_data)

        # 3. Generate report
        report = BacktestReport(result)
        summary = report.summary()

        # Verify complete pipeline
        assert result.strategy_name == "EMA_Crossover"
        assert result.metrics is not None
        assert "BACKTEST REPORT" in summary

    def test_multiple_strategy_comparison(self, sample_ohlcv_data):
        """Test comparing multiple strategies."""
        strategies = [
            EMACrossoverStrategy("BTC/USDT"),
            BollingerSqueezeStrategy("BTC/USDT"),
        ]

        engine = BacktestEngine(initial_capital=10000)
        results = {}

        for strategy in strategies:
            result = engine.run(strategy, sample_ohlcv_data)
            results[strategy.name] = result

        # Compare metrics
        for name, result in results.items():
            assert result.metrics is not None
            print(f"{name}: Return={result.metrics.total_return_pct:.2f}%, "
                  f"Sharpe={result.metrics.sharpe_ratio:.2f}")

    def test_backtest_with_optimization(self, sample_ohlcv_data):
        """Test parameter optimization."""
        engine = BacktestEngine()

        param_grid = {
            "symbol": ["BTC/USDT"],
            "fast_period": [5, 9, 12],
            "slow_period": [15, 21, 26],
        }

        best = engine.optimize(
            EMACrossoverStrategy,
            sample_ohlcv_data,
            param_grid,
            metric="sharpe_ratio"
        )

        assert best is not None
        assert "fast_period" in best
        assert "slow_period" in best


class TestDataConsistency:
    """Test data consistency across pipeline."""

    def test_indicator_output_consistency(self, sample_ohlcv_data):
        """Test same input produces same output."""
        rsi = RSI(14)

        result1 = rsi.calculate(sample_ohlcv_data)
        result2 = rsi.calculate(sample_ohlcv_data)

        pd.testing.assert_series_equal(result1, result2)

    def test_strategy_determinism(self, sample_ohlcv_data):
        """Test strategy produces consistent signals."""
        strategy = EMACrossoverStrategy("BTC/USDT")

        result1 = strategy.evaluate(sample_ohlcv_data)
        result2 = strategy.evaluate(sample_ohlcv_data)

        assert len(result1.signals) == len(result2.signals)

    def test_backtest_reproducibility(self, sample_ohlcv_data):
        """Test backtest produces same results."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = EMACrossoverStrategy("BTC/USDT")

        result1 = engine.run(strategy, sample_ohlcv_data)
        result2 = engine.run(strategy, sample_ohlcv_data)

        assert result1.final_capital == pytest.approx(result2.final_capital)
        assert len(result1.trades) == len(result2.trades)
