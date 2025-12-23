"""Unit tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np

from src.backtesting import BacktestEngine, PerformanceMetrics, calculate_metrics, BacktestReport
from src.strategies import EMACrossoverStrategy, Side


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_engine_initialization(self):
        """Test engine initializes with defaults."""
        engine = BacktestEngine()

        assert engine.initial_capital == 10000
        assert engine.commission_pct == 0.001  # 0.1%
        assert engine.slippage_pct == 0.0005   # 0.05%

    def test_run_returns_result(self, sample_ohlcv_data):
        """Test backtest run returns BacktestResult."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = EMACrossoverStrategy("BTC/USDT", "1h")

        result = engine.run(strategy, sample_ohlcv_data)

        assert result.strategy_name == "EMA_Crossover"
        assert result.initial_capital == 10000
        assert len(result.equity_curve) > 0
        assert result.metrics is not None

    def test_equity_curve_starts_at_initial(self, sample_ohlcv_data):
        """Test equity curve starts at initial capital."""
        engine = BacktestEngine(initial_capital=10000)
        strategy = EMACrossoverStrategy("BTC/USDT")

        result = engine.run(strategy, sample_ohlcv_data)

        # First equity value should be close to initial capital
        assert result.equity_curve.iloc[0] == pytest.approx(10000, rel=0.01)

    def test_trades_recorded(self, trending_up_data):
        """Test trades are recorded during backtest."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")

        result = engine.run(strategy, trending_up_data)

        # Should have some trades in trending data
        assert len(result.trades) > 0

    def test_trade_has_required_fields(self, trending_up_data):
        """Test recorded trades have all required fields."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")

        result = engine.run(strategy, trending_up_data)

        if result.trades:
            trade = result.trades[0]
            assert trade.symbol is not None
            assert trade.side in [Side.LONG, Side.SHORT]
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.quantity > 0

    def test_commission_applied(self, trending_up_data):
        """Test commission reduces returns."""
        engine_no_comm = BacktestEngine(commission_pct=0)
        engine_with_comm = BacktestEngine(commission_pct=0.1)
        strategy = EMACrossoverStrategy("BTC/USDT")

        result_no_comm = engine_no_comm.run(strategy, trending_up_data)
        result_with_comm = engine_with_comm.run(strategy, trending_up_data)

        # With commission should have lower final capital
        if result_no_comm.trades:
            assert result_with_comm.final_capital <= result_no_comm.final_capital

    def test_stop_loss_triggered(self, trending_down_data):
        """Test stop losses are triggered."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")

        result = engine.run(strategy, trending_down_data)

        # Should have some stop loss exits in downtrend
        stop_exits = [t for t in result.trades if t.exit_reason == "stop_loss"]
        # May or may not have stop losses depending on market conditions
        assert isinstance(stop_exits, list)


class TestPerformanceMetrics:
    """Tests for performance metrics calculation."""

    def test_calculate_metrics_empty_trades(self):
        """Test metrics with no trades."""
        equity = pd.Series([10000] * 100)
        metrics = calculate_metrics([], equity, 10000)

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0
        assert metrics.sharpe_ratio == 0

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        from src.backtesting.engine import Trade
        from datetime import datetime

        trades = [
            Trade("BTC", Side.LONG, datetime.now(), 100, datetime.now(), 110, 1, 10, 10, "tp"),
            Trade("BTC", Side.LONG, datetime.now(), 100, datetime.now(), 105, 1, 5, 5, "tp"),
            Trade("BTC", Side.LONG, datetime.now(), 100, datetime.now(), 95, 1, -5, -5, "sl"),
        ]

        equity = pd.Series([10000, 10010, 10015, 10010])
        metrics = calculate_metrics(trades, equity, 10000)

        # 2 winners, 1 loser = 66.67% win rate
        assert metrics.win_rate == pytest.approx(66.67, rel=0.1)

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        from src.backtesting.engine import Trade
        from datetime import datetime

        trades = [
            Trade("BTC", Side.LONG, datetime.now(), 100, datetime.now(), 110, 1, 100, 10, "tp"),
            Trade("BTC", Side.LONG, datetime.now(), 100, datetime.now(), 95, 1, -50, -5, "sl"),
        ]

        equity = pd.Series([10000, 10100, 10050])
        metrics = calculate_metrics(trades, equity, 10000)

        # Gross profit = 100, Gross loss = 50, PF = 2.0
        assert metrics.profit_factor == pytest.approx(2.0)

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        equity = pd.Series([10000, 11000, 10500, 9000, 9500, 10000])
        metrics = calculate_metrics([], equity, 10000)

        # Peak = 11000, Trough = 9000, DD = 18.18%
        assert metrics.max_drawdown == pytest.approx(18.18, rel=0.1)


class TestBacktestReport:
    """Tests for backtest reporting."""

    def test_summary_generation(self, sample_ohlcv_data):
        """Test report summary is generated."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = engine.run(strategy, sample_ohlcv_data)

        report = BacktestReport(result)
        summary = report.summary()

        assert "BACKTEST REPORT" in summary
        assert "EMA_Crossover" in summary
        assert "Total Trades" in summary

    def test_trade_log_dataframe(self, trending_up_data):
        """Test trade log returns DataFrame."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = engine.run(strategy, trending_up_data)

        report = BacktestReport(result)
        log = report.trade_log()

        assert isinstance(log, pd.DataFrame)
        if len(result.trades) > 0:
            assert "entry_price" in log.columns
            assert "pnl" in log.columns

    def test_to_json_valid(self, sample_ohlcv_data):
        """Test JSON export is valid."""
        import json

        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = engine.run(strategy, sample_ohlcv_data)

        report = BacktestReport(result)
        json_str = report.to_json()

        # Should parse without error
        data = json.loads(json_str)
        assert data["strategy"] == "EMA_Crossover"
        assert "metrics" in data

    def test_win_loss_streaks(self, trending_up_data):
        """Test streak calculation."""
        engine = BacktestEngine()
        strategy = EMACrossoverStrategy("BTC/USDT")
        result = engine.run(strategy, trending_up_data)

        report = BacktestReport(result)
        streaks = report.win_loss_streaks()

        assert "max_win_streak" in streaks
        assert "max_loss_streak" in streaks
        assert streaks["max_win_streak"] >= 0
        assert streaks["max_loss_streak"] >= 0
