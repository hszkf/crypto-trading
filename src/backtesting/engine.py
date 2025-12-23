"""Backtesting engine for strategy evaluation."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import pandas as pd
import numpy as np

from ..strategies.base import Strategy, Signal, Side, SignalType
from .metrics import PerformanceMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Trade duration in hours."""
        return (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class Position:
    """Open position during backtest."""
    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: list[Trade]
    equity_curve: pd.Series
    metrics: PerformanceMetrics


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates strategy execution on historical data with
    realistic fills, slippage, and commission handling.
    """

    def __init__(self, initial_capital: float = 10000,
                 commission_pct: float = 0.1,
                 slippage_pct: float = 0.05,
                 risk_per_trade: float = 0.02):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital in quote currency
            commission_pct: Commission as percentage (0.1 = 0.1%)
            slippage_pct: Slippage as percentage
            risk_per_trade: Risk per trade as decimal (0.02 = 2%)
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.risk_per_trade = risk_per_trade

        self._capital = initial_capital
        self._position: Optional[Position] = None
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []

    def run(self, strategy: Strategy, data: pd.DataFrame) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            strategy: Trading strategy to test
            data: OHLCV DataFrame with datetime index

        Returns:
            BacktestResult with trades and metrics
        """
        self._reset()

        logger.info(f"Starting backtest: {strategy.name} on {strategy.symbol}")
        logger.info(f"Period: {data.index[0]} to {data.index[-1]}")

        # Determine warmup period based on strategy attributes
        warmup = 200  # Default warmup
        if hasattr(strategy, 'trend_ema'):
            warmup = strategy.trend_ema.period
        elif hasattr(strategy, 'lookback'):
            warmup = max(strategy.lookback + 50, 100)
        elif hasattr(strategy, 'bb'):
            warmup = strategy.bb.period + 50

        for i in range(min(warmup, len(data) - 50), len(data)):
            # Get data up to current bar
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            timestamp = data.index[i]

            # Record equity
            self._update_equity(current_bar["close"])

            # Check for exit if in position
            if self._position:
                exit_signal = self._check_exit(strategy, current_data, current_bar, timestamp)
                if exit_signal:
                    continue

            # Check for entry if not in position
            else:
                self._check_entry(strategy, current_data, current_bar, timestamp)

        # Close any remaining position
        if self._position:
            self._close_position(data.iloc[-1]["close"], data.index[-1], "end_of_data")

        # Calculate metrics
        equity_series = pd.Series(self._equity_curve, index=data.index[-len(self._equity_curve):])
        metrics = calculate_metrics(self._trades, equity_series, self.initial_capital)

        return BacktestResult(
            strategy_name=strategy.name,
            symbol=strategy.symbol,
            timeframe=strategy.timeframe,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self._capital,
            trades=self._trades,
            equity_curve=equity_series,
            metrics=metrics
        )

    def _reset(self) -> None:
        """Reset engine state for new backtest."""
        self._capital = self.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = []

    def _update_equity(self, current_price: float) -> None:
        """Update equity curve with current position value."""
        equity = self._capital
        if self._position:
            if self._position.side == Side.LONG:
                pnl = (current_price - self._position.entry_price) * self._position.quantity
            else:
                pnl = (self._position.entry_price - current_price) * self._position.quantity
            equity += pnl
        self._equity_curve.append(equity)

    def _check_entry(self, strategy: Strategy, data: pd.DataFrame,
                     bar: pd.Series, timestamp: datetime) -> None:
        """Check for entry signal."""
        signal = strategy.get_entry_signal(data)
        if not signal:
            return

        # Calculate position size with proper risk management
        max_position_value = self._capital * 0.25  # Max 25% of capital per trade

        if signal.stop_loss:
            risk_amount = self._capital * self.risk_per_trade
            stop_distance = abs(bar["close"] - signal.stop_loss)
            if stop_distance > 0:
                quantity = risk_amount / stop_distance
            else:
                quantity = (self._capital * 0.1) / bar["close"]
        else:
            quantity = (self._capital * 0.1) / bar["close"]

        # Cap position size to max percentage of capital
        position_value = quantity * bar["close"]
        if position_value > max_position_value:
            quantity = max_position_value / bar["close"]

        # Apply slippage
        if signal.side == Side.LONG:
            entry_price = bar["close"] * (1 + self.slippage_pct)
        else:
            entry_price = bar["close"] * (1 - self.slippage_pct)

        # Apply commission
        commission = entry_price * quantity * self.commission_pct
        self._capital -= commission

        self._position = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_time=timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=signal.metadata
        )

        logger.debug(f"Entry: {signal.side.value} {quantity:.4f} @ {entry_price:.2f}")

    def _check_exit(self, strategy: Strategy, data: pd.DataFrame,
                    bar: pd.Series, timestamp: datetime) -> bool:
        """Check for exit conditions. Returns True if position closed."""
        pos = self._position

        # Check stop loss
        if pos.stop_loss:
            if pos.side == Side.LONG and bar["low"] <= pos.stop_loss:
                self._close_position(pos.stop_loss, timestamp, "stop_loss")
                return True
            elif pos.side == Side.SHORT and bar["high"] >= pos.stop_loss:
                self._close_position(pos.stop_loss, timestamp, "stop_loss")
                return True

        # Check take profit
        if pos.take_profit:
            if pos.side == Side.LONG and bar["high"] >= pos.take_profit:
                self._close_position(pos.take_profit, timestamp, "take_profit")
                return True
            elif pos.side == Side.SHORT and bar["low"] <= pos.take_profit:
                self._close_position(pos.take_profit, timestamp, "take_profit")
                return True

        # Check strategy exit signal
        exit_signal = strategy.get_exit_signal(data, pos.side)
        if exit_signal:
            self._close_position(bar["close"], timestamp, "signal")
            return True

        return False

    def _close_position(self, exit_price: float, timestamp: datetime,
                        reason: str) -> None:
        """Close current position and record trade."""
        pos = self._position
        if not pos:
            return

        # Apply slippage
        if pos.side == Side.LONG:
            adjusted_price = exit_price * (1 - self.slippage_pct)
        else:
            adjusted_price = exit_price * (1 + self.slippage_pct)

        # Calculate PnL
        if pos.side == Side.LONG:
            pnl = (adjusted_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - adjusted_price) * pos.quantity

        # Apply commission
        commission = adjusted_price * pos.quantity * self.commission_pct
        pnl -= commission

        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100

        # Update capital
        self._capital += pos.entry_price * pos.quantity + pnl

        # Record trade
        trade = Trade(
            symbol=pos.symbol,
            side=pos.side,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=timestamp,
            exit_price=adjusted_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            metadata=pos.metadata
        )
        self._trades.append(trade)

        logger.debug(f"Exit: {pos.side.value} @ {adjusted_price:.2f}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")

        self._position = None

    def optimize(self, strategy_class: type, data: pd.DataFrame,
                 param_grid: dict, metric: str = "sharpe_ratio") -> dict:
        """Optimize strategy parameters.

        Args:
            strategy_class: Strategy class to optimize
            data: Historical data
            param_grid: Dict of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')

        Returns:
            Best parameters dict
        """
        import itertools

        best_params = None
        best_metric = float('-inf')

        # Generate all parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))

            try:
                strategy = strategy_class(**params)
                result = self.run(strategy, data)

                metric_value = getattr(result.metrics, metric, 0)

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()
                    best_params['_metric_value'] = metric_value

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")
                continue

        logger.info(f"Best params: {best_params}")
        return best_params
