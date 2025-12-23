"""Performance metrics calculation."""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in bars

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade: float

    # Expectancy
    expectancy: float
    expectancy_ratio: float

    # Time metrics
    average_trade_duration: float  # hours
    average_bars_in_trade: float

    # Risk-adjusted
    calmar_ratio: float
    risk_reward_ratio: float


def calculate_metrics(trades: list, equity_curve: pd.Series,
                      initial_capital: float,
                      risk_free_rate: float = 0.02) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics.

    Args:
        trades: List of Trade objects
        equity_curve: Series of equity values over time
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PerformanceMetrics dataclass
    """
    if not trades:
        # Still calculate equity-based metrics if equity curve provided
        if len(equity_curve) > 1:
            return _calculate_equity_only_metrics(equity_curve, initial_capital, risk_free_rate)
        return _empty_metrics()

    # Basic trade stats
    pnls = [t.pnl for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]

    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]

    total_trades = len(trades)
    winning_trades = len(winning)
    losing_trades = len(losing)

    # Win rate
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Average win/loss
    avg_win = np.mean(winning) if winning else 0
    avg_loss = abs(np.mean(losing)) if losing else 0

    # Profit factor
    gross_profit = sum(winning) if winning else 0
    gross_loss = abs(sum(losing)) if losing else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Returns
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    # Annualized return (assuming daily data)
    days = len(equity_curve)
    years = days / 252 if days > 0 else 1
    annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Returns series for risk calculations
    returns = equity_curve.pct_change().dropna()

    # Sharpe ratio (annualized)
    if len(returns) > 1 and returns.std() > 0:
        excess_returns = returns.mean() - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino_ratio = (returns.mean() * 252 - risk_free_rate) / (downside_returns.std() * np.sqrt(252))
    else:
        sortino_ratio = 0

    # Maximum drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdown.min()) * 100

    # Max drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_lengths = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Calmar ratio
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    # Expectancy
    expectancy = np.mean(pnls) if pnls else 0

    # Expectancy ratio (R-multiple)
    if avg_loss > 0:
        expectancy_ratio = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
    else:
        expectancy_ratio = 0

    # Trade durations
    durations = [t.duration for t in trades]
    avg_duration = np.mean(durations) if durations else 0

    # Risk/reward from actual trades
    if avg_loss > 0:
        risk_reward = avg_win / avg_loss
    else:
        risk_reward = 0

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate * 100,
        profit_factor=profit_factor,
        average_win=avg_win,
        average_loss=avg_loss,
        largest_win=max(pnls) if pnls else 0,
        largest_loss=min(pnls) if pnls else 0,
        average_trade=np.mean(pnls) if pnls else 0,
        expectancy=expectancy,
        expectancy_ratio=expectancy_ratio,
        average_trade_duration=avg_duration,
        average_bars_in_trade=len(equity_curve) / total_trades if total_trades > 0 else 0,
        calmar_ratio=calmar_ratio,
        risk_reward_ratio=risk_reward
    )


def _calculate_equity_only_metrics(equity_curve: pd.Series, initial_capital: float,
                                    risk_free_rate: float = 0.02) -> PerformanceMetrics:
    """Calculate metrics from equity curve only (no trades)."""
    final_capital = equity_curve.iloc[-1]
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    days = len(equity_curve)
    years = days / 252 if days > 0 else 1
    annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    returns = equity_curve.pct_change().dropna()

    # Sharpe ratio
    if len(returns) > 1 and returns.std() > 0:
        excess_returns = returns.mean() - (risk_free_rate / 252)
        sharpe_ratio = (excess_returns / returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # Sortino ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and downside_returns.std() > 0:
        sortino_ratio = (returns.mean() * 252 - risk_free_rate) / (downside_returns.std() * np.sqrt(252))
    else:
        sortino_ratio = 0

    # Maximum drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdown.min()) * 100

    # Max drawdown duration
    in_drawdown = drawdown < 0
    dd_groups = (~in_drawdown).cumsum()
    dd_lengths = in_drawdown.groupby(dd_groups).sum()
    max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

    return PerformanceMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_dd_duration,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0,
        profit_factor=0,
        average_win=0,
        average_loss=0,
        largest_win=0,
        largest_loss=0,
        average_trade=0,
        expectancy=0,
        expectancy_ratio=0,
        average_trade_duration=0,
        average_bars_in_trade=0,
        calmar_ratio=calmar_ratio,
        risk_reward_ratio=0
    )


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics for no-trade scenarios."""
    return PerformanceMetrics(
        total_return=0, total_return_pct=0, annualized_return=0,
        sharpe_ratio=0, sortino_ratio=0, max_drawdown=0, max_drawdown_duration=0,
        total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
        profit_factor=0, average_win=0, average_loss=0, largest_win=0,
        largest_loss=0, average_trade=0, expectancy=0, expectancy_ratio=0,
        average_trade_duration=0, average_bars_in_trade=0,
        calmar_ratio=0, risk_reward_ratio=0
    )
