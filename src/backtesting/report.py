"""Backtest reporting and visualization."""

import json

import pandas as pd

from .engine import BacktestResult


class BacktestReport:
    """Generate backtest reports and summaries."""

    def __init__(self, result: BacktestResult):
        self.result = result

    def summary(self) -> str:
        """Generate text summary of backtest results."""
        r = self.result
        m = r.metrics

        return f"""
{"=" * 60}
BACKTEST REPORT: {r.strategy_name}
{"=" * 60}

Period: {r.start_date.strftime("%Y-%m-%d")} to {r.end_date.strftime("%Y-%m-%d")}
Symbol: {r.symbol}
Timeframe: {r.timeframe}

RETURNS
-------
Initial Capital:  ${r.initial_capital:,.2f}
Final Capital:    ${r.final_capital:,.2f}
Total Return:     ${m.total_return:,.2f} ({m.total_return_pct:.2f}%)
Annualized:       {m.annualized_return:.2f}%

RISK METRICS
------------
Sharpe Ratio:     {m.sharpe_ratio:.2f}
Sortino Ratio:    {m.sortino_ratio:.2f}
Max Drawdown:     {m.max_drawdown:.2f}%
Calmar Ratio:     {m.calmar_ratio:.2f}

TRADE STATISTICS
----------------
Total Trades:     {m.total_trades}
Win Rate:         {m.win_rate:.1f}%
Profit Factor:    {m.profit_factor:.2f}
Avg Win:          ${m.average_win:,.2f}
Avg Loss:         ${m.average_loss:,.2f}
Largest Win:      ${m.largest_win:,.2f}
Largest Loss:     ${m.largest_loss:,.2f}
Avg Trade:        ${m.average_trade:,.2f}

RISK/REWARD
-----------
Expectancy:       ${m.expectancy:,.2f}
Expectancy Ratio: {m.expectancy_ratio:.2f}R
Risk/Reward:      {m.risk_reward_ratio:.2f}

{"=" * 60}
"""

    def trade_log(self) -> pd.DataFrame:
        """Generate DataFrame of all trades."""
        if not self.result.trades:
            return pd.DataFrame()

        data = []
        for t in self.result.trades:
            data.append(
                {
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "side": t.side.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "duration_hrs": t.duration,
                    "exit_reason": t.exit_reason,
                }
            )

        return pd.DataFrame(data)

    def monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns table."""
        equity = self.result.equity_curve
        if equity.empty:
            return pd.DataFrame()

        # Resample to monthly
        monthly = equity.resample("M").last()
        monthly_returns = monthly.pct_change() * 100

        # Pivot to year x month format
        df = pd.DataFrame(
            {
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month,
                "return": monthly_returns.values,
            }
        )

        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(pivot.columns)]

        return pivot

    def to_json(self) -> str:
        """Export results to JSON."""
        r = self.result
        m = r.metrics

        data = {
            "strategy": r.strategy_name,
            "symbol": r.symbol,
            "timeframe": r.timeframe,
            "period": {"start": r.start_date.isoformat(), "end": r.end_date.isoformat()},
            "capital": {"initial": r.initial_capital, "final": r.final_capital},
            "metrics": {
                "total_return": m.total_return,
                "total_return_pct": m.total_return_pct,
                "annualized_return": m.annualized_return,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "max_drawdown": m.max_drawdown,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
                "expectancy": m.expectancy,
                "risk_reward_ratio": m.risk_reward_ratio,
            },
            "trades": [
                {
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "side": t.side.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "exit_reason": t.exit_reason,
                }
                for t in r.trades
            ],
        }

        return json.dumps(data, indent=2)

    def to_csv(self, filepath: str) -> None:
        """Export trade log to CSV."""
        self.trade_log().to_csv(filepath, index=False)

    def drawdown_analysis(self) -> pd.DataFrame:
        """Analyze drawdown periods."""
        equity = self.result.equity_curve
        if equity.empty:
            return pd.DataFrame()

        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax * 100

        # Find drawdown periods
        in_dd = drawdown < 0
        dd_start = in_dd & ~in_dd.shift(1).fillna(False)
        dd_end = ~in_dd & in_dd.shift(1).fillna(False)

        periods = []
        current_start = None

        for idx in equity.index:
            if dd_start.get(idx, False):
                current_start = idx
            if dd_end.get(idx, False) and current_start:
                period_dd = drawdown[current_start:idx]
                periods.append(
                    {
                        "start": current_start,
                        "end": idx,
                        "max_dd": period_dd.min(),
                        "duration_days": (idx - current_start).days,
                    }
                )
                current_start = None

        return pd.DataFrame(periods).sort_values("max_dd")

    def win_loss_streaks(self) -> dict:
        """Calculate winning and losing streaks."""
        if not self.result.trades:
            return {"max_win_streak": 0, "max_loss_streak": 0}

        wins = [1 if t.pnl > 0 else 0 for t in self.result.trades]

        max_win = 0
        max_loss = 0
        current_win = 0
        current_loss = 0

        for w in wins:
            if w:
                current_win += 1
                current_loss = 0
                max_win = max(max_win, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss = max(max_loss, current_loss)

        return {
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "current_streak": current_win if wins[-1] else -current_loss,
        }
