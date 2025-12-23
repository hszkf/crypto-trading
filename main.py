#!/usr/bin/env python3
"""
Crypto Trading System - Example Usage

This script demonstrates how to use the trading system components:
1. Technical indicators
2. Trading strategies
3. Signal generation
4. Backtesting
5. Live trading (paper mode)
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.indicators import RSI, EMA, MACD, BollingerBands, ATR, VWAP
from src.strategies import (
    EMACrossoverStrategy,
    BollingerSqueezeStrategy,
    RSIDivergenceStrategy,
    VWAPBounceStrategy,
)
from src.signals import SignalGenerator, SignalManager
from src.backtesting import BacktestEngine, BacktestReport
from config.settings import Settings, TradingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def generate_sample_data(n_bars: int = 500, trend: str = "random") -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration.

    Args:
        n_bars: Number of bars to generate
        trend: "up", "down", or "random"

    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1h")

    # Generate price movement based on trend
    if trend == "up":
        drift = 0.0002
    elif trend == "down":
        drift = -0.0002
    else:
        drift = 0.0

    returns = np.random.normal(drift, 0.015, n_bars)
    close = 50000 * np.exp(np.cumsum(returns))  # Start at ~50000 (BTC-like)

    # Generate OHLC
    high = close * (1 + np.abs(np.random.normal(0, 0.008, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.uniform(100, 1000, n_bars) * (close / 50000)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }, index=dates)


def demo_indicators():
    """Demonstrate technical indicator usage."""
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS DEMO")
    print("="*60)

    data = generate_sample_data(500)

    # Trend indicators
    ema_fast = EMA(9)
    ema_slow = EMA(21)
    macd = MACD(12, 26, 9)

    ema_fast_values = ema_fast.calculate(data)
    ema_slow_values = ema_slow.calculate(data)
    macd_line, signal_line, histogram = macd.calculate_all(data)

    print(f"\nLatest EMA(9):  ${ema_fast_values.iloc[-1]:,.2f}")
    print(f"Latest EMA(21): ${ema_slow_values.iloc[-1]:,.2f}")
    print(f"EMA Trend: {'Bullish' if ema_fast_values.iloc[-1] > ema_slow_values.iloc[-1] else 'Bearish'}")

    print(f"\nMACD Line:   {macd_line.iloc[-1]:.2f}")
    print(f"Signal Line: {signal_line.iloc[-1]:.2f}")
    print(f"Histogram:   {histogram.iloc[-1]:.2f}")

    # Momentum indicators
    rsi = RSI(14)
    rsi_values = rsi.calculate(data)

    print(f"\nRSI(14): {rsi_values.iloc[-1]:.2f}")
    if rsi_values.iloc[-1] > 70:
        print("Status: OVERBOUGHT - Consider taking profits")
    elif rsi_values.iloc[-1] < 30:
        print("Status: OVERSOLD - Watch for reversal")
    else:
        print("Status: Neutral")

    # Volatility indicators
    bb = BollingerBands(20, 2.0)
    atr = ATR(14)

    bb_upper, bb_middle, bb_lower = bb.calculate_all(data)
    atr_value = atr.calculate(data).iloc[-1]

    print(f"\nBollinger Bands:")
    print(f"  Upper:  ${bb_upper.iloc[-1]:,.2f}")
    print(f"  Middle: ${bb_middle.iloc[-1]:,.2f}")
    print(f"  Lower:  ${bb_lower.iloc[-1]:,.2f}")
    print(f"  Price:  ${data['close'].iloc[-1]:,.2f}")

    print(f"\nATR(14): ${atr_value:,.2f} ({atr_value/data['close'].iloc[-1]*100:.2f}%)")

    # Volume indicators
    vwap = VWAP()
    vwap_value = vwap.calculate(data).iloc[-1]
    print(f"\nVWAP: ${vwap_value:,.2f}")
    print(f"Price vs VWAP: {'Above' if data['close'].iloc[-1] > vwap_value else 'Below'}")


def demo_strategies():
    """Demonstrate strategy evaluation."""
    print("\n" + "="*60)
    print("TRADING STRATEGIES DEMO")
    print("="*60)

    data = generate_sample_data(500, trend="up")

    strategies = [
        EMACrossoverStrategy("BTC/USDT", "1h"),
        BollingerSqueezeStrategy("BTC/USDT", "1h"),
        RSIDivergenceStrategy("BTC/USDT", "1h"),
        VWAPBounceStrategy("BTC/USDT", "15m"),
    ]

    for strategy in strategies:
        print(f"\n--- {strategy.name} ---")
        result = strategy.evaluate(data)

        long_signals = [s for s in result.signals if s.side.value == "long"]
        short_signals = [s for s in result.signals if s.side.value == "short"]

        print(f"Total Signals: {len(result.signals)}")
        print(f"  Long:  {len(long_signals)}")
        print(f"  Short: {len(short_signals)}")

        # Show latest signal if exists
        if result.signals:
            latest = result.signals[-1]
            print(f"\nLatest Signal:")
            print(f"  Side: {latest.side.value.upper()}")
            print(f"  Price: ${latest.price:,.2f}")
            if latest.stop_loss:
                print(f"  Stop Loss: ${latest.stop_loss:,.2f}")
            if latest.take_profit:
                print(f"  Take Profit: ${latest.take_profit:,.2f}")
            if latest.risk_reward_ratio:
                print(f"  Risk/Reward: {latest.risk_reward_ratio:.2f}")


def demo_signal_generator():
    """Demonstrate multi-strategy signal generation with confluence."""
    print("\n" + "="*60)
    print("SIGNAL GENERATOR DEMO (Confluence)")
    print("="*60)

    data = generate_sample_data(500)

    # Create multiple strategies
    strategies = [
        EMACrossoverStrategy("BTC/USDT", "1h"),
        BollingerSqueezeStrategy("BTC/USDT", "1h"),
        RSIDivergenceStrategy("BTC/USDT", "1h"),
    ]

    # Test with different confluence levels
    for confluence in [1, 2, 3]:
        generator = SignalGenerator(strategies, min_confluence=confluence)
        signals = generator.generate(data)

        print(f"\nConfluence >= {confluence}: {len(signals)} signals")

        if signals:
            signal = signals[0]
            print(f"  First Signal: {signal.side.value.upper()} @ ${signal.price:,.2f}")
            if "strategies" in signal.metadata:
                print(f"  Agreeing Strategies: {signal.metadata['strategies']}")


def demo_backtest():
    """Demonstrate backtesting functionality."""
    print("\n" + "="*60)
    print("BACKTESTING DEMO")
    print("="*60)

    # Generate data for different market conditions
    scenarios = {
        "Uptrend": generate_sample_data(500, trend="up"),
        "Downtrend": generate_sample_data(500, trend="down"),
        "Sideways": generate_sample_data(500, trend="random"),
    }

    strategy = EMACrossoverStrategy("BTC/USDT", "1h")
    engine = BacktestEngine(
        initial_capital=10000,
        commission_pct=0.1,
        slippage_pct=0.05,
        risk_per_trade=0.02
    )

    print(f"\nStrategy: {strategy.name}")
    print(f"Initial Capital: $10,000")
    print(f"Risk per Trade: 2%")
    print(f"Commission: 0.1%")

    for scenario_name, data in scenarios.items():
        print(f"\n--- {scenario_name} Market ---")
        result = engine.run(strategy, data)
        m = result.metrics

        print(f"Final Capital: ${result.final_capital:,.2f}")
        print(f"Total Return: {m.total_return_pct:+.2f}%")
        print(f"Total Trades: {m.total_trades}")
        print(f"Win Rate: {m.win_rate:.1f}%")
        print(f"Profit Factor: {m.profit_factor:.2f}")
        print(f"Sharpe Ratio: {m.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {m.max_drawdown:.2f}%")


def demo_full_report():
    """Generate a complete backtest report."""
    print("\n" + "="*60)
    print("FULL BACKTEST REPORT")
    print("="*60)

    data = generate_sample_data(800, trend="up")
    strategy = EMACrossoverStrategy("BTC/USDT", "1h")

    engine = BacktestEngine(
        initial_capital=10000,
        commission_pct=0.1,
        slippage_pct=0.05
    )

    result = engine.run(strategy, data)
    report = BacktestReport(result)

    # Print summary
    print(report.summary())

    # Win/loss streaks
    streaks = report.win_loss_streaks()
    print(f"Max Winning Streak: {streaks['max_win_streak']}")
    print(f"Max Losing Streak: {streaks['max_loss_streak']}")

    # Trade log sample
    trade_log = report.trade_log()
    if not trade_log.empty:
        print("\nLast 5 Trades:")
        print(trade_log.tail()[["entry_time", "side", "entry_price", "exit_price", "pnl", "exit_reason"]].to_string())


def demo_strategy_comparison():
    """Compare multiple strategies on same data."""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)

    data = generate_sample_data(600)

    strategies = [
        EMACrossoverStrategy("BTC/USDT", "1h"),
        BollingerSqueezeStrategy("BTC/USDT", "1h"),
        RSIDivergenceStrategy("BTC/USDT", "1h"),
    ]

    engine = BacktestEngine(initial_capital=10000)
    results = []

    print(f"\n{'Strategy':<25} {'Return':>10} {'Trades':>8} {'Win Rate':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 75)

    for strategy in strategies:
        result = engine.run(strategy, data)
        m = result.metrics

        print(f"{strategy.name:<25} {m.total_return_pct:>+9.2f}% {m.total_trades:>8} "
              f"{m.win_rate:>9.1f}% {m.sharpe_ratio:>8.2f} {m.max_drawdown:>7.2f}%")

        results.append((strategy.name, result))

    # Find best strategy by Sharpe ratio
    best = max(results, key=lambda x: x[1].metrics.sharpe_ratio)
    print(f"\nBest Risk-Adjusted Strategy: {best[0]}")


async def demo_live_trading():
    """Demonstrate live trading setup (paper mode)."""
    print("\n" + "="*60)
    print("LIVE TRADING DEMO (Paper Mode)")
    print("="*60)

    settings = Settings.from_env()

    print(f"\nTrading Mode: {settings.mode.value}")
    print(f"Exchange: {settings.exchange.exchange_id}")
    print(f"Testnet: {settings.exchange.testnet}")

    print("\nRisk Settings:")
    print(f"  Max Position Size: {settings.risk.max_position_size_pct*100}%")
    print(f"  Risk per Trade: {settings.risk.risk_per_trade_pct*100}%")
    print(f"  Max Daily Loss: {settings.risk.max_daily_loss_pct*100}%")
    print(f"  Max Drawdown: {settings.risk.max_drawdown_pct*100}%")

    # Simulate signal generation cycle
    print("\n--- Signal Generation Cycle ---")

    data = generate_sample_data(500)

    strategies = [
        EMACrossoverStrategy("BTC/USDT", "1h"),
        BollingerSqueezeStrategy("BTC/USDT", "1h"),
    ]

    generator = SignalGenerator(strategies, min_confluence=1)
    manager = SignalManager(signal_expiry_minutes=60)

    # Generate signals
    signals = generator.generate(data)

    if signals:
        for signal in signals[:3]:  # Show first 3
            managed = manager.add_signal(signal)
            print(f"\nNew Signal Queued:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Side: {signal.side.value.upper()}")
            print(f"  Price: ${signal.price:,.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:,.2f}" if signal.stop_loss else "")
            print(f"  Take Profit: ${signal.take_profit:,.2f}" if signal.take_profit else "")
            print(f"  Expires: {managed.expiry}")
    else:
        print("\nNo signals generated in current cycle")

    print(f"\nPending Signals: {manager.pending_count}")

    # Note about live trading
    print("\n" + "-"*40)
    print("NOTE: To enable live trading:")
    print("1. Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env")
    print("2. Set TRADING_MODE=live (use paper first!)")
    print("3. Set EXCHANGE_TESTNET=false for real trading")
    print("-"*40)


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("#" + " "*18 + "CRYPTO TRADING SYSTEM" + " "*17 + "#")
    print("#" + " "*14 + "Technical Analysis Demo" + " "*19 + "#")
    print("#"*60)

    # Run demos
    demo_indicators()
    demo_strategies()
    demo_signal_generator()
    demo_backtest()
    demo_full_report()
    demo_strategy_comparison()

    # Run async demo
    asyncio.run(demo_live_trading())

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Run tests: pytest tests/ -v")
    print("2. Configure .env with your exchange API keys")
    print("3. Start with paper trading mode")
    print("4. Backtest strategies on historical data")
    print("5. Monitor and refine your strategies")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
