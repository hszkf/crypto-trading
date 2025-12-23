# Crypto Trading System

A cryptocurrency trading system built with Python, featuring technical analysis indicators, multiple trading strategies, backtesting capabilities, and exchange integration.

## Features

- **Technical Indicators**: SMA, EMA, MACD, ADX, RSI, Stochastic RSI, CCI, Williams %R, Bollinger Bands, ATR, Keltner Channels, OBV, VWAP, MFI
- **Trading Strategies**: EMA Crossover, Bollinger Squeeze Breakout, RSI Divergence, VWAP Bounce
- **Signal Generation**: Multi-strategy confluence filtering with configurable thresholds
- **Backtesting Engine**: Event-driven backtester with realistic commission and slippage
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, profit factor, win rate
- **Exchange Integration**: CCXT-based client supporting Binance, Bybit, OKX, and more

## Project Structure

```
trading/
├── src/
│   ├── indicators/      # Technical analysis indicators
│   ├── strategies/      # Trading strategy implementations
│   ├── exchange/        # Exchange API integration
│   ├── signals/         # Signal generation and management
│   └── backtesting/     # Backtesting framework
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── config/              # Configuration management
├── skills/              # Claude Code skill definitions
├── main.py              # Example usage and demos
└── pyproject.toml       # Project configuration
```

## Requirements

- Python 3.11+
- pandas
- numpy
- ccxt (for live trading)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd trading
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install pandas numpy

   # For live trading
   pip install ccxt

   # For development
   pip install pytest pytest-cov ruff mypy
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your exchange API keys
   ```

## Quick Start

### Run the Demo

```bash
python main.py
```

This runs demonstrations of:
- Technical indicator calculations
- Strategy signal generation
- Backtesting with performance reports
- Strategy comparison

### Run Tests

```bash
pytest tests/ -v
```

## Usage Examples

### Technical Indicators

```python
import pandas as pd
from src.indicators import RSI, EMA, BollingerBands, MACD

# Load your OHLCV data
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Calculate indicators
rsi = RSI(period=14)
rsi_values = rsi.calculate(data)

ema = EMA(period=20)
ema_values = ema.calculate(data)

bb = BollingerBands(period=20, std_dev=2.0)
upper, middle, lower = bb.calculate_all(data)

macd = MACD(fast=12, slow=26, signal=9)
macd_line, signal_line, histogram = macd.calculate_all(data)
```

### Trading Strategies

```python
from src.strategies import EMACrossoverStrategy, BollingerSqueezeStrategy

# Create a strategy
strategy = EMACrossoverStrategy("BTC/USDT", timeframe="1h")

# Evaluate on historical data
result = strategy.evaluate(data)

print(f"Signals generated: {len(result.signals)}")
for signal in result.signals:
    print(f"{signal.side.value} @ ${signal.price:,.2f}")
    print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
    print(f"  Take Profit: ${signal.take_profit:,.2f}")
```

### Backtesting

```python
from src.backtesting import BacktestEngine, BacktestReport
from src.strategies import EMACrossoverStrategy

# Create engine and strategy
engine = BacktestEngine(
    initial_capital=10000,
    commission_pct=0.1,    # 0.1%
    slippage_pct=0.05,     # 0.05%
    risk_per_trade=0.02    # 2% risk per trade
)
strategy = EMACrossoverStrategy("BTC/USDT")

# Run backtest
result = engine.run(strategy, data)

# Generate report
report = BacktestReport(result)
print(report.summary())

# Access metrics
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown:.2f}%")
print(f"Win Rate: {result.metrics.win_rate:.1f}%")
```

### Signal Generation with Confluence

```python
from src.signals import SignalGenerator
from src.strategies import (
    EMACrossoverStrategy,
    BollingerSqueezeStrategy,
    RSIDivergenceStrategy
)

# Create multiple strategies
strategies = [
    EMACrossoverStrategy("BTC/USDT"),
    BollingerSqueezeStrategy("BTC/USDT"),
    RSIDivergenceStrategy("BTC/USDT"),
]

# Generate signals with confluence (require 2+ strategies to agree)
generator = SignalGenerator(strategies, min_confluence=2)
signals = generator.generate(data)
```

### Live Trading Setup

```python
import asyncio
from src.exchange import ExchangeClient

async def main():
    # Connect to exchange
    async with ExchangeClient(
        exchange_id="binance",
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True  # Use testnet first!
    ) as client:
        # Get account balance
        balance = await client.get_balance("USDT")
        print(f"Balance: ${balance.available:,.2f}")

        # Get market data
        ohlcv = await client.get_ohlcv("BTC/USDT", "1h", limit=500)

        # Get current price
        ticker = await client.get_ticker("BTC/USDT")
        print(f"BTC Price: ${ticker['last']:,.2f}")

asyncio.run(main())
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXCHANGE_ID` | Exchange identifier | `binance` |
| `EXCHANGE_API_KEY` | API key | - |
| `EXCHANGE_API_SECRET` | API secret | - |
| `EXCHANGE_TESTNET` | Use testnet | `true` |
| `TRADING_MODE` | `paper`, `live`, `backtest` | `paper` |
| `RISK_PER_TRADE_PCT` | Risk per trade | `0.02` |
| `MAX_POSITION_SIZE_PCT` | Max position size | `0.1` |
| `MAX_DAILY_LOSS_PCT` | Max daily loss | `0.05` |
| `MAX_DRAWDOWN_PCT` | Max drawdown | `0.15` |

### Risk Management

The system enforces risk management rules:
- **Position Sizing**: Based on ATR stop-loss distance and risk percentage
- **Max Position**: Capped at 25% of capital per trade
- **Stop Loss**: ATR-based dynamic stops
- **Take Profit**: Configurable risk/reward ratio (default 1:2)

## Trading Strategies

### EMA Crossover
- **Entry Long**: Fast EMA crosses above slow EMA, RSI 50-70, price above 200 EMA
- **Entry Short**: Fast EMA crosses below slow EMA, RSI 30-50, price below 200 EMA
- **Exit**: Opposite crossover or RSI extremes

### Bollinger Squeeze Breakout
- **Setup**: Bollinger Bands contract inside Keltner Channels (low volatility)
- **Entry**: Price closes outside bands with volume spike after squeeze
- **Exit**: Price returns to middle band

### RSI Divergence
- **Bullish**: Price lower low + RSI higher low, enter on RSI crossing above 30
- **Bearish**: Price higher high + RSI lower high, enter on RSI crossing below 70
- **Exit**: RSI reaches opposite extreme

### VWAP Bounce
- **Entry Long**: Price pulls back to VWAP from above with bullish candle
- **Entry Short**: Price rallies to VWAP from below with bearish candle
- **Exit**: Price moves significantly away from VWAP

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_indicators.py -v
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/
```

## Disclaimer

This software is for educational purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always test strategies thoroughly on paper trading before using real funds.

## License

MIT
