# Crypto Trading Project

## Overview
This project implements cryptocurrency trading strategies using expert technical analysis indicators and patterns.

## Tech Stack
- Python 3.11+
- pandas, numpy for data manipulation
- ta-lib or pandas-ta for technical indicators
- ccxt for exchange connectivity
- websockets for real-time data

## Architecture
- `/src/indicators/` - Technical analysis indicators
- `/src/strategies/` - Trading strategy implementations
- `/src/exchange/` - Exchange API integrations
- `/src/signals/` - Signal generation and management
- `/src/backtesting/` - Strategy backtesting framework
- `/tests/` - Unit and integration tests

## Key Concepts

### Technical Indicators
- **Trend**: Moving Averages (SMA, EMA, WMA), MACD, ADX, Parabolic SAR
- **Momentum**: RSI, Stochastic, CCI, Williams %R, ROC
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, VWAP, MFI, Accumulation/Distribution

### Timeframes
- Scalping: 1m, 5m, 15m
- Day trading: 15m, 1h, 4h
- Swing trading: 4h, 1D, 1W

### Risk Management
- Position sizing based on account percentage (1-2% risk per trade)
- Stop-loss and take-profit levels
- Risk-reward ratio minimum 1:2

## Code Standards
- Type hints on all functions
- Docstrings for public methods
- Unit tests for indicators and strategies
- Logging for all trade signals and executions

## Environment Variables
```
EXCHANGE_API_KEY=
EXCHANGE_API_SECRET=
TRADING_MODE=paper|live
LOG_LEVEL=INFO
```

## Important Notes
- Never commit API keys or secrets
- Always backtest strategies before live trading
- Use paper trading mode for testing
- Implement proper error handling for exchange API calls
