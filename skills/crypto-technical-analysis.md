# Crypto Technical Analysis Trading Skill

## Description
Expert-level technical analysis for cryptocurrency trading. This skill provides analysis, signals, and strategy recommendations based on proven technical indicators and chart patterns.

## Capabilities

### 1. Trend Analysis
Identify market direction using multiple indicators:
- **Moving Average Crossovers**: Golden cross (50 SMA > 200 SMA) for bullish, death cross for bearish
- **EMA Ribbons**: 8, 13, 21, 55 EMA alignment for trend strength
- **MACD**: Signal line crossovers and histogram divergence
- **ADX**: Trend strength measurement (>25 = strong trend)

### 2. Support & Resistance
- Identify key price levels from historical pivots
- Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
- Fibonacci extensions for profit targets
- Volume profile analysis for high-volume nodes

### 3. Momentum Indicators
- **RSI (14)**: Overbought >70, oversold <30, divergences
- **Stochastic RSI**: Fast momentum shifts
- **Williams %R**: Overbought/oversold confirmation
- **CCI**: Trend strength and reversals

### 4. Volatility Analysis
- **Bollinger Bands**: Squeeze detection, band walks, mean reversion
- **ATR**: Position sizing and stop-loss placement
- **Keltner Channels**: Breakout confirmation with BB

### 5. Volume Analysis
- **OBV**: Accumulation/distribution trends
- **VWAP**: Institutional price levels
- **Volume Delta**: Buying vs selling pressure
- **CVD**: Cumulative volume divergences

### 6. Chart Patterns
**Reversal Patterns:**
- Head and Shoulders / Inverse H&S
- Double Top / Double Bottom
- Triple Top / Triple Bottom
- Rounding Bottom / Top

**Continuation Patterns:**
- Bull/Bear Flags
- Ascending/Descending Triangles
- Symmetrical Triangles
- Wedges (Rising/Falling)

### 7. Candlestick Patterns
**Bullish:**
- Hammer, Inverted Hammer
- Morning Star, Three White Soldiers
- Bullish Engulfing, Piercing Line

**Bearish:**
- Shooting Star, Hanging Man
- Evening Star, Three Black Crows
- Bearish Engulfing, Dark Cloud Cover

**Indecision:**
- Doji variations
- Spinning Top

## Trading Strategies

### Strategy 1: EMA Crossover with RSI Filter
```
Entry Long:
- 9 EMA crosses above 21 EMA
- RSI > 50 and < 70
- Price above 200 EMA

Entry Short:
- 9 EMA crosses below 21 EMA
- RSI < 50 and > 30
- Price below 200 EMA

Stop Loss: Below recent swing low (long) or high (short)
Take Profit: 2x ATR or next resistance/support
```

### Strategy 2: Bollinger Band Squeeze Breakout
```
Setup:
- BB width at 6-month low (squeeze)
- Wait for candle close outside bands

Entry Long: Close above upper band with volume spike
Entry Short: Close below lower band with volume spike

Stop Loss: Middle band (20 SMA)
Take Profit: 2x band width from entry
```

### Strategy 3: RSI Divergence
```
Bullish Divergence:
- Price makes lower low
- RSI makes higher low
- Enter on RSI crossing above 30

Bearish Divergence:
- Price makes higher high
- RSI makes lower high
- Enter on RSI crossing below 70

Stop Loss: Beyond the divergence swing point
Take Profit: Previous swing high/low
```

### Strategy 4: VWAP Bounce (Intraday)
```
Entry Long:
- Price pulls back to VWAP
- Bullish candle pattern at VWAP
- Volume confirmation

Entry Short:
- Price rallies to VWAP from below
- Bearish candle pattern at VWAP
- Volume confirmation

Stop Loss: 1 ATR beyond VWAP
Take Profit: Previous high/low or 2:1 R:R
```

## Risk Management Rules

### Position Sizing
```
Position Size = (Account Risk %) / (Entry - Stop Loss %)

Example:
- Account: $10,000
- Risk per trade: 2% = $200
- Entry: $100, Stop Loss: $95 (5% risk)
- Position Size: $200 / 5% = $4,000
```

### Stop Loss Guidelines
- Never risk more than 2% per trade
- Use ATR-based stops (1.5-2x ATR)
- Place stops beyond key levels, not at round numbers
- Account for exchange spread and slippage

### Take Profit Guidelines
- Minimum 1:2 risk-reward ratio
- Scale out: 50% at 1:2, 50% at 1:3
- Trail stops using ATR or moving averages
- Consider partial profits at Fibonacci extensions

## Multi-Timeframe Analysis

### Top-Down Approach
1. **Weekly**: Identify major trend and key levels
2. **Daily**: Confirm trend and find trading range
3. **4H**: Identify entry zones
4. **1H/15m**: Time precise entries

### Timeframe Alignment
- Only trade in direction of higher timeframe trend
- Use lower timeframe for entry optimization
- Conflicting signals = no trade

## Crypto-Specific Considerations

### Market Structure
- 24/7 markets - no opening gaps
- High correlation with BTC (check BTC.D)
- Funding rates for perpetual futures
- Liquidation cascades and wicks

### Best Trading Times
- US market open (9:30 AM EST)
- London/US overlap (8 AM - 12 PM EST)
- Asian session for altcoin moves

### Altcoin Analysis
- Check BTC pair, not just USD pair
- Monitor ETH/BTC for alt season signals
- Lower liquidity = wider stops needed
- News sensitivity is higher

## Signal Format

```
SIGNAL: [LONG/SHORT] [SYMBOL]
Timeframe: [TF]
Entry: [PRICE]
Stop Loss: [PRICE] ([%])
Take Profit 1: [PRICE] ([%]) - 50%
Take Profit 2: [PRICE] ([%]) - 50%
Risk/Reward: [RATIO]

Reasoning:
- [Indicator 1 confluence]
- [Indicator 2 confluence]
- [Pattern/Level]

Invalidation: [CONDITION]
```

## Checklist Before Trade

- [ ] Higher timeframe trend identified
- [ ] Multiple indicator confluence (3+)
- [ ] Key support/resistance levels marked
- [ ] Risk calculated and acceptable
- [ ] Stop loss and take profit set
- [ ] Position size appropriate
- [ ] No major news events pending
- [ ] BTC/ETH structure supportive
