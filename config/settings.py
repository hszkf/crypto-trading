"""Application settings and configuration."""

import os
from dataclasses import dataclass, field
from enum import Enum


class TradingMode(Enum):
    """Trading mode."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    exchange_id: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    sandbox: bool = True

    @classmethod
    def from_env(cls) -> "ExchangeConfig":
        """Load config from environment variables."""
        return cls(
            exchange_id=os.getenv("EXCHANGE_ID", "binance"),
            api_key=os.getenv("EXCHANGE_API_KEY", ""),
            api_secret=os.getenv("EXCHANGE_API_SECRET", ""),
            testnet=os.getenv("EXCHANGE_TESTNET", "true").lower() == "true",
            sandbox=os.getenv("EXCHANGE_SANDBOX", "true").lower() == "true",
        )


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_pct: float = 0.1      # 10% of account per position
    risk_per_trade_pct: float = 0.02        # 2% risk per trade
    max_daily_loss_pct: float = 0.05        # 5% max daily loss
    max_drawdown_pct: float = 0.15          # 15% max drawdown
    min_risk_reward: float = 2.0            # Minimum 1:2 R:R
    max_open_positions: int = 3             # Maximum concurrent positions
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.02

    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Load config from environment variables."""
        return cls(
            max_position_size_pct=float(os.getenv("MAX_POSITION_SIZE_PCT", "0.1")),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.02")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.15")),
        )


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframes: list[str] = field(default_factory=lambda: ["1h"])
    min_confluence: int = 2
    signal_expiry_minutes: int = 60

    # EMA Crossover params
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 200

    # Bollinger params
    bb_period: int = 20
    bb_std: float = 2.0

    # RSI params
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30


@dataclass
class Settings:
    """Main application settings."""
    mode: TradingMode = TradingMode.PAPER
    log_level: str = "INFO"
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load all settings from environment."""
        mode_str = os.getenv("TRADING_MODE", "paper").lower()
        mode = TradingMode(mode_str) if mode_str in ["paper", "live", "backtest"] else TradingMode.PAPER

        return cls(
            mode=mode,
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            exchange=ExchangeConfig.from_env(),
            risk=RiskConfig.from_env(),
            strategy=StrategyConfig(),
        )


# Global settings instance
settings = Settings.from_env()
