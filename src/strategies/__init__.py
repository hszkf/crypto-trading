"""Trading strategies module."""

from .base import Strategy, StrategyResult, Signal, Side, SignalType
from .ema_crossover import EMACrossoverStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .rsi_divergence import RSIDivergenceStrategy
from .vwap_bounce import VWAPBounceStrategy

__all__ = [
    "Strategy",
    "StrategyResult",
    "Signal",
    "Side",
    "SignalType",
    "EMACrossoverStrategy",
    "BollingerSqueezeStrategy",
    "RSIDivergenceStrategy",
    "VWAPBounceStrategy",
]
