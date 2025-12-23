"""Trading strategies module."""

from .base import Side, Signal, SignalType, Strategy, StrategyResult
from .bollinger_squeeze import BollingerSqueezeStrategy
from .ema_crossover import EMACrossoverStrategy
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
