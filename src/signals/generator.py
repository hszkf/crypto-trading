"""Signal generator - combines multiple strategies."""

import logging

import pandas as pd

from ..strategies.base import Side, Signal, Strategy

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals from multiple strategies.

    Aggregates signals from multiple strategies and applies
    confluence filtering to improve signal quality.
    """

    def __init__(self, strategies: list[Strategy] = None,
                 min_confluence: int = 1):
        """Initialize signal generator.

        Args:
            strategies: List of trading strategies
            min_confluence: Minimum strategies that must agree for signal
        """
        self.strategies = strategies or []
        self.min_confluence = min_confluence
        self._last_signals: dict[str, Signal] = {}

    def add_strategy(self, strategy: Strategy) -> None:
        """Add a strategy to the generator."""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy by name."""
        for i, s in enumerate(self.strategies):
            if s.name == strategy_name:
                self.strategies.pop(i)
                logger.info(f"Removed strategy: {strategy_name}")
                return True
        return False

    def generate(self, data: pd.DataFrame) -> list[Signal]:
        """Generate signals from all active strategies.

        Args:
            data: OHLCV DataFrame

        Returns:
            List of signals passing confluence filter
        """
        all_signals: list[Signal] = []

        for strategy in self.strategies:
            if not strategy.is_active:
                continue

            try:
                signal = strategy.get_entry_signal(data)
                if signal:
                    signal.metadata["strategy"] = strategy.name
                    all_signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")

        # Apply confluence filter
        filtered = self._apply_confluence(all_signals)

        for signal in filtered:
            self._last_signals[signal.symbol] = signal

        return filtered

    def _apply_confluence(self, signals: list[Signal]) -> list[Signal]:
        """Filter signals by confluence (multiple strategies agree)."""
        if not signals or self.min_confluence <= 1:
            return signals

        # Group by symbol and direction
        groups: dict[tuple, list[Signal]] = {}
        for signal in signals:
            key = (signal.symbol, signal.side)
            if key not in groups:
                groups[key] = []
            groups[key].append(signal)

        # Filter by minimum confluence
        result = []
        for (_symbol, _side), group in groups.items():
            if len(group) >= self.min_confluence:
                # Merge signals - use best stop/target
                merged = self._merge_signals(group)
                result.append(merged)

        return result

    def _merge_signals(self, signals: list[Signal]) -> Signal:
        """Merge multiple signals into one with best parameters."""
        base = signals[0]

        # Collect all stop losses and take profits
        stop_losses = [s.stop_loss for s in signals if s.stop_loss]
        take_profits = [s.take_profit for s in signals if s.take_profit]

        # Use tightest stop loss (most conservative)
        if stop_losses:
            if base.side == Side.LONG:
                base.stop_loss = max(stop_losses)  # Highest for long
            else:
                base.stop_loss = min(stop_losses)  # Lowest for short

        # Use furthest take profit (most optimistic)
        if take_profits:
            if base.side == Side.LONG:
                base.take_profit = max(take_profits)
            else:
                base.take_profit = min(take_profits)

        # Update metadata
        base.metadata["confluence"] = len(signals)
        base.metadata["strategies"] = [s.metadata.get("strategy") for s in signals]
        base.confidence = len(signals) / len(self.strategies)

        return base

    def get_exit_signals(self, data: pd.DataFrame,
                         positions: dict[str, Side]) -> list[Signal]:
        """Generate exit signals for open positions.

        Args:
            data: OHLCV DataFrame
            positions: Dict mapping symbol to position side

        Returns:
            List of exit signals
        """
        exit_signals = []

        for symbol, side in positions.items():
            for strategy in self.strategies:
                if not strategy.is_active or strategy.symbol != symbol:
                    continue

                try:
                    signal = strategy.get_exit_signal(data, side)
                    if signal:
                        signal.metadata["strategy"] = strategy.name
                        exit_signals.append(signal)
                        break  # One exit signal per position is enough
                except Exception as e:
                    logger.error(f"Error getting exit signal from {strategy.name}: {e}")

        return exit_signals

    def get_last_signal(self, symbol: str) -> Optional[Signal]:
        """Get last generated signal for a symbol."""
        return self._last_signals.get(symbol)

    @property
    def active_strategies(self) -> list[Strategy]:
        """Get list of active strategies."""
        return [s for s in self.strategies if s.is_active]

    def __repr__(self) -> str:
        return f"SignalGenerator(strategies={len(self.strategies)}, confluence={self.min_confluence})"
