"""Signal manager - handles signal lifecycle and execution."""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..exchange.base import Exchange, Order, OrderSide, OrderType
from ..strategies.base import Side, Signal

logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Signal lifecycle state."""

    PENDING = "pending"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ManagedSignal:
    """Signal with management metadata."""

    signal: Signal
    state: SignalState = SignalState.PENDING
    entry_order: Order | None = None
    stop_order: Order | None = None
    take_profit_order: Order | None = None
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None
    expiry: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Check if signal is still actionable."""
        if self.state != SignalState.PENDING:
            return False
        if self.expiry and datetime.now() > self.expiry:
            return False
        return True


class SignalManager:
    """Manages signal lifecycle from generation to execution.

    Handles:
    - Signal queueing and deduplication
    - Expiration management
    - Order execution via exchange
    - Stop loss and take profit orders
    """

    def __init__(
        self,
        exchange: Exchange | None = None,
        signal_expiry_minutes: int = 60,
        auto_execute: bool = False,
    ):
        """Initialize signal manager.

        Args:
            exchange: Exchange client for order execution
            signal_expiry_minutes: Minutes until signal expires
            auto_execute: Automatically execute signals
        """
        self.exchange = exchange
        self.signal_expiry = timedelta(minutes=signal_expiry_minutes)
        self.auto_execute = auto_execute

        self._pending: dict[str, ManagedSignal] = {}  # symbol -> signal
        self._history: list[ManagedSignal] = []
        self._callbacks: list[Callable[[ManagedSignal], Awaitable[None]]] = []

    def add_signal(self, signal: Signal) -> ManagedSignal:
        """Add a new signal to the queue.

        Args:
            signal: Trading signal to add

        Returns:
            ManagedSignal wrapper
        """
        # Check for existing signal on same symbol
        existing = self._pending.get(signal.symbol)
        if existing and existing.is_active:
            # Only replace if new signal has higher confidence
            if signal.confidence <= existing.signal.confidence:
                logger.debug(
                    f"Ignoring signal for {signal.symbol}, existing signal has higher confidence"
                )
                return existing

            # Expire old signal
            existing.state = SignalState.EXPIRED
            self._history.append(existing)

        managed = ManagedSignal(signal=signal, expiry=datetime.now() + self.signal_expiry)
        self._pending[signal.symbol] = managed

        logger.info(f"New signal: {signal.side.value} {signal.symbol} @ {signal.price}")

        return managed

    def add_signals(self, signals: list[Signal]) -> list[ManagedSignal]:
        """Add multiple signals."""
        return [self.add_signal(s) for s in signals]

    async def execute_signal(self, managed: ManagedSignal, position_size: float) -> bool:
        """Execute a signal by placing orders.

        Args:
            managed: ManagedSignal to execute
            position_size: Size in base currency

        Returns:
            True if execution successful
        """
        if not self.exchange:
            logger.error("No exchange configured")
            return False

        if not managed.is_active:
            logger.warning(f"Signal is not active: {managed.state}")
            return False

        signal = managed.signal
        order_side = OrderSide.BUY if signal.side == Side.LONG else OrderSide.SELL

        try:
            # Place entry order
            entry_order = await self.exchange.place_market_order(
                signal.symbol, order_side, position_size
            )
            managed.entry_order = entry_order

            # Place stop loss order
            if signal.stop_loss:
                stop_side = OrderSide.SELL if signal.side == Side.LONG else OrderSide.BUY
                stop_order = await self.exchange.place_order(
                    signal.symbol,
                    stop_side,
                    OrderType.STOP_MARKET,
                    position_size,
                    stop_price=signal.stop_loss,
                )
                managed.stop_order = stop_order

            # Place take profit order
            if signal.take_profit:
                tp_side = OrderSide.SELL if signal.side == Side.LONG else OrderSide.BUY
                tp_order = await self.exchange.place_order(
                    signal.symbol,
                    tp_side,
                    OrderType.TAKE_PROFIT_MARKET,
                    position_size,
                    stop_price=signal.take_profit,
                )
                managed.take_profit_order = tp_order

            managed.state = SignalState.EXECUTED
            managed.executed_at = datetime.now()

            logger.info(f"Signal executed: {signal.symbol} entry @ {entry_order.average_price}")

            # Trigger callbacks
            for callback in self._callbacks:
                await callback(managed)

            return True

        except Exception as e:
            logger.error(f"Failed to execute signal: {e}")
            managed.state = SignalState.FAILED
            return False

    async def execute_pending(
        self, position_sizer: Callable[[Signal], float]
    ) -> list[ManagedSignal]:
        """Execute all pending signals.

        Args:
            position_sizer: Function to calculate position size for signal

        Returns:
            List of executed signals
        """
        executed = []

        for symbol, managed in list(self._pending.items()):
            if not managed.is_active:
                continue

            position_size = position_sizer(managed.signal)
            if position_size <= 0:
                continue

            success = await self.execute_signal(managed, position_size)
            if success:
                executed.append(managed)
                del self._pending[symbol]
                self._history.append(managed)

        return executed

    def cancel_signal(self, symbol: str) -> bool:
        """Cancel a pending signal."""
        managed = self._pending.get(symbol)
        if managed and managed.is_active:
            managed.state = SignalState.CANCELLED
            del self._pending[symbol]
            self._history.append(managed)
            logger.info(f"Cancelled signal for {symbol}")
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove expired signals. Returns count removed."""
        expired = []
        for symbol, managed in list(self._pending.items()):
            if managed.expiry and datetime.now() > managed.expiry:
                managed.state = SignalState.EXPIRED
                expired.append(symbol)
                self._history.append(managed)

        for symbol in expired:
            del self._pending[symbol]

        if expired:
            logger.info(f"Expired {len(expired)} signals")

        return len(expired)

    def on_execution(self, callback: Callable[[ManagedSignal], Awaitable[None]]) -> None:
        """Register callback for signal execution."""
        self._callbacks.append(callback)

    def get_pending(self, symbol: str | None = None) -> list[ManagedSignal]:
        """Get pending signals."""
        if symbol:
            managed = self._pending.get(symbol)
            return [managed] if managed and managed.is_active else []
        return [m for m in self._pending.values() if m.is_active]

    def get_history(self, limit: int = 100) -> list[ManagedSignal]:
        """Get signal history."""
        return self._history[-limit:]

    @property
    def pending_count(self) -> int:
        """Get count of pending signals."""
        return len([m for m in self._pending.values() if m.is_active])

    def __repr__(self) -> str:
        return f"SignalManager(pending={self.pending_count}, history={len(self._history)})"
