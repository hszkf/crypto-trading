"""Base exchange types and abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order representation."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity


@dataclass
class Position:
    """Position representation."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float = 0.0
    liquidation_price: float | None = None
    leverage: float = 1.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def notional_value(self) -> float:
        """Get notional value of position."""
        return self.quantity * self.current_price

    @property
    def pnl_percentage(self) -> float:
        """Get PnL as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == OrderSide.BUY:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100


@dataclass
class Balance:
    """Account balance."""
    currency: str
    total: float
    available: float
    locked: float = 0.0


class Exchange(ABC):
    """Abstract base class for exchange implementations."""

    def __init__(self, name: str, api_key: str = "", api_secret: str = "",
                 testnet: bool = True):
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to exchange."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        pass

    @abstractmethod
    async def get_balance(self, currency: str = "USDT") -> Balance:
        """Get account balance."""
        pass

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders."""
        pass

    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: float | None = None,
                          stop_price: float | None = None) -> Order:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders. Returns number cancelled."""
        pass

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                        limit: int = 500) -> pd.DataFrame:
        """Get OHLCV candle data."""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data."""
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Get orderbook."""
        pass

    async def place_market_order(self, symbol: str, side: OrderSide,
                                  quantity: float) -> Order:
        """Convenience method for market orders."""
        return await self.place_order(symbol, side, OrderType.MARKET, quantity)

    async def place_limit_order(self, symbol: str, side: OrderSide,
                                 quantity: float, price: float) -> Order:
        """Convenience method for limit orders."""
        return await self.place_order(symbol, side, OrderType.LIMIT, quantity, price)

    async def close_position(self, position: Position) -> Order:
        """Close an existing position."""
        close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
        return await self.place_market_order(position.symbol, close_side, position.quantity)

    def __repr__(self) -> str:
        mode = "testnet" if self.testnet else "live"
        return f"{self.__class__.__name__}({self.name}, {mode})"
