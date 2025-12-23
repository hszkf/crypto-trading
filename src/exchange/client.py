"""CCXT-based exchange client implementation."""

import logging
from datetime import datetime

import pandas as pd

from .base import (
    Balance,
    Exchange,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)


class ExchangeClient(Exchange):
    """Exchange client using ccxt library.

    Supports major crypto exchanges: Binance, Bybit, OKX, etc.
    """

    # Map internal order types to ccxt
    ORDER_TYPE_MAP = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP_MARKET: "stop",
        OrderType.STOP_LIMIT: "stopLimit",
    }

    # Map ccxt status to internal status
    STATUS_MAP = {
        "open": OrderStatus.OPEN,
        "closed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELLED,
        "expired": OrderStatus.EXPIRED,
        "rejected": OrderStatus.REJECTED,
    }

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "",
                 testnet: bool = True, sandbox: bool = True):
        super().__init__(exchange_id, api_key, api_secret, testnet)
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self._client = None
        self._connected = False

    async def connect(self) -> bool:
        """Connect to exchange via ccxt."""
        try:
            # Dynamic import to avoid hard dependency
            import ccxt.async_support as ccxt

            exchange_class = getattr(ccxt, self.exchange_id, None)
            if not exchange_class:
                logger.error(f"Exchange {self.exchange_id} not supported by ccxt")
                return False

            config = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},  # Use futures by default
            }

            self._client = exchange_class(config)

            if self.sandbox:
                self._client.set_sandbox_mode(True)

            # Test connection
            await self._client.load_markets()
            self._connected = True
            logger.info(f"Connected to {self.exchange_id} ({'sandbox' if self.sandbox else 'live'})")
            return True

        except ImportError:
            logger.error("ccxt library not installed. Run: pip install ccxt")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info(f"Disconnected from {self.exchange_id}")

    def _ensure_connected(self) -> None:
        """Ensure client is connected."""
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to exchange. Call connect() first.")

    async def get_balance(self, currency: str = "USDT") -> Balance:
        """Get account balance."""
        self._ensure_connected()

        balance = await self._client.fetch_balance()
        currency_balance = balance.get(currency, {})

        return Balance(
            currency=currency,
            total=float(currency_balance.get("total", 0)),
            available=float(currency_balance.get("free", 0)),
            locked=float(currency_balance.get("used", 0))
        )

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """Get open positions."""
        self._ensure_connected()

        positions = await self._client.fetch_positions(symbols=[symbol] if symbol else None)
        result = []

        for pos in positions:
            if float(pos.get("contracts", 0)) == 0:
                continue

            side = OrderSide.BUY if pos.get("side") == "long" else OrderSide.SELL

            result.append(Position(
                symbol=pos["symbol"],
                side=side,
                quantity=float(pos.get("contracts", 0)),
                entry_price=float(pos.get("entryPrice", 0)),
                current_price=float(pos.get("markPrice", 0)),
                liquidation_price=float(pos.get("liquidationPrice", 0)) if pos.get("liquidationPrice") else None,
                leverage=float(pos.get("leverage", 1)),
                unrealized_pnl=float(pos.get("unrealizedPnl", 0)),
                margin=float(pos.get("initialMargin", 0))
            ))

        return result

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get open orders."""
        self._ensure_connected()

        orders = await self._client.fetch_open_orders(symbol)
        return [self._parse_order(o) for o in orders]

    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType,
                          quantity: float, price: float | None = None,
                          stop_price: float | None = None) -> Order:
        """Place an order."""
        self._ensure_connected()

        ccxt_side = "buy" if side == OrderSide.BUY else "sell"
        ccxt_type = self.ORDER_TYPE_MAP.get(order_type, "market")

        params = {}
        if stop_price:
            params["stopPrice"] = stop_price

        try:
            result = await self._client.create_order(
                symbol=symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=quantity,
                price=price,
                params=params
            )

            logger.info(f"Order placed: {result['id']} {side.value} {quantity} {symbol}")
            return self._parse_order(result)

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order."""
        self._ensure_connected()

        try:
            await self._client.cancel_order(order_id, symbol)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all orders."""
        self._ensure_connected()

        try:
            result = await self._client.cancel_all_orders(symbol)
            count = len(result) if isinstance(result, list) else 1
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                        limit: int = 500) -> pd.DataFrame:
        """Get OHLCV candle data."""
        self._ensure_connected()

        ohlcv = await self._client.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

    async def get_ticker(self, symbol: str) -> dict:
        """Get current ticker data."""
        self._ensure_connected()

        ticker = await self._client.fetch_ticker(symbol)
        return {
            "symbol": ticker["symbol"],
            "last": ticker["last"],
            "bid": ticker["bid"],
            "ask": ticker["ask"],
            "high": ticker["high"],
            "low": ticker["low"],
            "volume": ticker["baseVolume"],
            "change": ticker["percentage"],
            "timestamp": ticker["timestamp"]
        }

    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """Get orderbook."""
        self._ensure_connected()

        ob = await self._client.fetch_order_book(symbol, limit)
        return {
            "bids": ob["bids"],
            "asks": ob["asks"],
            "timestamp": ob["timestamp"]
        }

    def _parse_order(self, data: dict) -> Order:
        """Parse ccxt order to internal Order."""
        side = OrderSide.BUY if data["side"] == "buy" else OrderSide.SELL

        order_type = OrderType.MARKET
        if data.get("type") == "limit":
            order_type = OrderType.LIMIT
        elif data.get("type") == "stop":
            order_type = OrderType.STOP_MARKET

        status = self.STATUS_MAP.get(data.get("status", ""), OrderStatus.PENDING)

        return Order(
            id=str(data["id"]),
            symbol=data["symbol"],
            side=side,
            order_type=order_type,
            quantity=float(data.get("amount", 0)),
            price=float(data.get("price", 0)) if data.get("price") else None,
            stop_price=float(data.get("stopPrice", 0)) if data.get("stopPrice") else None,
            status=status,
            filled_quantity=float(data.get("filled", 0)),
            average_price=float(data.get("average", 0)) if data.get("average") else 0,
            created_at=datetime.fromtimestamp(data["timestamp"] / 1000) if data.get("timestamp") else datetime.now()
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
