"""Unit tests for exchange module."""


import pytest

from src.exchange.base import (
    Balance,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class TestOrderType:
    """Tests for OrderType enum."""

    def test_order_types_exist(self):
        """Test all order types are defined."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_MARKET.value == "stop_market"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TAKE_PROFIT_MARKET.value == "take_profit_market"
        assert OrderType.TAKE_PROFIT_LIMIT.value == "take_profit_limit"


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_order_sides_exist(self):
        """Test order sides are defined."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_order_statuses_exist(self):
        """Test all order statuses are defined."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            id="order123",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1,
        )
        assert order.id == "order123"
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 0.1
        assert order.status == OrderStatus.PENDING

    def test_order_is_active_pending(self):
        """Test is_active returns True for pending orders."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.PENDING,
        )
        assert order.is_active is True

    def test_order_is_active_open(self):
        """Test is_active returns True for open orders."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.OPEN,
        )
        assert order.is_active is True

    def test_order_is_active_partially_filled(self):
        """Test is_active returns True for partially filled orders."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        assert order.is_active is True

    def test_order_is_not_active_filled(self):
        """Test is_active returns False for filled orders."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.FILLED,
        )
        assert order.is_active is False

    def test_order_is_not_active_cancelled(self):
        """Test is_active returns False for cancelled orders."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            status=OrderStatus.CANCELLED,
        )
        assert order.is_active is False

    def test_order_remaining_quantity_unfilled(self):
        """Test remaining_quantity for unfilled order."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            filled_quantity=0.0,
        )
        assert order.remaining_quantity == 1.0

    def test_order_remaining_quantity_partially_filled(self):
        """Test remaining_quantity for partially filled order."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            filled_quantity=0.3,
        )
        assert order.remaining_quantity == pytest.approx(0.7)

    def test_order_remaining_quantity_fully_filled(self):
        """Test remaining_quantity for fully filled order."""
        order = Order(
            id="1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            filled_quantity=1.0,
        )
        assert order.remaining_quantity == 0.0


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            entry_price=50000.0,
            current_price=51000.0,
        )
        assert position.symbol == "BTC/USDT"
        assert position.side == OrderSide.BUY
        assert position.quantity == 0.5
        assert position.entry_price == 50000.0
        assert position.current_price == 51000.0

    def test_position_notional_value(self):
        """Test notional value calculation."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=0.5,
            entry_price=50000.0,
            current_price=51000.0,
        )
        assert position.notional_value == 25500.0  # 0.5 * 51000

    def test_position_pnl_percentage_long_profit(self):
        """Test PnL percentage for profitable long position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            entry_price=50000.0,
            current_price=55000.0,
        )
        assert position.pnl_percentage == pytest.approx(10.0)  # 10% profit

    def test_position_pnl_percentage_long_loss(self):
        """Test PnL percentage for losing long position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            entry_price=50000.0,
            current_price=45000.0,
        )
        assert position.pnl_percentage == pytest.approx(-10.0)  # 10% loss

    def test_position_pnl_percentage_short_profit(self):
        """Test PnL percentage for profitable short position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            entry_price=50000.0,
            current_price=45000.0,
        )
        assert position.pnl_percentage == pytest.approx(10.0)  # 10% profit

    def test_position_pnl_percentage_short_loss(self):
        """Test PnL percentage for losing short position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=1.0,
            entry_price=50000.0,
            current_price=55000.0,
        )
        assert position.pnl_percentage == pytest.approx(-10.0)  # 10% loss

    def test_position_pnl_percentage_zero_entry(self):
        """Test PnL percentage with zero entry price."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.0,
            entry_price=0.0,
            current_price=50000.0,
        )
        assert position.pnl_percentage == 0.0


class TestBalance:
    """Tests for Balance dataclass."""

    def test_balance_creation(self):
        """Test basic balance creation."""
        balance = Balance(
            currency="USDT",
            total=10000.0,
            available=8000.0,
            locked=2000.0,
        )
        assert balance.currency == "USDT"
        assert balance.total == 10000.0
        assert balance.available == 8000.0
        assert balance.locked == 2000.0

    def test_balance_default_locked(self):
        """Test balance with default locked value."""
        balance = Balance(
            currency="USDT",
            total=10000.0,
            available=10000.0,
        )
        assert balance.locked == 0.0
