"""
Order Executor Module - FIXED VERSION
Comprehensive order execution system for trading bot
Handles order placement, tracking, and execution for both simulated and live trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import threading
import queue
import time
import logging
import warnings
from collections import defaultdict, deque
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class TimeInForce(Enum):
    """Time in force for orders"""
    GTC = "good_till_cancel"  # Good till cancelled
    IOC = "immediate_or_cancel"  # Immediate or cancel
    FOK = "fill_or_kill"  # Fill or kill
    GTD = "good_till_date"  # Good till date
    DAY = "day"  # Day order


@dataclass
class Order:
    """Order information"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    filled_quantity: float = 0
    filled_price: float = 0
    average_price: float = 0
    commission: float = 0
    slippage: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    error_message: Optional[str] = None
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to be filled"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, 
                               OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.FAILED]
    
    @property
    def is_active(self) -> bool:
        """Check if order is active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def fill_rate(self) -> float:
        """Percentage of order filled"""
        if self.quantity == 0:
            return 0
        return self.filled_quantity / self.quantity


@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order: Order
    execution_price: float
    actual_quantity: float
    commission: float
    slippage: float
    execution_time: datetime
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Current market conditions for execution"""
    bid: float
    ask: float
    last_price: float
    spread: float
    volume: float
    volatility: float
    liquidity: float
    order_book_depth: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


class OrderExecutor:
    """
    Advanced order execution system
    
    Features:
    - Multiple order types (market, limit, stop, etc.)
    - Smart order routing
    - Slippage modeling
    - Order tracking and management
    - Retry logic with exponential backoff
    - TWAP/VWAP execution algorithms
    - Iceberg orders
    - Simulated and live execution modes
    """
    
    def __init__(self,
                 mode: str = 'simulation',
                 commission_rate: float = 0.0026,  # Kraken's taker fee
                 slippage_model: str = 'linear',
                 max_slippage: float = 0.01,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize Order Executor
        
        Args:
            mode: 'simulation' or 'live'
            commission_rate: Trading commission rate
            slippage_model: 'linear', 'square_root', or 'logarithmic'
            max_slippage: Maximum allowed slippage
            retry_attempts: Maximum retry attempts for failed orders
            retry_delay: Base delay between retries (seconds)
        """
        self.mode = mode
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.max_slippage = max_slippage
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.active_orders: Dict[str, Order] = {}
        self.pending_orders: queue.Queue = queue.Queue()
        
        # Execution tracking
        self.execution_history: List[ExecutionResult] = []
        self.fills: List[Dict[str, Any]] = []
        
        # Market data cache
        self.market_data: Dict[str, MarketConditions] = {}
        self.order_books: Dict[str, Dict] = {}
        
        # Performance metrics
        self.total_commission_paid = 0
        self.total_slippage = 0
        self.execution_metrics = defaultdict(list)
        
        # Execution algorithms
        self.execution_algos = {
            'twap': self._execute_twap,
            'vwap': self._execute_vwap,
            'iceberg': self._execute_iceberg
        }
        
        # Callback functions
        self.on_fill_callbacks: List[Callable] = []
        self.on_cancel_callbacks: List[Callable] = []
        self.on_reject_callbacks: List[Callable] = []
        
        # For async execution
        self._stop_event = threading.Event()
        if mode == 'live':
            self._start_execution_thread()
        
        logger.info(f"OrderExecutor initialized in {mode} mode")
    
    # ================== ORDER CREATION ==================
    
    def create_order(self,
                    symbol: str,
                    side: Union[str, OrderSide],
                    quantity: float,
                    order_type: Union[str, OrderType] = OrderType.MARKET,
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: Union[str, TimeInForce] = TimeInForce.GTC,
                    metadata: Optional[Dict] = None) -> Order:
        """
        Create a new order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Type of order
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            metadata: Additional metadata
            
        Returns:
            Created order
        """
        # Convert strings to enums if necessary
        if isinstance(side, str):
            side = OrderSide(side.lower())
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())
        if isinstance(time_in_force, str):
            time_in_force = TimeInForce(time_in_force.lower())
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata or {}
        )
        
        # Store order
        self.orders[order_id] = order
        
        logger.info(f"Created {order_type.value} {side.value} order for {quantity} {symbol}")
        
        return order
    
    # ================== ORDER SUBMISSION ==================
    
    def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit order for execution
        
        Args:
            order: Order to submit
            
        Returns:
            Execution result
        """
        try:
            # Update order status
            order.status = OrderStatus.SUBMITTED
            order.submitted_time = datetime.now()
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Route based on mode
            if self.mode == 'simulation':
                result = self._simulate_execution(order)
            else:
                result = self._live_execution(order)
            
            # Update tracking
            if result.success:
                if order.status == OrderStatus.FILLED:
                    self._on_order_filled(order, result)
            else:
                # Keep order active if it's pending (limit/stop not triggered)
                if order.status != OrderStatus.PENDING:
                    order.status = OrderStatus.FAILED
                    order.error_message = result.message
            
            # Store result
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.FAILED
            order.error_message = str(e)
            
            return ExecutionResult(
                success=False,
                order=order,
                execution_price=0,
                actual_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                message=str(e)
            )
    
    # ================== EXECUTION MODES ==================
    
    def _simulate_execution(self, order: Order) -> ExecutionResult:
        """
        Simulate order execution for backtesting/paper trading
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        # Get market conditions
        market = self._get_market_conditions(order.symbol)
        
        # Calculate execution price based on order type
        if order.order_type == OrderType.MARKET:
            execution_price = self._simulate_market_order(order, market)
        elif order.order_type == OrderType.LIMIT:
            execution_price = self._simulate_limit_order(order, market)
        elif order.order_type == OrderType.STOP:
            execution_price = self._simulate_stop_order(order, market)
        else:
            execution_price = market.last_price
        
        # Check if order should fill
        if execution_price == 0:
            # Order doesn't fill (limit/stop condition not met)
            return ExecutionResult(
                success=False,
                order=order,
                execution_price=0,
                actual_quantity=0,
                commission=0,
                slippage=0,
                execution_time=datetime.now(),
                message="Order pending - conditions not met",
                metadata={'market_conditions': market}
            )
        
        # Apply slippage (FIXED: use execution_price instead of average_price)
        slippage = self._calculate_slippage(order, market, execution_price)
        if order.side == OrderSide.BUY:
            execution_price *= (1 + slippage)
        else:
            execution_price *= (1 - slippage)
        
        # Calculate commission
        commission = order.quantity * execution_price * self.commission_rate
        
        # Fill order
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.average_price = execution_price
        order.commission = commission
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        order.filled_time = datetime.now()
        
        # Update totals
        self.total_commission_paid += commission
        self.total_slippage += slippage * order.quantity * execution_price
        
        return ExecutionResult(
            success=True,
            order=order,
            execution_price=execution_price,
            actual_quantity=order.quantity,
            commission=commission,
            slippage=slippage,
            execution_time=datetime.now(),
            message="Order filled (simulated)",
            metadata={'market_conditions': market}
        )
    
    def _simulate_market_order(self, order: Order, market: MarketConditions) -> float:
        """Simulate market order execution"""
        if order.side == OrderSide.BUY:
            return market.ask
        else:
            return market.bid
    
    def _simulate_limit_order(self, order: Order, market: MarketConditions) -> float:
        """Simulate limit order execution"""
        if order.price is None:
            return market.last_price
        
        if order.side == OrderSide.BUY:
            # Buy limit order fills if market price <= limit price
            if market.ask <= order.price:
                return min(order.price, market.ask)
            else:
                # Order not filled yet
                order.status = OrderStatus.PENDING
                return 0
        else:
            # Sell limit order fills if market price >= limit price
            if market.bid >= order.price:
                return max(order.price, market.bid)
            else:
                order.status = OrderStatus.PENDING
                return 0
    
    def _simulate_stop_order(self, order: Order, market: MarketConditions) -> float:
        """Simulate stop order execution"""
        if order.stop_price is None:
            return market.last_price
        
        if order.side == OrderSide.BUY:
            # Buy stop triggers when price >= stop price
            if market.last_price >= order.stop_price:
                return market.ask
            else:
                order.status = OrderStatus.PENDING
                return 0
        else:
            # Sell stop triggers when price <= stop price
            if market.last_price <= order.stop_price:
                return market.bid
            else:
                order.status = OrderStatus.PENDING
                return 0
    
    def _live_execution(self, order: Order) -> ExecutionResult:
        """
        Execute order in live trading (placeholder for Kraken integration)
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        # This will be implemented when Kraken connector is ready
        # For now, return simulated execution
        logger.warning("Live execution not implemented, using simulation")
        return self._simulate_execution(order)
    
    # ================== SLIPPAGE MODELING (FIXED) ==================
    
    def _calculate_slippage(self, order: Order, market: MarketConditions, 
                          execution_price: float) -> float:
        """
        Calculate slippage based on order size and market conditions
        
        Args:
            order: Order being executed
            market: Current market conditions
            execution_price: Expected execution price (FIXED: added parameter)
            
        Returns:
            Slippage percentage
        """
        # Base slippage from spread
        spread_slippage = market.spread / market.last_price
        
        # Size impact (larger orders have more slippage)
        # FIXED: use execution_price instead of order.average_price
        order_value = order.quantity * execution_price
        size_impact = order_value / market.volume if market.volume > 0 else 0
        
        # Volatility impact
        volatility_impact = market.volatility * 0.1
        
        # Liquidity impact (lower liquidity = higher slippage)
        liquidity_factor = 2.0 - market.liquidity  # 1.0 to 2.0
        
        # Calculate based on model
        if self.slippage_model == 'linear':
            slippage = (spread_slippage + size_impact * 0.1 + volatility_impact) * liquidity_factor
        elif self.slippage_model == 'square_root':
            slippage = (spread_slippage + np.sqrt(size_impact) * 0.01 + volatility_impact) * liquidity_factor
        elif self.slippage_model == 'logarithmic':
            slippage = (spread_slippage + np.log1p(size_impact * 10) * 0.001 + volatility_impact) * liquidity_factor
        else:
            slippage = spread_slippage * liquidity_factor
        
        # Cap at maximum slippage
        return min(slippage, self.max_slippage)
    
    # ================== ADVANCED EXECUTION ALGORITHMS ==================
    
    def execute_twap(self,
                    symbol: str,
                    side: Union[str, OrderSide],
                    total_quantity: float,
                    duration_minutes: int,
                    intervals: int = 10) -> List[ExecutionResult]:
        """
        Execute TWAP (Time-Weighted Average Price) order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to execute
            duration_minutes: Duration for execution
            intervals: Number of intervals to split order
            
        Returns:
            List of execution results
        """
        return self._execute_twap(symbol, side, total_quantity, duration_minutes, intervals)
    
    def _execute_twap(self,
                     symbol: str,
                     side: Union[str, OrderSide],
                     total_quantity: float,
                     duration_minutes: int,
                     intervals: int) -> List[ExecutionResult]:
        """
        TWAP execution implementation
        """
        results = []
        slice_size = total_quantity / intervals
        interval_seconds = (duration_minutes * 60) / intervals
        
        logger.info(f"Starting TWAP: {total_quantity} {symbol} over {duration_minutes} minutes in {intervals} slices")
        
        for i in range(intervals):
            # Create order for this slice
            slice_order = self.create_order(
                symbol=symbol,
                side=side,
                quantity=slice_size,
                order_type='market',
                metadata={'twap_slice': i+1, 'total_slices': intervals}
            )
            
            # Execute slice
            result = self.submit_order(slice_order)
            results.append(result)
            
            # Wait for next interval (in live mode)
            if self.mode == 'live' and i < intervals - 1:
                time.sleep(interval_seconds)
        
        # Calculate average execution
        total_executed = sum(r.actual_quantity for r in results)
        if total_executed > 0:
            avg_price = sum(r.execution_price * r.actual_quantity for r in results) / total_executed
            logger.info(f"TWAP complete: executed {total_executed} @ avg {avg_price:.2f}")
        
        return results
    
    def execute_vwap(self,
                    symbol: str,
                    side: Union[str, OrderSide],
                    total_quantity: float,
                    volume_profile: List[float]) -> List[ExecutionResult]:
        """
        Execute VWAP (Volume-Weighted Average Price) order
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to execute
            volume_profile: Expected volume distribution
            
        Returns:
            List of execution results
        """
        return self._execute_vwap(symbol, side, total_quantity, volume_profile)
    
    def _execute_vwap(self,
                     symbol: str,
                     side: Union[str, OrderSide],
                     total_quantity: float,
                     volume_profile: List[float]) -> List[ExecutionResult]:
        """
        VWAP execution implementation
        """
        results = []
        total_volume = sum(volume_profile)
        
        logger.info(f"Starting VWAP: {total_quantity} {symbol} following volume profile")
        
        for i, volume_weight in enumerate(volume_profile):
            # Calculate slice size based on volume weight
            slice_size = total_quantity * (volume_weight / total_volume)
            
            # Create order for this slice
            slice_order = self.create_order(
                symbol=symbol,
                side=side,
                quantity=slice_size,
                order_type='market',
                metadata={'vwap_slice': i+1, 'volume_weight': volume_weight}
            )
            
            # Execute slice
            result = self.submit_order(slice_order)
            results.append(result)
        
        # Summary
        total_executed = sum(r.actual_quantity for r in results)
        if total_executed > 0:
            avg_price = sum(r.execution_price * r.actual_quantity for r in results) / total_executed
            logger.info(f"VWAP complete: executed {total_executed} @ avg {avg_price:.2f}")
        
        return results
    
    def execute_iceberg(self,
                       symbol: str,
                       side: Union[str, OrderSide],
                       total_quantity: float,
                       visible_quantity: float,
                       price: Optional[float] = None) -> List[ExecutionResult]:
        """
        Execute Iceberg order (show only part of total order)
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            total_quantity: Total quantity to execute
            visible_quantity: Visible quantity per slice
            price: Optional limit price
            
        Returns:
            List of execution results
        """
        return self._execute_iceberg(symbol, side, total_quantity, visible_quantity, price)
    
    def _execute_iceberg(self,
                        symbol: str,
                        side: Union[str, OrderSide],
                        total_quantity: float,
                        visible_quantity: float,
                        price: Optional[float]) -> List[ExecutionResult]:
        """
        Iceberg execution implementation
        """
        results = []
        remaining = total_quantity
        slice_count = 0
        
        logger.info(f"Starting Iceberg: {total_quantity} {symbol} with {visible_quantity} visible")
        
        while remaining > 0:
            # Calculate slice size
            current_slice = min(visible_quantity, remaining)
            slice_count += 1
            
            # Determine order type
            order_type = 'limit' if price else 'market'
            
            # Create order for visible slice
            slice_order = self.create_order(
                symbol=symbol,
                side=side,
                quantity=current_slice,
                order_type=order_type,
                price=price,
                metadata={'iceberg_slice': slice_count, 'remaining': remaining}
            )
            
            # Execute slice
            result = self.submit_order(slice_order)
            results.append(result)
            
            # Update remaining
            remaining -= result.actual_quantity
            
            # Break if order wasn't fully filled (market moved)
            if result.actual_quantity < current_slice:
                logger.warning(f"Iceberg slice partially filled: {result.actual_quantity}/{current_slice}")
                if price and self.mode == 'simulation':
                    break  # Stop if limit price can't be met
        
        # Summary
        total_executed = sum(r.actual_quantity for r in results)
        avg_price = sum(r.execution_price * r.actual_quantity for r in results) / total_executed if total_executed > 0 else 0
        
        logger.info(f"Iceberg complete: executed {total_executed}/{total_quantity} @ avg {avg_price:.2f}")
        
        return results
    
    # ================== ORDER MANAGEMENT ==================
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELLED
        
        # Remove from active orders
        if order_id in self.active_orders:
            del self.active_orders[order_id]
        
        # Move to history
        self.order_history.append(order)
        
        # Call callbacks
        for callback in self.on_cancel_callbacks:
            callback(order)
        
        logger.info(f"Cancelled order {order_id}")
        return True
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all orders or all orders for a symbol
        
        Args:
            symbol: Optional symbol to filter
            
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        
        # Create list of order IDs to cancel (avoid modifying dict during iteration)
        orders_to_cancel = []
        for order_id, order in self.active_orders.items():
            if symbol is None or order.symbol == symbol:
                orders_to_cancel.append(order_id)
        
        # Cancel orders
        for order_id in orders_to_cancel:
            if self.cancel_order(order_id):
                cancelled += 1
        
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled
    
    def modify_order(self,
                    order_id: str,
                    new_quantity: Optional[float] = None,
                    new_price: Optional[float] = None,
                    new_stop: Optional[float] = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            new_quantity: New quantity
            new_price: New limit price
            new_stop: New stop price
            
        Returns:
            Success status
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            logger.warning(f"Order {order_id} is not active")
            return False
        
        # In simulation mode, just update the order
        if new_quantity is not None:
            order.quantity = new_quantity
        if new_price is not None:
            order.price = new_price
        if new_stop is not None:
            order.stop_price = new_stop
        
        logger.info(f"Modified order {order_id}")
        return True
    
    # ================== ORDER TRACKING ==================
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders"""
        orders = list(self.active_orders.values())
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    def get_order_history(self,
                         symbol: Optional[str] = None,
                         limit: int = 100) -> List[Order]:
        """Get order history"""
        history = self.order_history[-limit:]
        
        if symbol:
            history = [o for o in history if o.symbol == symbol]
        
        return history
    
    def get_fills(self,
                 symbol: Optional[str] = None,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> List[Dict]:
        """Get fill history"""
        fills = self.fills
        
        if symbol:
            fills = [f for f in fills if f['symbol'] == symbol]
        
        if start_time:
            fills = [f for f in fills if f['time'] >= start_time]
        
        if end_time:
            fills = [f for f in fills if f['time'] <= end_time]
        
        return fills
    
    # ================== CALLBACKS ==================
    
    def add_fill_callback(self, callback: Callable) -> None:
        """Add callback for order fills"""
        self.on_fill_callbacks.append(callback)
    
    def add_cancel_callback(self, callback: Callable) -> None:
        """Add callback for order cancellations"""
        self.on_cancel_callbacks.append(callback)
    
    def add_reject_callback(self, callback: Callable) -> None:
        """Add callback for order rejections"""
        self.on_reject_callbacks.append(callback)
    
    def _on_order_filled(self, order: Order, result: ExecutionResult) -> None:
        """Handle order fill"""
        # Record fill
        fill = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': result.actual_quantity,
            'price': result.execution_price,
            'commission': result.commission,
            'time': result.execution_time
        }
        self.fills.append(fill)
        
        # Remove from active orders
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        
        # Move to history
        self.order_history.append(order)
        
        # Call callbacks
        for callback in self.on_fill_callbacks:
            callback(order, result)
    
    # ================== MARKET CONDITIONS ==================
    
    def update_market_conditions(self, symbol: str, conditions: MarketConditions) -> None:
        """Update market conditions for a symbol"""
        self.market_data[symbol] = conditions
    
    def _get_market_conditions(self, symbol: str) -> MarketConditions:
        """Get current market conditions"""
        if symbol in self.market_data:
            return self.market_data[symbol]
        
        # Return default conditions for simulation
        return MarketConditions(
            bid=100.0,
            ask=100.1,
            last_price=100.05,
            spread=0.1,
            volume=1000000,
            volatility=0.02,
            liquidity=1.0
        )
    
    # ================== RETRY LOGIC ==================
    
    def _retry_order(self, order: Order) -> ExecutionResult:
        """
        Retry failed order with exponential backoff
        
        Args:
            order: Order to retry
            
        Returns:
            Execution result
        """
        max_attempts = self.retry_attempts
        base_delay = self.retry_delay
        
        for attempt in range(max_attempts):
            logger.info(f"Retry attempt {attempt + 1}/{max_attempts} for order {order.order_id}")
            
            # Exponential backoff
            if attempt > 0:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
            
            # Try to execute
            result = self.submit_order(order)
            
            if result.success:
                logger.info(f"Order {order.order_id} succeeded on retry {attempt + 1}")
                return result
            
            order.retry_count = attempt + 1
        
        logger.error(f"Order {order.order_id} failed after {max_attempts} attempts")
        order.status = OrderStatus.FAILED
        
        return ExecutionResult(
            success=False,
            order=order,
            execution_price=0,
            actual_quantity=0,
            commission=0,
            slippage=0,
            execution_time=datetime.now(),
            message=f"Failed after {max_attempts} retry attempts"
        )
    
    # ================== PERFORMANCE METRICS ==================
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics"""
        if not self.execution_history:
            return {}
        
        successful = [r for r in self.execution_history if r.success]
        failed = [r for r in self.execution_history if not r.success]
        
        metrics = {
            'total_orders': len(self.orders),
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.order_history),
            'success_rate': len(successful) / len(self.execution_history) if self.execution_history else 0,
            'total_commission': self.total_commission_paid,
            'total_slippage': self.total_slippage,
            'avg_slippage': np.mean([r.slippage for r in successful]) if successful else 0,
            'avg_fill_time': np.mean([(r.execution_time - r.order.created_time).total_seconds() 
                                     for r in successful]) if successful else 0,
            'failed_orders': len(failed),
            'pending_orders': len([o for o in self.active_orders.values() 
                                  if o.status == OrderStatus.PENDING])
        }
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get executor summary"""
        metrics = self.get_execution_metrics()
        
        return {
            'mode': self.mode,
            'orders': {
                'total': metrics.get('total_orders', 0),
                'active': metrics.get('active_orders', 0),
                'completed': metrics.get('completed_orders', 0),
                'pending': metrics.get('pending_orders', 0)
            },
            'performance': {
                'success_rate': metrics.get('success_rate', 0),
                'avg_slippage': metrics.get('avg_slippage', 0),
                'avg_fill_time': metrics.get('avg_fill_time', 0)
            },
            'costs': {
                'commission': self.total_commission_paid,
                'slippage': self.total_slippage
            }
        }
    
    # ================== CLEANUP ==================
    
    def stop(self) -> None:
        """Stop executor and cleanup"""
        # Cancel all active orders
        self.cancel_all_orders()
        
        # Stop execution thread
        if hasattr(self, '_execution_thread'):
            self._stop_event.set()
            self._execution_thread.join()
        
        logger.info("OrderExecutor stopped")
    
    def _start_execution_thread(self) -> None:
        """Start background thread for order execution (for live mode)"""
        self._execution_thread = threading.Thread(target=self._execution_loop)
        self._execution_thread.daemon = True
        self._execution_thread.start()
    
    def _execution_loop(self) -> None:
        """Background execution loop for processing orders"""
        while not self._stop_event.is_set():
            try:
                # Process pending orders
                if not self.pending_orders.empty():
                    order = self.pending_orders.get(timeout=1)
                    self.submit_order(order)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize executor in simulation mode
    executor = OrderExecutor(
        mode='simulation',
        commission_rate=0.0026,
        slippage_model='linear'
    )
    
    print("=== Order Executor Test ===\n")
    
    # Set up market conditions
    market = MarketConditions(
        bid=45000,
        ask=45010,
        last_price=45005,
        spread=10,
        volume=1000000,
        volatility=0.02,
        liquidity=0.9
    )
    executor.update_market_conditions('BTC', market)
    
    # 1. Market Order
    print("1. Market Order:")
    market_order = executor.create_order(
        symbol='BTC',
        side='buy',
        quantity=0.1,
        order_type='market'
    )
    result = executor.submit_order(market_order)
    print(f"   Executed: {result.actual_quantity} BTC @ ${result.execution_price:.2f}")
    print(f"   Commission: ${result.commission:.2f}")
    print(f"   Slippage: {result.slippage:.4%}\n")
    
    # 2. Limit Order
    print("2. Limit Order:")
    limit_order = executor.create_order(
        symbol='BTC',
        side='sell',
        quantity=0.05,
        order_type='limit',
        price=45100
    )
    result = executor.submit_order(limit_order)
    print(f"   Status: {limit_order.status.value}")
    if result.success:
        print(f"   Executed: {result.actual_quantity} BTC @ ${result.execution_price:.2f}\n")
    else:
        print(f"   Order pending (limit not met)\n")
    
    # 3. TWAP Order
    print("3. TWAP Order (Time-Weighted Average Price):")
    twap_results = executor.execute_twap(
        symbol='BTC',
        side='buy',
        total_quantity=1.0,
        duration_minutes=10,
        intervals=5
    )
    avg_price = sum(r.execution_price * r.actual_quantity for r in twap_results) / sum(r.actual_quantity for r in twap_results)
    print(f"   Executed {len(twap_results)} slices")
    print(f"   Average price: ${avg_price:.2f}\n")
    
    # 4. Iceberg Order
    print("4. Iceberg Order:")
    iceberg_results = executor.execute_iceberg(
        symbol='BTC',
        side='buy',
        total_quantity=2.0,
        visible_quantity=0.5
    )
    print(f"   Executed {len(iceberg_results)} slices")
    total_qty = sum(r.actual_quantity for r in iceberg_results)
    print(f"   Total executed: {total_qty} BTC\n")
    
    # 5. Get active orders
    print("5. Active Orders:")
    active = executor.get_active_orders()
    print(f"   {len(active)} active orders")
    for order in active:
        print(f"   - {order.symbol}: {order.side.value} {order.quantity} @ {order.price or 'market'}")
    
    # 6. Execution metrics
    print("\n6. Execution Metrics:")
    metrics = executor.get_execution_metrics()
    print(f"   Success Rate: {metrics.get('success_rate', 0):.2%}")
    print(f"   Avg Slippage: {metrics.get('avg_slippage', 0):.4%}")
    print(f"   Total Commission: ${metrics.get('total_commission', 0):.2f}")
    
    # 7. Summary
    print("\n7. Executor Summary:")
    summary = executor.get_summary()
    print(f"   Mode: {summary['mode']}")
    print(f"   Total Orders: {summary['orders']['total']}")
    print(f"   Success Rate: {summary['performance']['success_rate']:.2%}")
    print(f"   Total Costs: ${summary['costs']['commission'] + summary['costs']['slippage']:.2f}")
    
    print("\n Order Executor ready for integration!")