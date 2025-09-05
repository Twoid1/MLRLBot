"""
Portfolio Manager Module
Complete portfolio management system for multi-asset trading
Tracks positions, calculates metrics, manages allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Individual position information"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    position_type: str  # 'long' or 'short'
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: bool = False
    highest_price: float = 0
    lowest_price: float = float('inf')
    fees_paid: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        if self.position_type == 'long':
            return (self.current_price - self.entry_price) * self.quantity - self.fees_paid
        else:  # short
            return (self.entry_price - self.current_price) * self.quantity - self.fees_paid
    
    @property
    def unrealized_return(self) -> float:
        """Unrealized return percentage"""
        cost_basis = self.entry_price * self.quantity
        if cost_basis == 0:
            return 0
        return self.unrealized_pnl / cost_basis
    
    @property
    def duration(self) -> timedelta:
        """How long position has been held"""
        return datetime.now() - self.entry_time


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio at a point in time"""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    n_positions: int
    exposure: Dict[str, float]
    allocation: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    recovery_factor: float
    calmar_ratio: float
    var_95: float
    cvar_95: float


class Portfolio:
    """
    Main portfolio management class
    
    Features:
    - Multi-asset position tracking
    - Real-time P&L calculation
    - Portfolio metrics and analytics
    - Rebalancing and allocation
    - Risk exposure tracking
    - Transaction history
    """
    
    def __init__(self,
                 initial_capital: float,
                 base_currency: str = 'USD',
                 max_positions: int = 10,
                 enable_short: bool = False):
        """
        Initialize portfolio
        
        Args:
            initial_capital: Starting capital
            base_currency: Base currency for portfolio
            max_positions: Maximum number of concurrent positions
            enable_short: Allow short positions
        """
        self.initial_capital = initial_capital
        self.base_currency = base_currency
        self.max_positions = max_positions
        self.enable_short = enable_short
        
        # Current state
        self.cash_balance = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: Dict[str, Dict] = {}
        
        # Performance tracking
        self.realized_pnl = 0
        self.total_fees_paid = 0
        self.peak_value = initial_capital
        
        # History tracking
        self.transaction_history = []
        self.closed_positions = []
        self.portfolio_history = []
        self.daily_returns = []
        
        # Risk tracking
        self.exposure_by_asset = defaultdict(float)
        self.exposure_by_sector = defaultdict(float)
        self.correlation_matrix = pd.DataFrame()
        
        # Take initial snapshot
        self._take_snapshot()
        
        logger.info(f"Portfolio initialized with ${initial_capital:,.2f} capital")
    
    # ================== POSITION MANAGEMENT ==================
    
    def open_position(self,
                     symbol: str,
                     quantity: float,
                     price: float,
                     position_type: str = 'long',
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     fees: float = 0,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Open a new position
        
        Args:
            symbol: Asset symbol
            quantity: Position size
            price: Entry price
            position_type: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
            fees: Trading fees
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        # Validation
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        if not self.enable_short and position_type == 'short':
            logger.warning("Short positions not enabled")
            return False
        
        required_capital = quantity * price + fees
        
        if required_capital > self.cash_balance:
            logger.warning(f"Insufficient capital: need ${required_capital:.2f}, have ${self.cash_balance:.2f}")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            position_type=position_type,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees_paid=fees,
            highest_price=price,
            lowest_price=price,
            metadata=metadata or {}
        )
        
        # Update portfolio
        self.positions[symbol] = position
        self.cash_balance -= required_capital
        self.total_fees_paid += fees
        
        # Update exposure
        self.exposure_by_asset[symbol] += quantity * price
        
        # Record transaction
        self._record_transaction({
            'type': 'open',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'position_type': position_type,
            'fees': fees,
            'timestamp': datetime.now(),
            'cash_balance': self.cash_balance
        })
        
        logger.info(f"Opened {position_type} position: {quantity} {symbol} @ ${price:.2f}")
        
        return True
    
    def close_position(self,
                      symbol: str,
                      price: float,
                      fees: float = 0,
                      reason: str = 'manual') -> Optional[Dict]:
        """
        Close an existing position
        
        Args:
            symbol: Asset symbol
            price: Exit price
            fees: Trading fees
            reason: Reason for closing
            
        Returns:
            Trade result dictionary
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.position_type == 'long':
            gross_pnl = (price - position.entry_price) * position.quantity
        else:  # short
            gross_pnl = (position.entry_price - price) * position.quantity
        
        net_pnl = gross_pnl - fees - position.fees_paid
        
        # Calculate return
        cost_basis = position.entry_price * position.quantity
        return_pct = net_pnl / cost_basis if cost_basis > 0 else 0
        
        # Update portfolio
        self.cash_balance += position.quantity * price - fees
        self.realized_pnl += net_pnl
        self.total_fees_paid += fees
        
        # Update exposure
        self.exposure_by_asset[symbol] -= position.market_value
        
        # Store in closed positions
        closed_position = {
            'symbol': symbol,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': price,
            'position_type': position.position_type,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'duration': position.duration.total_seconds() / 3600,  # hours
            'gross_pnl': gross_pnl,
            'fees': fees + position.fees_paid,
            'net_pnl': net_pnl,
            'return_pct': return_pct,
            'reason': reason
        }
        
        self.closed_positions.append(closed_position)
        
        # Record transaction
        self._record_transaction({
            'type': 'close',
            'symbol': symbol,
            'quantity': position.quantity,
            'price': price,
            'position_type': position.position_type,
            'pnl': net_pnl,
            'fees': fees,
            'timestamp': datetime.now(),
            'cash_balance': self.cash_balance
        })
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {position.position_type} position: {symbol} P&L: ${net_pnl:.2f} ({return_pct:.2%})")
        
        return closed_position
    
    def update_position_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a position
        
        Args:
            symbol: Asset symbol
            price: Current price
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.current_price = price
        
        # Track highs and lows
        position.highest_price = max(position.highest_price, price)
        position.lowest_price = min(position.lowest_price, price)
        
        # Check stop loss
        if position.stop_loss:
            if position.position_type == 'long' and price <= position.stop_loss:
                self.close_position(symbol, price, reason='stop_loss')
            elif position.position_type == 'short' and price >= position.stop_loss:
                self.close_position(symbol, price, reason='stop_loss')
        
        # Check take profit
        if position.take_profit:
            if position.position_type == 'long' and price >= position.take_profit:
                self.close_position(symbol, price, reason='take_profit')
            elif position.position_type == 'short' and price <= position.take_profit:
                self.close_position(symbol, price, reason='take_profit')
    
    def update_all_prices(self, prices: Dict[str, float]) -> None:
        """
        Update prices for all positions
        
        Args:
            prices: Dictionary of symbol to price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.update_position_price(symbol, price)
    
    # ================== PORTFOLIO METRICS ==================
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash_balance + positions_value
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions"""
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_return(self) -> float:
        """Total return percentage"""
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak"""
        if self.total_value > self.peak_value:
            self.peak_value = self.total_value
            return 0
        return (self.peak_value - self.total_value) / self.peak_value
    
    def get_position_allocation(self) -> Dict[str, float]:
        """
        Get current position allocation as percentage of portfolio
        
        Returns:
            Dictionary of symbol to allocation percentage
        """
        total = self.total_value
        if total == 0:
            return {}
        
        allocation = {}
        for symbol, position in self.positions.items():
            allocation[symbol] = position.market_value / total
        
        allocation['cash'] = self.cash_balance / total
        
        return allocation
    
    def get_exposure_summary(self) -> Dict[str, Any]:
        """
        Get exposure summary
        
        Returns:
            Dictionary with exposure metrics
        """
        total_long = sum(
            p.market_value for p in self.positions.values() 
            if p.position_type == 'long'
        )
        total_short = sum(
            p.market_value for p in self.positions.values() 
            if p.position_type == 'short'
        )
        
        return {
            'total_exposure': total_long + total_short,
            'long_exposure': total_long,
            'short_exposure': total_short,
            'net_exposure': total_long - total_short,
            'gross_leverage': (total_long + total_short) / self.total_value if self.total_value > 0 else 0,
            'n_positions': len(self.positions),
            'n_long': sum(1 for p in self.positions.values() if p.position_type == 'long'),
            'n_short': sum(1 for p in self.positions.values() if p.position_type == 'short'),
            'largest_position': max(self.positions.values(), key=lambda p: p.market_value).symbol if self.positions else None,
            'concentration': max(self.get_position_allocation().values()) if self.positions else 0
        }
    
    def calculate_performance_metrics(self,
                                     period_days: Optional[int] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            period_days: Period for calculation (None = all history)
            
        Returns:
            Performance metrics
        """
        # Get returns history
        returns = self._get_returns_series(period_days)
        
        if len(returns) < 2:
            # Return default metrics if insufficient data
            return PerformanceMetrics(
                total_return=self.total_return,
                annual_return=0,
                volatility=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                current_drawdown=self.current_drawdown,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                recovery_factor=0,
                calmar_ratio=0,
                var_95=0,
                cvar_95=0
            )
        
        # Basic metrics
        total_return = (self.total_value / self.initial_capital) - 1
        
        # Annualized return (assuming daily returns)
        n_periods = len(returns)
        annual_factor = 252 / n_periods if n_periods > 0 else 1
        annual_return = (1 + total_return) ** annual_factor - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = 0
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Win/loss statistics
        closed = self.closed_positions[-period_days:] if period_days else self.closed_positions
        
        if closed:
            wins = [t for t in closed if t['net_pnl'] > 0]
            losses = [t for t in closed if t['net_pnl'] < 0]
            
            win_rate = len(wins) / len(closed)
            avg_win = np.mean([t['net_pnl'] for t in wins]) if wins else 0
            avg_loss = abs(np.mean([t['net_pnl'] for t in losses])) if losses else 0
            
            # Profit factor
            total_wins = sum(t['net_pnl'] for t in wins)
            total_losses = abs(sum(t['net_pnl'] for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            # Recovery factor
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            recovery_factor = 0
        
        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) * self.total_value
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * self.total_value
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            current_drawdown=self.current_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar_ratio,
            var_95=abs(var_95),
            cvar_95=abs(cvar_95)
        )
    
    # ================== REBALANCING ==================
    
    def rebalance_to_weights(self,
                            target_weights: Dict[str, float],
                            prices: Dict[str, float],
                            threshold: float = 0.05) -> List[Dict]:
        """
        Rebalance portfolio to target weights
        
        Args:
            target_weights: Target allocation weights
            prices: Current prices
            threshold: Rebalancing threshold (don't rebalance if difference < threshold)
            
        Returns:
            List of rebalancing trades
        """
        trades = []
        current_allocation = self.get_position_allocation()
        total_value = self.total_value
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_allocation.get(symbol, 0)
            
            # Skip if difference is below threshold
            if abs(current_weight - target_weight) < threshold:
                continue
            
            # Calculate target value and current value
            target_value = total_value * target_weight
            current_value = self.positions[symbol].market_value if symbol in self.positions else 0
            
            # Calculate trade size
            trade_value = target_value - current_value
            
            if symbol not in prices:
                logger.warning(f"No price available for {symbol}")
                continue
            
            trade_quantity = trade_value / prices[symbol]
            
            if trade_quantity > 0:
                # Buy
                trades.append({
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': trade_quantity,
                    'price': prices[symbol],
                    'value': trade_value
                })
            elif trade_quantity < 0:
                # Sell
                trades.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': abs(trade_quantity),
                    'price': prices[symbol],
                    'value': abs(trade_value)
                })
        
        return trades
    
    def apply_rebalancing_trades(self, trades: List[Dict], fees_per_trade: float = 0) -> bool:
        """
        Apply rebalancing trades
        
        Args:
            trades: List of trades from rebalance_to_weights
            fees_per_trade: Fees per trade
            
        Returns:
            Success status
        """
        for trade in trades:
            if trade['action'] == 'buy':
                success = self.open_position(
                    symbol=trade['symbol'],
                    quantity=trade['quantity'],
                    price=trade['price'],
                    position_type='long',
                    fees=fees_per_trade
                )
            else:  # sell
                if trade['symbol'] in self.positions:
                    self.close_position(
                        symbol=trade['symbol'],
                        price=trade['price'],
                        fees=fees_per_trade,
                        reason='rebalancing'
                    )
            
            if not success:
                logger.warning(f"Failed to execute rebalancing trade: {trade}")
                return False
        
        return True
    
    # ================== RISK ANALYTICS ==================
    
    def calculate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for positions
        
        Args:
            returns_data: DataFrame with returns for each asset
            
        Returns:
            Correlation matrix
        """
        # Filter to current positions
        current_symbols = list(self.positions.keys())
        
        if not current_symbols:
            return pd.DataFrame()
        
        # Calculate correlation
        if all(symbol in returns_data.columns for symbol in current_symbols):
            correlation_matrix = returns_data[current_symbols].corr()
            self.correlation_matrix = correlation_matrix
            return correlation_matrix
        
        return pd.DataFrame()
    
    def get_portfolio_var(self, returns_data: pd.DataFrame, confidence: float = 0.95) -> float:
        """
        Calculate portfolio Value at Risk
        
        Args:
            returns_data: Historical returns data
            confidence: Confidence level
            
        Returns:
            Portfolio VaR
        """
        if self.positions:
            # Get weights
            allocation = self.get_position_allocation()
            weights = np.array([allocation.get(s, 0) for s in returns_data.columns])
            
            # Portfolio returns
            portfolio_returns = (returns_data * weights).sum(axis=1)
            
            # Calculate VaR
            var_percentile = (1 - confidence) * 100
            var = np.percentile(portfolio_returns, var_percentile)
            
            return abs(var * self.total_value)
        
        return 0
    
    def get_beta_to_market(self,
                          market_returns: pd.Series,
                          portfolio_returns: Optional[pd.Series] = None) -> float:
        """
        Calculate portfolio beta to market
        
        Args:
            market_returns: Market returns series
            portfolio_returns: Portfolio returns (calculated if not provided)
            
        Returns:
            Beta coefficient
        """
        if portfolio_returns is None:
            portfolio_returns = self._get_returns_series()
        
        if len(portfolio_returns) < 20 or len(market_returns) < 20:
            return 1.0
        
        # Align series
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 20:
            return 1.0
        
        # Calculate beta
        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        market_variance = aligned.iloc[:, 1].var()
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    # ================== UTILITY METHODS ==================
    
    def _record_transaction(self, transaction: Dict) -> None:
        """Record transaction in history"""
        self.transaction_history.append(transaction)
        
        # Keep only last 10000 transactions
        if len(self.transaction_history) > 10000:
            self.transaction_history = self.transaction_history[-10000:]
    
    def _take_snapshot(self) -> None:
        """Take portfolio snapshot"""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=self.total_value,
            cash_balance=self.cash_balance,
            positions_value=sum(p.market_value for p in self.positions.values()),
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            n_positions=len(self.positions),
            exposure=dict(self.exposure_by_asset),
            allocation=self.get_position_allocation()
        )
        
        self.portfolio_history.append(snapshot)
        
        # Calculate daily return if new day
        if self.portfolio_history:
            last_snapshot = self.portfolio_history[-2] if len(self.portfolio_history) > 1 else self.portfolio_history[-1]
            if snapshot.timestamp.date() != last_snapshot.timestamp.date():
                daily_return = (snapshot.total_value - last_snapshot.total_value) / last_snapshot.total_value
                self.daily_returns.append(daily_return)
    
    def _get_returns_series(self, period_days: Optional[int] = None) -> pd.Series:
        """Get returns series for analysis"""
        if not self.portfolio_history:
            return pd.Series()
        
        # Convert snapshots to values
        values = [s.total_value for s in self.portfolio_history]
        timestamps = [s.timestamp for s in self.portfolio_history]
        
        # Create series
        series = pd.Series(values, index=timestamps)
        
        # Filter by period if specified
        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            series = series[series.index >= cutoff]
        
        # Calculate returns
        returns = series.pct_change().dropna()
        
        return returns
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary
        
        Returns:
            Comprehensive portfolio summary
        """
        metrics = self.calculate_performance_metrics()
        exposure = self.get_exposure_summary()
        
        return {
            'value': {
                'total': self.total_value,
                'cash': self.cash_balance,
                'positions': self.total_value - self.cash_balance,
                'initial_capital': self.initial_capital
            },
            'pnl': {
                'unrealized': self.unrealized_pnl,
                'realized': self.realized_pnl,
                'total': self.total_pnl,
                'total_return': self.total_return
            },
            'positions': {
                'count': len(self.positions),
                'symbols': list(self.positions.keys()),
                'allocation': self.get_position_allocation()
            },
            'performance': {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor
            },
            'exposure': exposure,
            'trades': {
                'total': len(self.closed_positions),
                'today': sum(1 for t in self.closed_positions 
                           if t['exit_time'].date() == datetime.now().date())
            }
        }
    
    def export_history(self, filepath=None):
        """Export portfolio history to file"""
        data = {
            'transactions': self.transaction_history,
            'closed_positions': self.closed_positions,
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'total_value': s.total_value,
                    'cash': s.cash_balance,
                    'unrealized_pnl': s.unrealized_pnl,
                    'realized_pnl': s.realized_pnl
                }
                for s in self.portfolio_history
            ],
            'summary': self.get_summary()
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Portfolio history exported to {filepath}")

        return data

    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as a pandas Series
        
        Returns:
            Series with equity values over time
        """
        # Check if we have any history to export
        history = self.export_history()
        
        if not history or 'portfolio_value' not in history:
            # Return current value if no history
            return pd.Series([self.total_value])
        
        # Extract portfolio values over time
        portfolio_values = history['portfolio_value']
        
        if isinstance(portfolio_values, list):
            # If it's a list of values
            if len(portfolio_values) > 0:
                # Check if first element is a dict with timestamp
                if isinstance(portfolio_values[0], dict) and 'timestamp' in portfolio_values[0]:
                    timestamps = [v['timestamp'] for v in portfolio_values]
                    values = [v['value'] for v in portfolio_values]
                    return pd.Series(values, index=timestamps)
                else:
                    # Just a list of values
                    return pd.Series(portfolio_values)
        elif isinstance(portfolio_values, pd.Series):
            return portfolio_values
        else:
            # Fallback - return current value
            return pd.Series([self.total_value])


# Example usage and testing
if __name__ == "__main__":
    # Initialize portfolio
    portfolio = Portfolio(
        initial_capital=10000,
        max_positions=5,
        enable_short=False
    )
    
    print("=== Portfolio Manager Test ===\n")
    
    # Open some positions
    print("1. Opening positions:")
    portfolio.open_position('BTC', 0.01, 45000, 'long', stop_loss=43000, fees=10)
    portfolio.open_position('ETH', 1, 3000, 'long', take_profit=3500, fees=5)
    portfolio.open_position('SOL', 10, 150, 'long', fees=2)
    
    # Show allocation
    allocation = portfolio.get_position_allocation()
    print(f"\n2. Current allocation:")
    for asset, weight in allocation.items():
        print(f"   {asset}: {weight:.2%}")
    
    # Update prices
    print("\n3. Updating prices:")
    portfolio.update_all_prices({'BTC': 46000, 'ETH': 3100, 'SOL': 155})
    
    # Show unrealized P&L
    print(f"\n4. Unrealized P&L: ${portfolio.unrealized_pnl:.2f}")
    for symbol, pos in portfolio.positions.items():
        print(f"   {symbol}: ${pos.unrealized_pnl:.2f} ({pos.unrealized_return:.2%})")
    
    # Close a position
    print("\n5. Closing ETH position:")
    result = portfolio.close_position('ETH', 3150, fees=5)
    if result:
        print(f"   P&L: ${result['net_pnl']:.2f} ({result['return_pct']:.2%})")
    
    # Calculate metrics
    print("\n6. Performance metrics:")
    metrics = portfolio.calculate_performance_metrics()
    print(f"   Total Return: {metrics.total_return:.2%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Win Rate: {metrics.win_rate:.2%}")
    
    # Get summary
    print("\n7. Portfolio Summary:")
    summary = portfolio.get_summary()
    print(f"   Total Value: ${summary['value']['total']:.2f}")
    print(f"   Total P&L: ${summary['pnl']['total']:.2f}")
    print(f"   Positions: {summary['positions']['count']}")
    
    print("\n Portfolio Manager ready for integration!")