"""
Risk Manager Module
Comprehensive risk management system for trading bot
Implements position sizing, risk limits, and capital protection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    current_drawdown: float
    max_drawdown: float
    value_at_risk: float
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    margin_usage: float


@dataclass
class PositionLimits:
    """Position size limits and constraints"""
    max_position_size: float
    min_position_size: float
    max_positions: int
    max_correlation: float
    max_sector_exposure: float
    max_single_loss: float


@dataclass
class RiskConfig:
    """Risk management configuration"""
    # Risk levels
    risk_level: RiskLevel = RiskLevel.MODERATE
    max_risk_per_trade: float = 0.02  # 2% per trade
    max_daily_risk: float = 0.06  # 6% daily loss limit
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    
    # Position sizing
    use_kelly_criterion: bool = True
    kelly_fraction: float = 0.25  # Use 25% of Kelly
    position_sizing_method: str = 'volatility_based'  # 'fixed', 'volatility_based', 'kelly', 'optimal_f'
    
    # Volatility parameters
    volatility_lookback: int = 20
    volatility_target: float = 0.15  # 15% annual target volatility
    
    # Stop loss and take profit
    use_stops: bool = True
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_stop_activation: float = 0.02  # Activate at 2% profit
    trailing_stop_distance: float = 0.01  # 1% trailing distance
    
    # Circuit breakers
    enable_circuit_breakers: bool = True
    consecutive_losses_limit: int = 5
    daily_trades_limit: int = 20
    hourly_trades_limit: int = 5
    
    # Correlation and diversification
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.7
    min_time_between_trades: int = 60  # seconds
    
    # Margin and leverage
    max_leverage: float = 1.0  # No leverage by default
    margin_call_level: float = 0.3  # 30% margin level
    
    # Recovery mode
    enable_recovery_mode: bool = True
    recovery_mode_threshold: float = 0.1  # Enter recovery at 10% drawdown
    recovery_position_reduction: float = 0.5  # Reduce positions by 50%


class RiskManager:
    """
    Comprehensive risk management system
    
    Features:
    - Multiple position sizing algorithms
    - Dynamic risk adjustment
    - Stop loss and take profit management
    - Circuit breakers and limits
    - Drawdown control
    - Correlation risk management
    - Recovery mode
    """
    
    def __init__(self, 
                initial_capital: float,
                config: Optional[RiskConfig] = None):
        """
        Initialize Risk Manager
        
        Args:
            initial_capital: Starting capital
            config: Risk configuration
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config = config or RiskConfig()  # Creates default config if None
        
        # Track positions and exposure
        self.open_positions = {}
        self.position_history = []
        self.total_exposure = 0
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.daily_trades = 0
        self.hourly_trades = []
        
        # Risk metrics
        self.risk_metrics = self._initialize_risk_metrics()
        self.position_limits = self._calculate_position_limits()
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.recovery_mode_active = False
        self.last_trade_time = None
        
        # Historical data for calculations
        self.returns_history = []
        self.volatility_history = []
        self.correlation_matrix = pd.DataFrame()
        
        # FIXED: Use self.config instead of config
        logger.info(f"RiskManager initialized with {initial_capital} capital, risk level: {self.config.risk_level.value}")
    
    def _initialize_risk_metrics(self) -> RiskMetrics:
        """Initialize risk metrics"""
        return RiskMetrics(
            current_drawdown=0,
            max_drawdown=0,
            value_at_risk=0,
            expected_shortfall=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            win_rate=0.5,
            profit_factor=1.0,
            avg_win_loss_ratio=1.0,
            correlation_risk=0,
            concentration_risk=0,
            leverage_ratio=0,
            margin_usage=0
        )
    
    def _calculate_position_limits(self) -> PositionLimits:
        """Calculate position limits based on risk level"""
        if self.config.risk_level == RiskLevel.CONSERVATIVE:
            return PositionLimits(
                max_position_size=self.current_capital * 0.1,
                min_position_size=self.current_capital * 0.001,
                max_positions=5,
                max_correlation=0.5,
                max_sector_exposure=0.3,
                max_single_loss=self.current_capital * 0.02
            )
        elif self.config.risk_level == RiskLevel.MODERATE:
            return PositionLimits(
                max_position_size=self.current_capital * 0.2,
                min_position_size=self.current_capital * 0.001,
                max_positions=10,
                max_correlation=0.7,
                max_sector_exposure=0.5,
                max_single_loss=self.current_capital * 0.05
            )
        elif self.config.risk_level == RiskLevel.AGGRESSIVE:
            return PositionLimits(
                max_position_size=self.current_capital * 0.3,
                min_position_size=self.current_capital * 0.001,
                max_positions=15,
                max_correlation=0.9,
                max_sector_exposure=0.7,
                max_single_loss=self.current_capital * 0.1
            )
        else:  # CUSTOM
            return PositionLimits(
                max_position_size=self.current_capital * 0.25,
                min_position_size=self.current_capital * 0.001,
                max_positions=10,
                max_correlation=0.7,
                max_sector_exposure=0.5,
                max_single_loss=self.current_capital * self.config.max_risk_per_trade
            )
    
    # ================== POSITION SIZING ALGORITHMS ==================
    
    def calculate_position_size(self,
                               symbol: str,
                               signal_strength: float,
                               current_price: float,
                               volatility: float,
                               win_rate: Optional[float] = None,
                               avg_win: Optional[float] = None,
                               avg_loss: Optional[float] = None,
                               method: Optional[str] = None) -> float:
        """
        Calculate position size using specified method
        
        Args:
            symbol: Trading symbol
            signal_strength: Strength of trading signal (0-1)
            current_price: Current asset price
            volatility: Asset volatility
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            method: Override default method
            
        Returns:
            Position size in units
        """
        # Check if we can trade
        if not self.can_open_position(symbol):
            return 0
        
        method = method or self.config.position_sizing_method
        
        # Calculate base position value
        if method == 'fixed':
            position_value = self._fixed_position_size()
        elif method == 'volatility_based':
            position_value = self._volatility_based_size(volatility)
        elif method == 'kelly':
            position_value = self._kelly_criterion_size(win_rate, avg_win, avg_loss)
        elif method == 'optimal_f':
            position_value = self._optimal_f_size(win_rate, avg_win, avg_loss)
        elif method == 'risk_parity':
            position_value = self._risk_parity_size(volatility)
        else:
            position_value = self._fixed_position_size()
        
        # Adjust for signal strength
        position_value *= signal_strength
        
        # Apply risk adjustments
        position_value = self._apply_risk_adjustments(position_value)
        
        # Convert to units
        position_size = position_value / current_price
        
        # Apply limits
        position_size = self._apply_position_limits(position_size, current_price)
        
        return position_size
    
    def _fixed_position_size(self) -> float:
        """Fixed percentage of capital"""
        return self.current_capital * self.config.max_risk_per_trade
    
    def _volatility_based_size(self, volatility: float) -> float:
        """
        Position size based on volatility targeting
        Lower volatility = larger position, higher volatility = smaller position
        """
        if volatility <= 0:
            return 0
        
        # Use a base volatility for comparison (e.g., 2% daily vol)
        base_volatility = 0.02
        
        # Scale position inversely with volatility ratio
        # If volatility is half of base, position is 2x
        # If volatility is 2x base, position is 0.5x
        vol_ratio = base_volatility / volatility
        
        # Apply reasonable bounds
        vol_adjustment = max(0.5, min(vol_ratio, 2.0))
        
        # Calculate position value
        base_position = self.current_capital * self.config.max_risk_per_trade
        
        return base_position * vol_adjustment
    
    def _kelly_criterion_size(self,
                             win_rate: Optional[float],
                             avg_win: Optional[float],
                             avg_loss: Optional[float]) -> float:
        """
        Kelly Criterion for optimal position sizing
        f* = (p*b - q) / b
        where:
        f* = fraction of capital to bet
        p = probability of win
        q = probability of loss (1-p)
        b = ratio of win to loss
        """
        if not all([win_rate, avg_win, avg_loss]) or avg_loss == 0:
            return self._fixed_position_size()
        
        p = win_rate
        q = 1 - win_rate
        b = abs(avg_win / avg_loss)
        
        # Kelly formula
        kelly_fraction = (p * b - q) / b if b > 0 else 0
        
        # Apply safety factor (use fraction of Kelly)
        kelly_fraction *= self.config.kelly_fraction
        
        # Ensure positive and reasonable
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        return self.current_capital * kelly_fraction
    
    def _optimal_f_size(self,
                       win_rate: Optional[float],
                       avg_win: Optional[float],
                       avg_loss: Optional[float]) -> float:
        """
        Optimal f position sizing (Ralph Vince method)
        More aggressive than Kelly
        """
        if not all([win_rate, avg_win, avg_loss]) or avg_loss == 0:
            return self._fixed_position_size()
        
        # Simplified optimal f calculation
        # In practice, this requires historical trade data
        win_loss_ratio = abs(avg_win / avg_loss)
        
        # Approximate optimal f
        optimal_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        optimal_f = max(0, min(optimal_f, 0.25))  # Cap at 25%
        
        # Apply safety factor
        optimal_f *= 0.5  # Use 50% of optimal f
        
        return self.current_capital * optimal_f
    
    def _risk_parity_size(self, volatility: float) -> float:
        """
        Risk parity position sizing
        Equal risk contribution from each position
        """
        if volatility <= 0:
            return 0
        
        # Number of current positions
        n_positions = len(self.open_positions) + 1
        
        # Equal risk budget for each position
        risk_budget = self.config.max_risk_per_trade / n_positions
        
        # Position size inversely proportional to volatility
        position_value = (risk_budget * self.current_capital) / volatility
        
        return position_value
    
    def _apply_risk_adjustments(self, position_value: float) -> float:
        """
        Apply risk adjustments based on current market conditions
        
        Args:
            position_value: Base position value
            
        Returns:
            Adjusted position value
        """
        adjustment_factor = 1.0
        
        # Drawdown adjustment
        if self.current_drawdown > 0.05:  # 5% drawdown
            # Reduce position size proportionally to drawdown
            adjustment_factor *= (1 - self.current_drawdown)
        
        # Recovery mode adjustment
        if self.recovery_mode_active:
            adjustment_factor *= self.config.recovery_position_reduction
        
        # Consecutive losses adjustment
        if self.consecutive_losses > 2:
            # Reduce by 20% for each loss after 2
            adjustment_factor *= (0.8 ** (self.consecutive_losses - 2))
        
        # Volatility spike adjustment
        if len(self.volatility_history) > 20:
            recent_vol = np.mean(self.volatility_history[-5:])
            avg_vol = np.mean(self.volatility_history[-20:])
            if recent_vol > avg_vol * 1.5:  # 50% spike
                adjustment_factor *= 0.7
        
        # Circuit breaker
        if self.circuit_breaker_active:
            adjustment_factor = 0  # No new positions
        
        return position_value * adjustment_factor
    
    def _apply_position_limits(self, position_size: float, price: float) -> float:
        """
        Apply position limits and constraints
        
        Args:
            position_size: Calculated position size
            price: Asset price
            
        Returns:
            Limited position size
        """
        position_value = position_size * price
        
        # Maximum position size
        if position_value > self.position_limits.max_position_size:
            position_size = self.position_limits.max_position_size / price
        
        # Minimum position size
        if position_value < self.position_limits.min_position_size:
            return 0  # Too small, don't trade
        
        # Check total exposure
        new_exposure = self.total_exposure + position_value
        if new_exposure > self.current_capital * self.config.max_leverage:
            # Reduce position to stay within leverage
            max_additional = self.current_capital * self.config.max_leverage - self.total_exposure
            if max_additional <= 0:
                return 0
            position_size = max_additional / price
        
        return position_size
    
    # ================== STOP LOSS & TAKE PROFIT ==================
    
    def calculate_stop_loss(self,
                           entry_price: float,
                           position_type: str,
                           atr: Optional[float] = None,
                           method: str = 'atr') -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Position entry price
            position_type: 'long' or 'short'
            atr: Average True Range
            method: 'atr', 'percentage', or 'fixed'
            
        Returns:
            Stop loss price
        """
        if not self.config.use_stops:
            return 0
        
        if method == 'atr' and atr:
            stop_distance = atr * self.config.stop_loss_atr_multiplier
            if position_type == 'long':
                return entry_price - stop_distance
            else:  # short
                return entry_price + stop_distance
                
        elif method == 'percentage':
            stop_percentage = self.config.max_risk_per_trade
            if position_type == 'long':
                return entry_price * (1 - stop_percentage)
            else:  # short
                return entry_price * (1 + stop_percentage)
                
        else:  # fixed
            if position_type == 'long':
                return entry_price * 0.98  # 2% stop
            else:
                return entry_price * 1.02
    
    def calculate_take_profit(self,
                             entry_price: float,
                             position_type: str,
                             atr: Optional[float] = None,
                             risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Position entry price
            position_type: 'long' or 'short'
            atr: Average True Range
            risk_reward_ratio: Risk/reward ratio
            
        Returns:
            Take profit price
        """
        if not self.config.use_stops:
            return 0
        
        if atr:
            profit_distance = atr * self.config.take_profit_atr_multiplier
        else:
            # Use risk/reward ratio
            stop_loss = self.calculate_stop_loss(entry_price, position_type, atr)
            stop_distance = abs(entry_price - stop_loss)
            profit_distance = stop_distance * risk_reward_ratio
        
        if position_type == 'long':
            return entry_price + profit_distance
        else:  # short
            return entry_price - profit_distance
    
    def update_trailing_stop(self,
                            position_id: str,
                            current_price: float,
                            highest_price: float,
                            entry_price: float,
                            position_type: str) -> Optional[float]:
        """
        Update trailing stop loss
        
        Args:
            position_id: Position identifier
            current_price: Current market price
            highest_price: Highest price since entry (for long)
            entry_price: Position entry price
            position_type: 'long' or 'short'
            
        Returns:
            New stop loss price if updated
        """
        if position_type == 'long':
            # Check if we should activate trailing stop
            profit_pct = (highest_price - entry_price) / entry_price
            
            if profit_pct >= self.config.trailing_stop_activation:
                # Calculate new trailing stop
                new_stop = highest_price * (1 - self.config.trailing_stop_distance)
                
                # Only update if higher than current stop
                if position_id in self.open_positions:
                    current_stop = self.open_positions[position_id].get('stop_loss', 0)
                    if new_stop > current_stop:
                        return new_stop
        
        else:  # short
            # For short positions, trail on lowest price
            profit_pct = (entry_price - current_price) / entry_price
            
            if profit_pct >= self.config.trailing_stop_activation:
                new_stop = current_price * (1 + self.config.trailing_stop_distance)
                
                if position_id in self.open_positions:
                    current_stop = self.open_positions[position_id].get('stop_loss', float('inf'))
                    if new_stop < current_stop:
                        return new_stop
        
        return None
    
    # ================== RISK MONITORING ==================
    
    def update_capital(self, new_capital: float) -> None:
        """Update current capital and related metrics"""
        self.current_capital = new_capital
        
        # Update drawdown
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - new_capital) / self.peak_capital
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Check for recovery mode
        if self.config.enable_recovery_mode:
            if self.current_drawdown >= self.config.recovery_mode_threshold:
                if not self.recovery_mode_active:
                    logger.warning(f"Entering recovery mode at {self.current_drawdown:.2%} drawdown")
                    self.recovery_mode_active = True
            elif self.recovery_mode_active and self.current_drawdown < self.config.recovery_mode_threshold * 0.5:
                logger.info("Exiting recovery mode")
                self.recovery_mode_active = False
        
        # Update position limits
        self.position_limits = self._calculate_position_limits()
    
    def can_open_position(self, symbol: str) -> bool:
        """
        Check if we can open a new position
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position can be opened
        """
        # Circuit breaker check
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active - cannot open position")
            return False
        
        # Maximum positions check
        if len(self.open_positions) >= self.position_limits.max_positions:
            logger.warning(f"Maximum positions ({self.position_limits.max_positions}) reached")
            return False
        
        # Daily loss limit check
        if self.daily_pnl <= -self.config.max_daily_risk * self.initial_capital:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            return False
        
        # Drawdown limit check
        if self.current_drawdown >= self.config.max_drawdown_limit:
            logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
            return False
        
        # Consecutive losses check
        if self.consecutive_losses >= self.config.consecutive_losses_limit:
            logger.warning(f"Consecutive losses limit reached: {self.consecutive_losses}")
            self.activate_circuit_breaker("consecutive_losses")
            return False
        
        # Time between trades check
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.config.min_time_between_trades:
                logger.warning(f"Too soon since last trade: {time_since_last:.0f}s")
                return False
        
        # Daily trades limit
        if self.daily_trades >= self.config.daily_trades_limit:
            logger.warning(f"Daily trades limit reached: {self.daily_trades}")
            return False
        
        # Hourly trades limit
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        hourly_count = sum(1 for t in self.hourly_trades if t >= current_hour)
        if hourly_count >= self.config.hourly_trades_limit:
            logger.warning(f"Hourly trades limit reached: {hourly_count}")
            return False
        
        # Check if symbol already has position
        if symbol in self.open_positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        return True
    
    def activate_circuit_breaker(self, reason: str) -> None:
        """
        Activate circuit breaker
        
        Args:
            reason: Reason for activation
        """
        self.circuit_breaker_active = True
        logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        
        # Close all positions or reduce exposure
        if self.config.enable_circuit_breakers:
            logger.info("Initiating emergency position reduction")
            # In practice, this would trigger position closures
    
    def deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker"""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker deactivated")
    
    
    def record_trade(self,
                    symbol: str,
                    pnl: float,
                    position_size: float,
                    entry_price: float,
                    exit_price: float,
                    position_type: str) -> None:
        """
        Record completed trade for risk tracking
        
        Args:
            symbol: Trading symbol
            pnl: Profit/loss from trade
            position_size: Size of position
            entry_price: Entry price
            exit_price: Exit price
            position_type: 'long' or 'short'
        """
        # Calculate return with zero check
        if position_size > 0 and entry_price > 0:
            trade_return = pnl / (position_size * entry_price)
        else:
            # If position_size or entry_price is 0, return is 0
            trade_return = 0.0
            # Log warning if this happens
            if position_size == 0:
                print(f"Warning: Zero position size recorded for {symbol}")
            if entry_price == 0:
                print(f"Warning: Zero entry price recorded for {symbol}")
        
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'pnl': pnl,
            'position_size': position_size,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_type': position_type,
            'return': trade_return  # Use the calculated return with zero check
        }
        
        self.position_history.append(trade_data)
        
        # Update metrics
        self.daily_pnl += pnl
        self.daily_trades += 1
        self.hourly_trades.append(datetime.now())
        self.last_trade_time = datetime.now()
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update returns history
        self.returns_history.append(trade_data['return'])
        
        # Clean old hourly trades
        cutoff = datetime.now() - timedelta(hours=1)
        self.hourly_trades = [t for t in self.hourly_trades if t > cutoff]
    
    def calculate_var(self, confidence_level: float = 0.95, periods: int = 252) -> float:
        """
        Calculate Value at Risk
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            periods: Number of periods for calculation
            
        Returns:
            VaR value
        """
        if len(self.returns_history) < periods:
            return 0
        
        recent_returns = self.returns_history[-periods:]
        var_value = np.percentile(recent_returns, (1 - confidence_level) * 100)
        
        return abs(var_value) * self.current_capital
    
    def calculate_expected_shortfall(self, confidence_level: float = 0.95, periods: int = 252) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            confidence_level: Confidence level
            periods: Number of periods
            
        Returns:
            Expected shortfall value
        """
        if len(self.returns_history) < periods:
            return 0
        
        recent_returns = self.returns_history[-periods:]
        var_threshold = np.percentile(recent_returns, (1 - confidence_level) * 100)
        
        # Average of returns worse than VaR
        tail_returns = [r for r in recent_returns if r <= var_threshold]
        if tail_returns:
            es = abs(np.mean(tail_returns)) * self.current_capital
        else:
            es = 0
        
        return es
    
    def get_risk_metrics(self) -> RiskMetrics:
        """
        Calculate and return current risk metrics
        
        Returns:
            Current risk metrics
        """
        # Update basic metrics
        self.risk_metrics.current_drawdown = self.current_drawdown
        self.risk_metrics.max_drawdown = self.max_drawdown
        
        # Calculate VaR and ES
        self.risk_metrics.value_at_risk = self.calculate_var()
        self.risk_metrics.expected_shortfall = self.calculate_expected_shortfall()
        
        # Calculate performance ratios
        if len(self.returns_history) > 20:
            returns = pd.Series(self.returns_history[-252:])  # Last year
            
            # Sharpe ratio
            if returns.std() > 0:
                self.risk_metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                self.risk_metrics.sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
            
            # Calmar ratio
            if self.max_drawdown > 0:
                annual_return = (1 + returns.mean()) ** 252 - 1
                self.risk_metrics.calmar_ratio = annual_return / self.max_drawdown
        
        # Win rate and profit factor
        if self.position_history:
            wins = [t for t in self.position_history if t['pnl'] > 0]
            losses = [t for t in self.position_history if t['pnl'] < 0]
            
            self.risk_metrics.win_rate = len(wins) / len(self.position_history)
            
            if losses:
                total_wins = sum(t['pnl'] for t in wins)
                total_losses = abs(sum(t['pnl'] for t in losses))
                if total_losses > 0:
                    self.risk_metrics.profit_factor = total_wins / total_losses
                
                avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
                avg_loss = abs(np.mean([t['pnl'] for t in losses]))
                if avg_loss > 0:
                    self.risk_metrics.avg_win_loss_ratio = avg_win / avg_loss
        
        # Leverage and margin
        if self.current_capital > 0:
            self.risk_metrics.leverage_ratio = self.total_exposure / self.current_capital
            self.risk_metrics.margin_usage = self.total_exposure / (self.current_capital * self.config.max_leverage)
        
        return self.risk_metrics
    
    def reset_daily_metrics(self) -> None:
        """Reset daily tracking metrics (call at start of each day)"""
        self.daily_pnl = 0
        self.daily_trades = 0
        logger.info("Daily metrics reset")
    
    def get_position_info(self, symbol: str) -> Optional[Dict]:
        """Get information about open position"""
        return self.open_positions.get(symbol)
    
    def update_position(self,
                       symbol: str,
                       current_price: float,
                       highest_price: Optional[float] = None,
                       lowest_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Update position information and check for exits
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            
        Returns:
            Dictionary with action recommendations
        """
        if symbol not in self.open_positions:
            return {'action': 'none'}
        
        position = self.open_positions[symbol]
        recommendations = {'action': 'none'}
        
        # Check stop loss
        if position.get('stop_loss'):
            if position['type'] == 'long' and current_price <= position['stop_loss']:
                recommendations['action'] = 'close'
                recommendations['reason'] = 'stop_loss'
            elif position['type'] == 'short' and current_price >= position['stop_loss']:
                recommendations['action'] = 'close'
                recommendations['reason'] = 'stop_loss'
        
        # Check take profit
        if position.get('take_profit'):
            if position['type'] == 'long' and current_price >= position['take_profit']:
                recommendations['action'] = 'close'
                recommendations['reason'] = 'take_profit'
            elif position['type'] == 'short' and current_price <= position['take_profit']:
                recommendations['action'] = 'close'
                recommendations['reason'] = 'take_profit'
        
        # Update trailing stop
        if highest_price or lowest_price:
            new_stop = self.update_trailing_stop(
                symbol,
                current_price,
                highest_price if position['type'] == 'long' else lowest_price,
                position['entry_price'],
                position['type']
            )
            if new_stop:
                position['stop_loss'] = new_stop
                recommendations['new_stop'] = new_stop
        
        return recommendations
    
    def add_position(self,
                    symbol: str,
                    position_size: float,
                    entry_price: float,
                    position_type: str,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None) -> None:
        """
        Add new position to tracking
        
        Args:
            symbol: Trading symbol
            position_size: Size of position
            entry_price: Entry price
            position_type: 'long' or 'short'
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        self.open_positions[symbol] = {
            'size': position_size,
            'entry_price': entry_price,
            'type': position_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'highest_price': entry_price,
            'lowest_price': entry_price
        }
        
        # Update exposure
        self.total_exposure += position_size * entry_price
        logger.info(f"Position added: {symbol} {position_type} {position_size} @ {entry_price}")
    
    def remove_position(self, symbol: str, exit_price: float) -> Dict[str, float]:
        """
        Remove position and calculate P&L
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            
        Returns:
            Dictionary with P&L information
        """
        if symbol not in self.open_positions:
            return {'pnl': 0, 'return': 0}
        
        position = self.open_positions[symbol]
        
        # Calculate P&L
        if position['type'] == 'long':
            pnl = (exit_price - position['entry_price']) * position['size']
        else:  # short
            pnl = (position['entry_price'] - exit_price) * position['size']
        
        # Calculate return
        position_value = position['size'] * position['entry_price']
        return_pct = pnl / position_value if position_value > 0 else 0
        
        # Record trade
        self.record_trade(
            symbol=symbol,
            pnl=pnl,
            position_size=position['size'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            position_type=position['type']
        )
        
        # Update exposure
        self.total_exposure -= position_value
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        logger.info(f"Position closed: {symbol} P&L: {pnl:.2f} Return: {return_pct:.2%}")
        
        return {'pnl': pnl, 'return': return_pct}


# Example usage and testing
if __name__ == "__main__":
    # Initialize risk manager
    risk_manager = RiskManager(
        initial_capital=10000,
        config=RiskConfig(
            risk_level=RiskLevel.MODERATE,
            max_risk_per_trade=0.02,
            position_sizing_method='volatility_based'
        )
    )
    
    # Test position sizing
    print("=== Position Sizing Test ===")
    position_size = risk_manager.calculate_position_size(
        symbol='BTC/USDT',
        signal_strength=0.8,
        current_price=45000,
        volatility=0.02,
        win_rate=0.55,
        avg_win=200,
        avg_loss=150
    )
    print(f"Recommended position size: {position_size:.6f} BTC")
    print(f"Position value: ${position_size * 45000:.2f}")
    
    # Test stop loss and take profit
    print("\n=== Stop Loss & Take Profit ===")
    stop_loss = risk_manager.calculate_stop_loss(45000, 'long', atr=500)
    take_profit = risk_manager.calculate_take_profit(45000, 'long', atr=500)
    print(f"Entry: $45,000")
    print(f"Stop Loss: ${stop_loss:.2f}")
    print(f"Take Profit: ${take_profit:.2f}")
    
    # Add a position
    risk_manager.add_position(
        symbol='BTC/USDT',
        position_size=position_size,
        entry_price=45000,
        position_type='long',
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    
    # Update position
    recommendations = risk_manager.update_position(
        'BTC/USDT',
        current_price=46000,
        highest_price=46500
    )
    print(f"\n=== Position Update ===")
    print(f"Recommendations: {recommendations}")
    
    # Get risk metrics
    metrics = risk_manager.get_risk_metrics()
    print(f"\n=== Risk Metrics ===")
    print(f"Current Drawdown: {metrics.current_drawdown:.2%}")
    print(f"VaR (95%): ${metrics.value_at_risk:.2f}")
    print(f"Leverage: {metrics.leverage_ratio:.2f}x")
    
    print("\n Risk Manager ready for integration!")