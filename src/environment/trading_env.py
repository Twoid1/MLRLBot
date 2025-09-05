"""
Trading Environment Module
Custom Gym-like environment for cryptocurrency trading
Built from scratch with no external dependencies
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from enum import IntEnum
import warnings
warnings.filterwarnings('ignore')


class Actions(IntEnum):
    """Trading actions"""
    HOLD = 0
    BUY = 1
    SELL = 2


class Positions(IntEnum):
    """Position states"""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class TradingState:
    """Container for environment state"""
    step: int
    position: int
    entry_price: float
    balance: float
    equity: float
    portfolio_value: float
    position_size: float
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_drawdown: float
    max_drawdown: float
    sharpe_ratio: float


class TradingEnvironment:
    """
    Custom trading environment for cryptocurrency trading
    Supports both long and short positions with realistic fee modeling
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 10000,
                 leverage: float = 1.0,
                 fee_rate: float = 0.0026,  # Kraken's taker fee
                 slippage: float = 0.001,
                 position_sizing: str = 'fixed',
                 risk_per_trade: float = 0.02,
                 reward_scaling: float = 1.0,
                 window_size: int = 50,
                 enable_short: bool = False,
                 stop_loss: Optional[float] = None,
                 take_profit: Optional[float] = None,
                 features_df: Optional[pd.DataFrame] = None):
        """
        Initialize trading environment
        
        Args:
            df: DataFrame with OHLCV data
            initial_balance: Starting capital
            leverage: Maximum leverage (1.0 = no leverage)
            fee_rate: Trading fee rate
            slippage: Slippage rate
            position_sizing: 'fixed' or 'dynamic'
            risk_per_trade: Risk per trade for dynamic sizing
            reward_scaling: Scaling factor for rewards
            window_size: Lookback window for state
            enable_short: Allow short positions
            stop_loss: Stop loss percentage (e.g., 0.05 for 5%)
            take_profit: Take profit percentage
            features_df: Pre-calculated features DataFrame
        """
        # FIXED: Set window_size BEFORE validation
        self.window_size = window_size
        
        # Validate input data (now window_size is available)
        self._validate_data(df)
        
        # Market data
        self.df = df.copy()
        self.features_df = features_df
        self.prices = df[['open', 'high', 'low', 'close']].values
        self.volumes = df['volume'].values
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.reward_scaling = reward_scaling
        self.enable_short = enable_short
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # State tracking
        self.current_step = 0
        self.position = Positions.FLAT
        self.entry_price = 0
        self.position_size = 0
        
        # Account tracking
        self.balance = initial_balance
        self.equity = initial_balance
        self.portfolio_values = []
        
        # Performance tracking
        self.trades = []
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_fees_paid = 0
        
        # Risk metrics
        self.peak_portfolio_value = initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0
        
        # Episode tracking
        self.done = False
        self.truncated = False
        
        # Action and observation spaces (Gym-like)
        self.action_space_n = 3  # HOLD, BUY, SELL
        self.observation_space_shape = self._get_observation_shape()
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < self.window_size:
            raise ValueError(f"DataFrame has {len(df)} rows, less than window_size {self.window_size}")
    
    def _get_observation_shape(self) -> Tuple[int]:
        """Get shape of observation space"""
        # Market features + account features + position features
        market_features = self.window_size * 5  # OHLCV
        
        if self.features_df is not None:
            technical_features = self.features_df.shape[1]
        else:
            technical_features = 0
        
        account_features = 10  # balance, equity, pnl, etc.
        position_features = 5  # position, entry_price, size, etc.
        
        total_features = market_features + technical_features + account_features + position_features
        return (total_features,)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset step counter
        self.current_step = self.window_size
        
        # Reset position
        self.position = Positions.FLAT
        self.entry_price = 0
        self.position_size = 0
        
        # Reset account
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        
        # Reset performance tracking
        self.trades = []
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        self.total_fees_paid = 0
        
        # Reset risk metrics
        self.peak_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0
        
        # Reset episode flags
        self.done = False
        self.truncated = False
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Trading action (0=HOLD, 1=BUY, 2=SELL)
            
        Returns:
            observation: Current state
            reward: Step reward
            done: Episode finished
            truncated: Episode truncated
            info: Additional information
        """
        # Store previous portfolio value
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_action(action)
        
        # Update market step
        self.current_step += 1
        
        # Check if episode is done
        self._check_done()
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Get new observation
        observation = self._get_observation()
        
        # Update performance metrics
        self._update_metrics()
        
        # Create info dictionary
        info = self._get_info()
        
        return observation, reward, self.done, self.truncated, info
    
    def _execute_action(self, action: int) -> None:
        """Execute trading action"""
        current_price = self._get_current_price()
        
        if action == Actions.BUY:
            self._execute_buy(current_price)
        elif action == Actions.SELL:
            self._execute_sell(current_price)
        # HOLD action doesn't change position
        
        # Check stop loss and take profit
        if self.position != Positions.FLAT:
            self._check_exit_conditions(current_price)
    
    def _execute_buy(self, price: float) -> None:
        """Execute buy order"""
        if self.position == Positions.FLAT:
            # Open long position
            position_value = self._calculate_position_size(price)
            
            # Apply slippage and fees
            execution_price = price * (1 + self.slippage)
            fees = position_value * self.fee_rate
            
            # Check if we have enough balance
            if self.balance >= fees:
                self.position_size = (position_value - fees) / execution_price
                self.entry_price = execution_price
                self.position = Positions.LONG
                self.balance -= fees
                self.total_fees_paid += fees
                
                # Record trade
                self._record_trade('BUY', execution_price, self.position_size, fees)
                
        elif self.position == Positions.SHORT and self.enable_short:
            # Close short position (buy to cover)
            self._close_position(price, 'BUY')
    
    def _execute_sell(self, price: float) -> None:
        """Execute sell order"""
        if self.position == Positions.LONG:
            # Close long position
            self._close_position(price, 'SELL')
            
        elif self.position == Positions.FLAT and self.enable_short:
            # Open short position
            position_value = self._calculate_position_size(price)
            
            # Apply slippage and fees
            execution_price = price * (1 - self.slippage)
            fees = position_value * self.fee_rate
            
            if self.balance >= fees:
                self.position_size = position_value / execution_price
                self.entry_price = execution_price
                self.position = Positions.SHORT
                self.balance -= fees
                self.total_fees_paid += fees
                
                # Record trade
                self._record_trade('SELL', execution_price, self.position_size, fees)
    
    def _close_position(self, price: float, action: str) -> None:
        """Close current position"""
        if self.position == Positions.FLAT:
            return
        
        # Apply slippage
        if action == 'SELL':
            execution_price = price * (1 - self.slippage)
        else:
            execution_price = price * (1 + self.slippage)
        
        # Calculate PnL
        if self.position == Positions.LONG:
            pnl = (execution_price - self.entry_price) * self.position_size
        else:  # SHORT
            pnl = (self.entry_price - execution_price) * self.position_size
        
        # Calculate fees
        position_value = self.position_size * execution_price
        fees = position_value * self.fee_rate
        
        # Update balance
        self.balance += position_value - fees
        self.realized_pnl += pnl - fees
        self.total_fees_paid += fees
        
        # Record trade
        self._record_trade(action, execution_price, self.position_size, fees, pnl)
        
        # Reset position
        self.position = Positions.FLAT
        self.entry_price = 0
        self.position_size = 0
        self.unrealized_pnl = 0
    
    def _check_exit_conditions(self, current_price: float) -> None:
        """Check stop loss and take profit conditions"""
        if self.position == Positions.FLAT:
            return
        
        # Calculate current PnL percentage
        if self.position == Positions.LONG:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Check stop loss
        if self.stop_loss and pnl_pct <= -self.stop_loss:
            if self.position == Positions.LONG:
                self._close_position(current_price, 'SELL')
            else:
                self._close_position(current_price, 'BUY')
        
        # Check take profit
        elif self.take_profit and pnl_pct >= self.take_profit:
            if self.position == Positions.LONG:
                self._close_position(current_price, 'SELL')
            else:
                self._close_position(current_price, 'BUY')
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on strategy"""
        if self.position_sizing == 'fixed':
            # Use fixed percentage of balance
            return self.balance * 0.95  # Use 95% of balance
        
        elif self.position_sizing == 'dynamic':
            # Kelly Criterion or risk-based sizing
            # Calculate based on recent volatility
            recent_returns = pd.Series(self.prices[-20:, 3]).pct_change().dropna()
            volatility = recent_returns.std()
            
            if volatility > 0:
                # Adjust position size based on volatility
                position_size = (self.balance * self.risk_per_trade) / (volatility * price)
                return min(position_size * price, self.balance * 0.95)
            else:
                return self.balance * 0.95
        
        else:
            return self.balance * 0.95
    
    def _get_current_price(self) -> float:
        """Get current market price"""
        return self.prices[self.current_step, 3]  # Close price
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        current_price = self._get_current_price()
        
        if self.position == Positions.FLAT:
            return self.balance
        
        # Calculate position value
        position_value = self.position_size * current_price
        
        # Calculate unrealized PnL
        if self.position == Positions.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
        
        return self.balance + position_value
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """
        Calculate step reward
        
        Multiple reward formulations available:
        1. Simple returns
        2. Risk-adjusted returns (Sharpe)
        3. Profit factor
        4. Custom scoring
        """
        current_portfolio_value = self._get_portfolio_value()
        
        # Simple return-based reward
        returns = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Add portfolio value tracking
        self.portfolio_values.append(current_portfolio_value)
        
        # Calculate Sharpe ratio component if enough history
        if len(self.portfolio_values) > 20:
            recent_returns = pd.Series(self.portfolio_values[-20:]).pct_change().dropna()
            if len(recent_returns) > 0 and recent_returns.std() > 0:
                sharpe_component = recent_returns.mean() / recent_returns.std()
            else:
                sharpe_component = 0
        else:
            sharpe_component = 0
        
        # Penalize drawdowns
        drawdown_penalty = 0
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
        else:
            self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            drawdown_penalty = -self.current_drawdown * 0.5
            
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Combine reward components
        reward = returns + sharpe_component * 0.1 + drawdown_penalty
        
        # Scale reward
        return reward * self.reward_scaling
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state
        
        Includes:
        - Price history (OHLCV)
        - Technical indicators (if provided)
        - Account state
        - Position information
        """
        obs = []
        
        # Price history (normalized)
        if self.current_step >= self.window_size:
            price_history = self.prices[self.current_step - self.window_size:self.current_step]
            volume_history = self.volumes[self.current_step - self.window_size:self.current_step]
            
            # Normalize prices
            current_price = self._get_current_price()
            normalized_prices = price_history / current_price
            obs.extend(normalized_prices.flatten())
            
            # Normalize volumes
            mean_volume = np.mean(volume_history)
            if mean_volume > 0:
                normalized_volumes = volume_history / mean_volume
            else:
                normalized_volumes = volume_history
            obs.extend(normalized_volumes)
        
        # Technical indicators (if available)
        if self.features_df is not None and self.current_step < len(self.features_df):
            technical_features = self.features_df.iloc[self.current_step].values
            obs.extend(technical_features)
        
        # Account state (normalized)
        obs.extend([
            self.balance / self.initial_balance,
            self._get_portfolio_value() / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            self.total_fees_paid / self.initial_balance,
            len(self.trades) / 100,  # Normalized trade count
            self._get_win_rate(),
            self.current_drawdown,
            self.max_drawdown,
            self._get_sharpe_ratio()
        ])
        
        # Position information
        obs.extend([
            float(self.position),
            self.entry_price / current_price if self.entry_price > 0 else 0,
            self.position_size,
            self.unrealized_pnl / self.initial_balance if self.position != Positions.FLAT else 0,
            self._get_position_duration() / 100  # Normalized duration
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _check_done(self) -> None:
        """Check if episode should end"""
        # End if we've processed all data
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            self.truncated = False
            return
        
        # End if balance is too low (bankruptcy)
        if self.balance < self.initial_balance * 0.1:  # Lost 90% of capital
            self.done = True
            self.truncated = True
            return
        
        # End if drawdown is too high
        if self.max_drawdown > 0.5:  # 50% drawdown
            self.done = True
            self.truncated = True
            return
    
    def _update_metrics(self) -> None:
        """Update performance metrics"""
        # Update equity
        self.equity = self._get_portfolio_value()
        
        # Update peak and drawdown
        if self.equity > self.peak_portfolio_value:
            self.peak_portfolio_value = self.equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_portfolio_value - self.equity) / self.peak_portfolio_value
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def _record_trade(self, action: str, price: float, size: float, 
                     fees: float, pnl: Optional[float] = None) -> None:
        """Record trade information"""
        trade = {
            'step': self.current_step,
            'timestamp': self.df.index[self.current_step] if hasattr(self.df.index, '__iter__') else self.current_step,
            'action': action,
            'price': price,
            'size': size,
            'fees': fees,
            'pnl': pnl,
            'balance': self.balance,
            'equity': self._get_portfolio_value()
        }
        self.trades.append(trade)
    
    def _get_win_rate(self) -> float:
        """Calculate win rate from trades"""
        if not self.trades:
            return 0.5
        
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) and t['pnl'] > 0)
        total_trades = sum(1 for t in self.trades if t.get('pnl') is not None)
        
        if total_trades == 0:
            return 0.5
        
        return winning_trades / total_trades
    
    def _get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 20:
            return 0
        
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        return 0
    
    def _get_position_duration(self) -> int:
        """Get how long position has been held"""
        if self.position == Positions.FLAT:
            return 0
        
        # Find when position was opened
        for i in range(len(self.trades) - 1, -1, -1):
            trade = self.trades[i]
            if trade['action'] in ['BUY', 'SELL'] and trade.get('pnl') is None:
                return self.current_step - trade['step']
        
        return 0
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position.name,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_fees': self.total_fees_paid,
            'num_trades': len(self.trades),
            'win_rate': self._get_win_rate(),
            'sharpe_ratio': self._get_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'current_step': self.current_step,
            'portfolio_value': self._get_portfolio_value()
        }
    
    def get_trading_state(self) -> TradingState:
        """Get complete trading state"""
        return TradingState(
            step=self.current_step,
            position=int(self.position),
            entry_price=self.entry_price,
            balance=self.balance,
            equity=self.equity,
            portfolio_value=self._get_portfolio_value(),
            position_size=self.position_size,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            total_trades=len(self.trades),
            winning_trades=sum(1 for t in self.trades if t.get('pnl', 0) and t['pnl'] > 0),
            losing_trades=sum(1 for t in self.trades if t.get('pnl', 0) and t['pnl'] < 0),
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=self._get_sharpe_ratio()
        )
    
    def render(self, mode: str = 'human') -> None:
        """
        Render environment state
        
        Args:
            mode: Rendering mode ('human' for text output)
        """
        if mode == 'human':
            state = self.get_trading_state()
            print(f"\n{'='*50}")
            print(f"Step: {state.step}")
            print(f"Position: {Positions(state.position).name}")
            print(f"Balance: ${state.balance:.2f}")
            print(f"Equity: ${state.equity:.2f}")
            print(f"Portfolio Value: ${state.portfolio_value:.2f}")
            print(f"Unrealized PnL: ${state.unrealized_pnl:.2f}")
            print(f"Realized PnL: ${state.realized_pnl:.2f}")
            print(f"Total Trades: {state.total_trades}")
            print(f"Win Rate: {self._get_win_rate():.2%}")
            print(f"Sharpe Ratio: {state.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {state.max_drawdown:.2%}")
            print(f"{'='*50}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get complete performance summary"""
        if not self.trades:
            return {}
        
        trades_with_pnl = [t for t in self.trades if t.get('pnl') is not None]
        
        if not trades_with_pnl:
            return {
                'total_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        winning_trades = [t for t in trades_with_pnl if t['pnl'] > 0]
        losing_trades = [t for t in trades_with_pnl if t['pnl'] < 0]
        
        total_return = (self._get_portfolio_value() - self.initial_balance) / self.initial_balance
        
        return {
            'total_trades': len(trades_with_pnl),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_with_pnl) if trades_with_pnl else 0,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_fees': self.total_fees_paid,
            'sharpe_ratio': self._get_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'final_balance': self.balance,
            'final_equity': self.equity,
            'final_portfolio_value': self._get_portfolio_value(),
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'profit_factor': abs(sum(t['pnl'] for t in winning_trades) / 
                               sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        }


# Testing function
def test_environment():
    """Test the trading environment"""
    print("Testing Trading Environment...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(45000, 46000, len(dates)),
        'low': np.random.uniform(39000, 40000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Initialize environment
    env = TradingEnvironment(
        df=sample_data,
        initial_balance=10000,
        fee_rate=0.0026,
        window_size=20,
        enable_short=False,
        stop_loss=0.05,
        take_profit=0.10
    )
    
    # Run a simple episode
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for i in range(100):
        # Random action
        action = np.random.choice([0, 1, 2])
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Print info every 20 steps
        if i % 20 == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if done:
            break
    
    # Print final summary
    print("\n=== Episode Summary ===")
    summary = env.get_performance_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    return env


if __name__ == "__main__":
    # Test the environment
    env = test_environment()
    print("\nTrading Environment is ready!")