"""
Multi-Timeframe Trading Environment - FIXED VERSION
====================================================

CRITICAL FIXES:
1. Trade-completion based rewards (not step-by-step)
2. Minimum hold duration enforcement
3. Short trade penalties
4. Holding bonuses
5. Profit threshold requirements
6. Fee-aware reward scaling

This version encourages the agent to:
- Hold positions for 3-6+ hours
- Only exit on strong signals
- Avoid excessive trading
- Let winners run
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import IntEnum
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


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


@dataclass
class RewardConfig:
    """Configuration for the improved reward system"""
    # Minimum hold requirements
    min_hold_steps: int = 36          # 3 hours on 5m (36 × 5 = 180 min)
    target_hold_steps: int = 72       # 6 hours target
    max_hold_steps: int = 200         # Auto-close after this
    
    # Profit thresholds
    min_profit_for_reward: float = 0.015   # 1.5% minimum for full positive reward
    fee_rate: float = 0.001                 # 0.1% per trade
    slippage: float = 0.0005                # 0.05% slippage
    
    # Reward scaling
    hold_bonus_per_step: float = 0.002     # Small bonus each step held
    short_trade_penalty: float = 1.0       # Penalty for trades < min_hold
    early_exit_multiplier: float = 0.3     # Reduce reward by 70% for early exits
    
    # Step rewards (during hold)
    step_reward_scale: float = 0.05        # 5% of normal (drastically reduced)
    flat_penalty: float = 0.0005           # Small penalty for sitting flat


class MultiTimeframeEnvironment:
    """
    FIXED Multi-timeframe trading environment
    
    Key Changes:
    1. Rewards primarily on trade COMPLETION, not every step
    2. Minimum hold duration enforced through penalties
    3. Holding bonuses that grow over time
    4. Short trade penalties even if profitable
    5. Fee-awareness built into reward calculation
    """
    
    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        features_dfs: Dict[str, pd.DataFrame],
        execution_timeframe: str = '5m',
        initial_balance: float = 10000,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        stop_loss: Optional[float] = 0.04,      # Wider: 4% (was 3%)
        take_profit: Optional[float] = 0.08,    # Wider: 8% (was 5-6%)
        window_size: int = 50,
        asset: str = 'BTC_USDT',
        selected_features: Optional[List[str]] = None,
        max_position_hold_steps: int = 200,
        enable_short: bool = False,
        reward_config: Optional[RewardConfig] = None
    ):
        """Initialize with FIXED reward system"""
        
        self.dataframes = dataframes
        self.features_dfs = features_dfs
        self.execution_timeframe = execution_timeframe
        self.timeframes = sorted(dataframes.keys())
        self.asset = asset
        self.enable_short = enable_short
        
        # Validate inputs
        self._validate_inputs()
        
        # Synchronize timeframes
        self.synchronized_data = self._synchronize_timeframes()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.window_size = window_size
        self.selected_features = selected_features
        self.max_position_hold_steps = max_position_hold_steps
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Reward configuration
        # ═══════════════════════════════════════════════════════════════
        self.reward_config = reward_config or RewardConfig(
            fee_rate=fee_rate,
            slippage=slippage
        )
        
        # Track trade state for reward calculation
        self._trade_just_closed = False
        self._last_trade_pnl = None
        self._last_hold_duration = 0
        
        # Filter features
        if selected_features:
            for tf in self.timeframes:
                available = [f for f in selected_features if f in self.features_dfs[tf].columns]
                if available:
                    self.features_dfs[tf] = self.features_dfs[tf][available]
        
        # Asset encoding
        self.asset_encoding = self._encode_asset(asset)
        
        # State tracking
        self.current_step = 0
        self.position = Positions.FLAT
        self.entry_price = 0
        self.position_size = 0
        self.position_opened_step = None
        
        # Account tracking
        self.balance = initial_balance
        self.equity = initial_balance
        self.portfolio_values = [initial_balance]
        
        # Performance tracking
        self.trades = []
        self.total_fees_paid = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        # Risk metrics
        self.peak_portfolio_value = initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0
        
        # Episode flags
        self.done = False
        self.truncated = False
        
        # Spaces
        self.action_space_n = 3
        self.observation_space_shape = self._get_observation_shape()
        
        # Pre-compute observations
        self.precompute_observations = True
        self.precomputed_obs = None
        if self.precompute_observations:
            self._precompute_all_observations()
        
        # ═══════════════════════════════════════════════════════════════
        # NEW: Reward statistics tracking
        # ═══════════════════════════════════════════════════════════════
        self.reward_stats = {
            'total_step_rewards': 0,
            'total_trade_rewards': 0,
            'total_hold_bonuses': 0,
            'short_trades': 0,
            'long_trades': 0,
            'trades_penalized': 0
        }
        
        logger.info(f" FIXED Multi-Timeframe Environment initialized")
        logger.info(f"  Asset: {self.asset}")
        logger.info(f"  Min hold: {self.reward_config.min_hold_steps} steps ({self.reward_config.min_hold_steps * 5} min)")
        logger.info(f"  Target hold: {self.reward_config.target_hold_steps} steps")
        logger.info(f"  Stop loss: {self.stop_loss*100:.1f}%")
        logger.info(f"  Take profit: {self.take_profit*100:.1f}%")

    def _validate_inputs(self):
        """Validate inputs"""
        if not self.dataframes:
            raise ValueError("Must provide at least one timeframe")
        
        if self.execution_timeframe not in self.dataframes:
            raise ValueError(f"Execution timeframe {self.execution_timeframe} not in dataframes")
        
        for tf, df in self.dataframes.items():
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(f"Timeframe {tf} missing columns: {missing}")
        
        for tf in self.dataframes.keys():
            if tf not in self.features_dfs:
                raise ValueError(f"Missing features for timeframe {tf}")

    def _synchronize_timeframes(self) -> pd.DataFrame:
        """Synchronize all timeframes"""
        start_time = time.time()
        
        exec_df = self.dataframes[self.execution_timeframe].copy()
        sync_data = pd.DataFrame(index=exec_df.index)
        
        for tf in self.timeframes:
            tf_df = self.dataframes[tf].copy()
            tf_df['_temp_index'] = tf_df.index
            
            merged = pd.merge_asof(
                pd.DataFrame(index=exec_df.index),
                tf_df[['_temp_index']],
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            sync_data[f'{tf}_index'] = merged['_temp_index'].values
        
        sync_data = sync_data.dropna()
        
        elapsed = time.time() - start_time
        logger.info(f"   Synchronized {len(sync_data)} candles in {elapsed:.2f}s")
        
        return sync_data

    def _get_observation_shape(self) -> Tuple[int]:
        """Calculate observation space dimensions"""
        total_features = sum(len(self.features_dfs[tf].columns) for tf in self.timeframes)
        
        position_dims = 5
        account_dims = 5
        asset_dims = 5
        encoding_dims = len(self.timeframes) * 6
        
        total = total_features + position_dims + account_dims + asset_dims + encoding_dims
        return (total,)

    def _precompute_all_observations(self):
        """Pre-compute observations for speed"""
        logger.info("   Pre-computing observations...")
        start_time = time.time()
        
        n_steps = len(self.synchronized_data)
        obs_shape = self.observation_space_shape[0]
        
        self.precomputed_obs = np.zeros((n_steps, obs_shape), dtype=np.float32)
        
        # Save state
        original_step = self.current_step
        original_position = self.position
        original_entry_price = self.entry_price
        original_position_size = self.position_size
        original_position_opened_step = self.position_opened_step
        original_balance = self.balance
        original_portfolio_values = self.portfolio_values.copy()
        original_trades = self.trades.copy()
        original_realized_pnl = self.realized_pnl
        original_unrealized_pnl = self.unrealized_pnl
        original_total_fees = self.total_fees_paid
        original_peak = self.peak_portfolio_value
        original_max_dd = self.max_drawdown
        original_current_dd = self.current_drawdown
        
        for i in range(self.window_size + 1, n_steps):
            self.current_step = i
            obs = self._calculate_observation()
            self.precomputed_obs[i] = obs
        
        # Restore state
        self.current_step = original_step
        self.position = original_position
        self.entry_price = original_entry_price
        self.position_size = original_position_size
        self.position_opened_step = original_position_opened_step
        self.balance = original_balance
        self.portfolio_values = original_portfolio_values
        self.trades = original_trades
        self.realized_pnl = original_realized_pnl
        self.unrealized_pnl = original_unrealized_pnl
        self.total_fees_paid = original_total_fees
        self.peak_portfolio_value = original_peak
        self.max_drawdown = original_max_dd
        self.current_drawdown = original_current_dd
        
        elapsed = time.time() - start_time
        logger.info(f"   Pre-computed {n_steps - self.window_size:,} observations in {elapsed:.2f}s")

    def _calculate_observation(self) -> np.ndarray:
        """Calculate observation from ALL timeframes (NO LOOK-AHEAD)"""
        obs_parts = []
        
        # Use PREVIOUS bar's data (t-1) to prevent look-ahead
        if self.current_step > 0:
            sync_row = self.synchronized_data.iloc[self.current_step - 1]
        else:
            sync_row = self.synchronized_data.iloc[self.current_step]
        
        # Features from each timeframe
        for tf in self.timeframes:
            tf_index = sync_row[f'{tf}_index']
            
            if tf_index in self.features_dfs[tf].index:
                features = self.features_dfs[tf].loc[tf_index].values
            else:
                features = np.zeros(len(self.features_dfs[tf].columns))
            
            obs_parts.append(features)
        
        # Position information
        position_info = self._get_position_info()
        obs_parts.append(position_info)
        
        # Account information
        account_info = self._get_account_info()
        obs_parts.append(account_info)
        
        # Asset encoding
        obs_parts.append(self.asset_encoding)
        
        # Timeframe encodings
        for tf in self.timeframes:
            encoding = self._encode_timeframe(tf)
            obs_parts.append(encoding)
        
        observation = np.concatenate(obs_parts).astype(np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -10, 10)
        
        return observation

    def _get_position_info(self) -> np.ndarray:
        """Get position information (5 dims)"""
        exec_index = self.synchronized_data.index[self.current_step]
        current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
        
        return np.array([
            float(self.position),
            self.entry_price / current_price if self.entry_price > 0 else 0,
            self.position_size / self.initial_balance if self.position_size > 0 else 0,
            self.unrealized_pnl / self.initial_balance if self.position != Positions.FLAT else 0,
            self._get_position_duration() / 100
        ], dtype=np.float32)

    def _get_account_info(self) -> np.ndarray:
        """Get account information (5 dims)"""
        portfolio_value = self._get_portfolio_value()
        
        return np.array([
            self.balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            self.total_fees_paid / self.initial_balance
        ], dtype=np.float32)

    def _get_position_duration(self) -> int:
        """Get how long position has been held"""
        if self.position == Positions.FLAT:
            return 0
        if self.position_opened_step is None:
            return 0
        return self.current_step - self.position_opened_step

    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        if self.precompute_observations and self.precomputed_obs is not None:
            obs = self.precomputed_obs[self.current_step].copy()
        else:
            obs = self._calculate_observation()
        return obs

    def _encode_asset(self, asset: str) -> np.ndarray:
        """Encode asset as one-hot vector"""
        asset_map = {
            'ETH_USDT': 0, 'ETH/USDT': 0,
            'SOL_USDT': 1, 'SOL/USDT': 1,
            'DOT_USDT': 2, 'DOT/USDT': 2,
            'AVAX_USDT': 3, 'AVAX/USDT': 3,
            'ADA_USDT': 4, 'ADA/USDT': 4
        }
        
        encoding = np.zeros(5, dtype=np.float32)
        if asset in asset_map:
            encoding[asset_map[asset]] = 1.0
        return encoding

    def _encode_timeframe(self, timeframe: str) -> np.ndarray:
        """Encode timeframe as one-hot vector"""
        timeframe_map = {'1m': 0, '5m': 1, '15m': 2, '30m': 3, '1h': 4, '4h': 5}
        
        encoding = np.zeros(6, dtype=np.float32)
        if timeframe in timeframe_map:
            encoding[timeframe_map[timeframe]] = 1.0
        return encoding

    def reset(self, seed: Optional[int] = None, random_start: bool = True, max_steps: int = 900) -> np.ndarray:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)
        
        # Random starting position
        if random_start and len(self.synchronized_data) > max_steps + self.window_size + 100:
            min_start = self.window_size
            max_start = len(self.synchronized_data) - max_steps - 50
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = self.window_size
        
        # Reset state
        self.position = Positions.FLAT
        self.entry_price = 0
        self.position_size = 0
        self.position_opened_step = None
        
        # Reset account
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        
        # Reset tracking
        self.trades = []
        self.total_fees_paid = 0
        self.realized_pnl = 0
        self.unrealized_pnl = 0
        
        # Reset risk metrics
        self.peak_portfolio_value = self.initial_balance
        self.max_drawdown = 0
        self.current_drawdown = 0
        
        # Reset flags
        self.done = False
        self.truncated = False
        
        # Reset trade state tracking
        self._trade_just_closed = False
        self._last_trade_pnl = None
        self._last_hold_duration = 0
        
        # Reset reward stats
        self.reward_stats = {
            'total_step_rewards': 0,
            'total_trade_rewards': 0,
            'total_hold_bonuses': 0,
            'short_trades': 0,
            'long_trades': 0,
            'trades_penalized': 0
        }
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take action and return next state with FIXED reward"""
        exec_index = self.synchronized_data.index[self.current_step]
        current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
        
        prev_portfolio_value = self._get_portfolio_value()
        
        # Check position timeout
        self._check_position_timeout()
        
        # Execute action
        self._execute_action(action)
        
        # Validate
        if not self._validate_balance():
            observation = self._get_observation()
            reward = -10.0  # Big penalty for breaking
            return observation, reward, True, True, self._get_info()
        
        # Check stop loss / take profit
        if self.position != Positions.FLAT:
            self._check_exit_conditions(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Update metrics
        self._update_metrics()
        
        # ═══════════════════════════════════════════════════════════════
        # FIXED: Calculate reward with new system
        # ═══════════════════════════════════════════════════════════════
        reward = self._calculate_reward_fixed(prev_portfolio_value)
        
        # Check if done
        self._check_done()
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Info dict
        info = self._get_info()
        
        return next_obs, reward, self.done, self.truncated, info

    def _execute_action(self, action: int):
        """Execute trading action"""
        current_price = self._get_current_price()
        
        if action == Actions.BUY:
            self._execute_buy(current_price)
        elif action == Actions.SELL:
            self._execute_sell(current_price)
        
        # Validate position size
        if not self._validate_position_size():
            logger.warning("Position validation failed")
            self.done = True
            self.truncated = True
            return
        
        # Check exit conditions
        if self.position != Positions.FLAT:
            self._check_exit_conditions(current_price)

    def _execute_buy(self, price: float) -> None:
        """Execute BUY action"""
        # Close short if open
        if self.position == Positions.SHORT:
            execution_price = price * (1 + self.slippage)
            gross_position_value = self.position_size * execution_price
            fees = gross_position_value * self.fee_rate
            net_proceeds = gross_position_value - fees
            
            gross_pnl = (self.entry_price - execution_price) * self.position_size
            net_pnl = gross_pnl - fees
            
            # Track trade completion
            self._trade_just_closed = True
            self._last_trade_pnl = net_pnl
            self._last_hold_duration = self._get_position_duration()
            
            self.balance += net_proceeds
            self.realized_pnl += net_pnl
            self.total_fees_paid += fees
            
            self._record_trade('BUY_COVER', price, self.position_size, fees, net_pnl)
            
            self.position = Positions.FLAT
            self.position_size = 0
            self.entry_price = 0
            self.position_opened_step = None
            return
        
        # Check if already long
        if self.position == Positions.LONG:
            return
        
        # Open long
        available_capital = self.balance * 0.95
        if available_capital < 10:
            return
        
        execution_price = price * (1 + self.slippage)
        gross_position_size = available_capital / execution_price
        fee_in_dollars = available_capital * self.fee_rate
        coins_lost_to_fees = fee_in_dollars / execution_price
        net_position_size = gross_position_size - coins_lost_to_fees
        
        # Validate
        position_value = net_position_size * execution_price
        if position_value > self.initial_balance * 1.5:
            return
        
        self.balance -= available_capital
        
        if self.balance < 0:
            self.balance += available_capital
            self.done = True
            self.truncated = True
            return
        
        self.position_size = net_position_size
        self.entry_price = execution_price
        self.position = Positions.LONG
        self.position_opened_step = self.current_step
        self.total_fees_paid += fee_in_dollars
        
        self._record_trade('BUY', execution_price, self.position_size, fee_in_dollars, None)

    def _execute_sell(self, price: float) -> None:
        """Execute SELL action"""
        # Close long if open
        if self.position == Positions.LONG:
            execution_price = price * (1 - self.slippage)
            proceeds = self.position_size * execution_price
            fee = proceeds * self.fee_rate
            net_proceeds = proceeds - fee
            
            pnl = (execution_price - self.entry_price) * self.position_size - fee
            
            # Track trade completion
            self._trade_just_closed = True
            self._last_trade_pnl = pnl
            self._last_hold_duration = self._get_position_duration()
            
            self.balance += net_proceeds
            self.realized_pnl += pnl
            self.total_fees_paid += fee
            
            self._record_trade('SELL', execution_price, self.position_size, fee, pnl)
            
            self.position = Positions.FLAT
            self.position_size = 0
            self.entry_price = 0
            self.unrealized_pnl = 0
            self.position_opened_step = None
            return
        
        # Can't sell when flat unless shorting enabled
        if self.position == Positions.FLAT and not self.enable_short:
            return
        
        if not self.enable_short:
            return
        
        # Open short
        available_capital = self.balance * 0.95
        if available_capital < 10:
            return
        
        execution_price = price * (1 - self.slippage)
        gross_position_size = available_capital / execution_price
        fee_in_dollars = available_capital * self.fee_rate
        coins_lost_to_fees = fee_in_dollars / execution_price
        net_position_size = gross_position_size - coins_lost_to_fees
        
        position_value = net_position_size * execution_price
        if position_value > self.initial_balance * 1.5:
            return
        
        self.balance -= available_capital
        
        if self.balance < 0:
            self.balance += available_capital
            self.done = True
            self.truncated = True
            return
        
        self.position_size = net_position_size
        self.entry_price = execution_price
        self.position = Positions.SHORT
        self.position_opened_step = self.current_step
        self.total_fees_paid += fee_in_dollars
        
        self._record_trade('SELL_SHORT', execution_price, self.position_size, fee_in_dollars, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # FIXED REWARD FUNCTION - THE KEY CHANGE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_reward_fixed(self, prev_portfolio_value: float) -> float:
        """
        FIXED reward calculation that encourages longer holds
        
        Key principles:
        1. Meaningful rewards only on trade COMPLETION
        2. Penalize short trades even if profitable
        3. Small holding bonus each step
        4. Minimum profit threshold for full reward
        5. Drastically reduced step-by-step rewards
        """
        current_portfolio_value = self._get_portfolio_value()
        portfolio_change = current_portfolio_value - prev_portfolio_value
        
        self.portfolio_values.append(current_portfolio_value)
        
        config = self.reward_config
        hold_duration = self._get_position_duration()
        
        # ═══════════════════════════════════════════════════════════════
        # CASE 1: Trade just closed - THIS IS WHERE REAL REWARDS HAPPEN
        # ═══════════════════════════════════════════════════════════════
        if self._trade_just_closed:
            pnl = self._last_trade_pnl or 0
            actual_duration = self._last_hold_duration
            pnl_pct = pnl / self.initial_balance
            
            # Reset flags
            self._trade_just_closed = False
            self._last_trade_pnl = None
            self._last_hold_duration = 0
            
            # Base reward: scaled P&L percentage
            base_reward = pnl_pct * 100  # Convert to percentage points
            
            # Check if trade was too short
            if actual_duration < config.min_hold_steps:
                # ═══════════════════════════════════════════════════════
                # SHORT TRADE - PENALIZE REGARDLESS OF PROFIT
                # ═══════════════════════════════════════════════════════
                self.reward_stats['short_trades'] += 1
                self.reward_stats['trades_penalized'] += 1
                
                duration_ratio = actual_duration / config.min_hold_steps
                
                if pnl_pct > 0:
                    # Profitable but too short: heavily reduced reward
                    reward = base_reward * config.early_exit_multiplier * duration_ratio
                    reward -= config.short_trade_penalty
                    
                    # Additional fee awareness: if profit < fees, extra penalty
                    total_fee_impact = config.fee_rate * 2 + config.slippage * 2
                    if pnl_pct < total_fee_impact:
                        reward -= 0.5  # Extra penalty for fee-losing trades
                else:
                    # Loss AND short: extra penalty
                    reward = base_reward * 1.5  # 50% extra loss penalty
                    reward -= config.short_trade_penalty
            else:
                # ═══════════════════════════════════════════════════════
                # GOOD DURATION - Held long enough
                # ═══════════════════════════════════════════════════════
                self.reward_stats['long_trades'] += 1
                
                if pnl_pct >= config.min_profit_for_reward:
                    # Profitable AND held well: full reward + bonus!
                    reward = base_reward
                    
                    # Bonus for holding beyond minimum
                    extra_hold = actual_duration - config.min_hold_steps
                    hold_bonus = min(extra_hold * 0.02, 2.0)  # Cap bonus at 2.0
                    reward += hold_bonus
                    
                    self.reward_stats['total_hold_bonuses'] += hold_bonus
                    
                elif pnl_pct > 0:
                    # Small profit: partial reward
                    reward = base_reward * 0.5
                else:
                    # Loss but held properly: standard penalty (no extra)
                    reward = base_reward
            
            self.reward_stats['total_trade_rewards'] += reward
            return reward
        
        # ═══════════════════════════════════════════════════════════════
        # CASE 2: Currently holding a position
        # ═══════════════════════════════════════════════════════════════
        if self.position != Positions.FLAT:
            # Minimal step reward (5% of normal)
            step_reward = (portfolio_change / self.initial_balance) * 100 * config.step_reward_scale
            
            # Add holding bonus (encourages patience)
            step_reward += config.hold_bonus_per_step
            
            # Extra bonus after minimum hold period
            if hold_duration > config.min_hold_steps:
                step_reward += config.hold_bonus_per_step * 0.5
            
            # Larger bonus approaching target
            if hold_duration > config.target_hold_steps * 0.8:
                step_reward += config.hold_bonus_per_step * 0.3
            
            self.reward_stats['total_step_rewards'] += step_reward
            return step_reward
        
        # ═══════════════════════════════════════════════════════════════
        # CASE 3: Flat (not in position)
        # ═══════════════════════════════════════════════════════════════
        else:
            # Small penalty for sitting flat (encourages being in market)
            # But not too much - we don't want forced bad trades
            return -config.flat_penalty

    def _check_exit_conditions(self, current_price: float) -> None:
        """Check stop loss and take profit"""
        if self.position == Positions.FLAT:
            return
        
        if self.position == Positions.LONG:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Stop loss
        if self.stop_loss and pnl_pct <= -self.stop_loss:
            if self.position == Positions.LONG:
                self._execute_sell(current_price)
            else:
                self._execute_buy(current_price)
        
        # Take profit
        elif self.take_profit and pnl_pct >= self.take_profit:
            if self.position == Positions.LONG:
                self._execute_sell(current_price)
            else:
                self._execute_buy(current_price)

    def _check_position_timeout(self) -> None:
        """Auto-close positions held too long"""
        if self.position == Positions.FLAT:
            return
        
        if self.position_opened_step is None:
            return
        
        hold_duration = self.current_step - self.position_opened_step
        
        if hold_duration >= self.max_position_hold_steps:
            exec_index = self.synchronized_data.index[self.current_step]
            current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
            
            if self.position == Positions.LONG:
                self._execute_sell(current_price)
            else:
                self._execute_buy(current_price)

    def _validate_position_size(self) -> bool:
        """Validate position size"""
        if self.position == Positions.FLAT:
            return True
        
        position_value_at_entry = self.position_size * self.entry_price
        max_reasonable = self.peak_portfolio_value * 1.5
        
        if position_value_at_entry > max_reasonable:
            exec_index = self.synchronized_data.index[self.current_step]
            current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
            
            if self.position == Positions.LONG:
                self._execute_sell(current_price)
            else:
                self._execute_buy(current_price)
            
            self.done = True
            self.truncated = True
            return False
        
        return True

    def _validate_balance(self) -> bool:
        """Validate balance"""
        if self.balance < 0:
            self.done = True
            self.truncated = True
            return False
        
        if self.balance > self.initial_balance * 1000:
            self.done = True
            self.truncated = True
            return False
        
        return True

    def _get_current_price(self) -> float:
        """Get current price"""
        exec_index = self.synchronized_data.index[self.current_step]
        return self.dataframes[self.execution_timeframe].loc[exec_index, 'close']

    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        if self.position == Positions.FLAT:
            return self.balance
        
        exec_index = self.synchronized_data.index[self.current_step]
        current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
        
        position_value = self.position_size * current_price
        
        if self.position == Positions.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.position_size
        
        return self.balance + position_value

    def _update_metrics(self) -> None:
        """Update performance metrics"""
        self.equity = self._get_portfolio_value()
        
        if self.equity > self.peak_portfolio_value:
            self.peak_portfolio_value = self.equity
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_portfolio_value - self.equity) / self.peak_portfolio_value
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown

    def _check_done(self) -> None:
        """Check if episode is done"""
        if self.current_step >= len(self.synchronized_data) - 1:
            if self.position != Positions.FLAT:
                exec_index = self.synchronized_data.index[self.current_step]
                current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
                
                if self.position == Positions.LONG:
                    self._execute_sell(current_price)
                else:
                    self._execute_buy(current_price)
            
            self.done = True
            self.truncated = True
            return
        
        # Bankruptcy
        portfolio_value = self._get_portfolio_value()
        if portfolio_value < self.initial_balance * 0.1:
            if self.position != Positions.FLAT:
                exec_index = self.synchronized_data.index[self.current_step]
                current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
                if self.position == Positions.LONG:
                    self._execute_sell(current_price)
                else:
                    self._execute_buy(current_price)
            
            self.done = True
            self.truncated = False
            return
        
        # Extreme drawdown
        if self.max_drawdown > 0.5:
            if self.position != Positions.FLAT:
                exec_index = self.synchronized_data.index[self.current_step]
                current_price = self.dataframes[self.execution_timeframe].loc[exec_index, 'close']
                if self.position == Positions.LONG:
                    self._execute_sell(current_price)
                else:
                    self._execute_buy(current_price)
            
            self.done = True
            self.truncated = False

    def _record_trade(self, action: str, price: float, size: float,
                     fees: float, pnl: Optional[float] = None) -> None:
        """Record trade"""
        exec_index = self.synchronized_data.index[self.current_step]
        
        duration = None
        if pnl is not None and self.position_opened_step is not None:
            calculated_duration = self.current_step - self.position_opened_step
            if 0 <= calculated_duration < 10000:
                duration = calculated_duration
        
        trade = {
            'step': self.current_step,
            'timestamp': exec_index,
            'action': action,
            'price': price,
            'size': size,
            'fees': fees,
            'pnl': pnl,
            'duration': duration,
            'balance': self.balance,
            'equity': self._get_portfolio_value()
        }
        self.trades.append(trade)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position.name,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_fees': self.total_fees_paid,
            'num_trades': len([t for t in self.trades if t.get('pnl') is not None]),
            'win_rate': self._get_win_rate(),
            'sharpe_ratio': self._get_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'current_step': self.current_step,
            'portfolio_value': self._get_portfolio_value(),
            # NEW: Reward stats
            'reward_stats': self.reward_stats.copy()
        }

    def _get_win_rate(self) -> float:
        """Calculate win rate"""
        trades_with_pnl = [t for t in self.trades if t.get('pnl') is not None]
        if not trades_with_pnl:
            return 0.5
        
        winning = sum(1 for t in trades_with_pnl if t['pnl'] > 0)
        return winning / len(trades_with_pnl)

    def _get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 20:
            return 0
        
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            return (returns.mean() / returns.std()) * np.sqrt(252)
        return 0

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
            winning_trades=sum(1 for t in self.trades if t.get('pnl') is not None and t['pnl'] > 0),
            losing_trades=sum(1 for t in self.trades if t.get('pnl') is not None and t['pnl'] < 0),
            current_drawdown=self.current_drawdown,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=self._get_sharpe_ratio()
        )

    def render(self, mode: str = 'human') -> None:
        """Render environment state"""
        if mode == 'human':
            state = self.get_trading_state()
            print(f"\n{'='*50}")
            print(f"Step: {state.step}")
            print(f"Position: {Positions(state.position).name}")
            print(f"Balance: ${state.balance:.2f}")
            print(f"Portfolio Value: ${state.portfolio_value:.2f}")
            print(f"Win Rate: {self._get_win_rate():.2%}")
            print(f"Reward Stats: {self.reward_stats}")
            print(f"{'='*50}")

    def get_performance_summary(self) -> dict:
        """Get complete performance summary"""
        if not self.trades:
            return {'total_trades': 0}
        
        trades_with_pnl = [t for t in self.trades if t.get('pnl') is not None]
        
        if not trades_with_pnl:
            return {'total_trades': 0}
        
        winning_trades = [t for t in trades_with_pnl if t['pnl'] > 0]
        losing_trades = [t for t in trades_with_pnl if t['pnl'] < 0]
        
        # Calculate duration stats
        durations = [t['duration'] for t in trades_with_pnl if t.get('duration') is not None]
        avg_duration = np.mean(durations) if durations else 0
        
        total_return = (self._get_portfolio_value() - self.initial_balance) / self.initial_balance
        
        return {
            'total_trades': len(trades_with_pnl),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_with_pnl) if trades_with_pnl else 0,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'realized_pnl': self.realized_pnl,
            'total_fees': self.total_fees_paid,
            'sharpe_ratio': self._get_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'final_portfolio_value': self._get_portfolio_value(),
            'avg_duration_steps': avg_duration,
            'avg_duration_minutes': avg_duration * 5,  # 5m timeframe
            # Reward stats
            'short_trades': self.reward_stats['short_trades'],
            'long_trades': self.reward_stats['long_trades'],
            'trades_penalized': self.reward_stats['trades_penalized']
        }

    def get_feature_names(self) -> List[str]:
        """Get feature names for explainability"""
        feature_names = []
        
        for tf in self.timeframes:
            if self.selected_features:
                for feat in self.selected_features[:len(self.features_dfs[tf].columns)]:
                    feature_names.append(f"{tf}_{feat}")
            else:
                for feat in self.features_dfs[tf].columns:
                    feature_names.append(f"{tf}_{feat}")
        
        feature_names.extend([
            'position_type', 'entry_price_normalized', 'position_size_normalized',
            'unrealized_pnl_normalized', 'position_duration_normalized'
        ])
        
        feature_names.extend([
            'balance_normalized', 'portfolio_value_normalized', 'realized_pnl_normalized',
            'unrealized_pnl_normalized', 'total_fees_normalized'
        ])
        
        feature_names.extend(['asset_eth', 'asset_sol', 'asset_dot', 'asset_avax', 'asset_ada'])
        
        for tf in self.timeframes:
            feature_names.extend([
                f'{tf}_encoding_1m', f'{tf}_encoding_5m', f'{tf}_encoding_15m',
                f'{tf}_encoding_30m', f'{tf}_encoding_1h', f'{tf}_encoding_4h'
            ])
        
        return feature_names


if __name__ == "__main__":
    print(" FIXED Multi-Timeframe Environment")
    print("  - Trade-completion based rewards")
    print("  - Minimum hold duration: 36 steps (3 hours)")
    print("  - Short trade penalties")
    print("  - Holding bonuses")