"""
Trade-Based Multi-Timeframe Environment
========================================

FUNDAMENTAL CHANGE: Each episode = 1 complete trade

This environment keeps all the multi-timeframe features from the original
but completely changes how learning works:

OLD (Time-Based):
- Episode = 500-900 steps
- Reward = Portfolio change each step
- Agent learns: "Predict next candle direction"
- Problem: Encourages frequent trading

NEW (Trade-Based):
- Episode = 1 complete trade (entry → exit)
- Reward = Trade P&L only (at episode end)
- Agent learns: "Find good trade setups"
- Benefit: Quality over quantity

PHASE STRUCTURE:
- PHASE 1 (SEARCHING): Agent is FLAT, looking for entry
  - Actions: 0=WAIT, 1=ENTER
  - Reward: 0 (no reward while searching)
  - Ends when: Agent enters OR timeout

- PHASE 2 (IN_TRADE): Agent is LONG, looking for exit
  - Actions: 0=HOLD, 1=EXIT
  - Reward: 0 (no reward while holding)
  - Ends when: Agent exits OR stop-loss/take-profit OR timeout

- PHASE 3 (COMPLETE): Episode ends
  - Final reward = Trade P&L (including fees/slippage)
  - Reset to new random starting point

Key Features Preserved:
✓ Multi-timeframe observations (5m, 15m, 1h)
✓ Pre-computed observations for speed
✓ Slippage and fee modeling
✓ Stop-loss and take-profit
✓ Asset encoding
✓ All safety validations
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


class Phase(IntEnum):
    """Episode phases"""
    SEARCHING = 0  # Looking for entry (FLAT)
    IN_TRADE = 1   # In position, looking for exit


class Actions(IntEnum):
    """
    Actions change meaning based on phase!
    
    SEARCHING phase: 0=WAIT, 1=ENTER
    IN_TRADE phase:  0=HOLD, 1=EXIT
    """
    WAIT_OR_HOLD = 0
    ENTER_OR_EXIT = 1


class Positions(IntEnum):
    """Position states"""
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass
class TradeConfig:
    """Configuration for trade-based episodes"""
    # Timeouts
    max_wait_steps: int = 200       # Max steps to find entry (200 × 5m = ~17 hours)
    max_hold_steps: int = 300       # Max steps to hold trade (300 × 5m = 25 hours)
    
    # Penalties
    no_trade_penalty: float = -2.0   # Penalty for timeout without trading
    timeout_exit_penalty: float = -1.0  # Extra penalty for forced exit
    
    # Bonuses
    patience_bonus: float = 0.5      # Bonus for waiting for good setup
    hold_duration_bonus: float = 0.01  # Per-step bonus for holding (encourages patience)
    
    # Minimum requirements
    min_hold_for_bonus: int = 36     # 36 × 5m = 3 hours minimum for hold bonus
    
    # Trading costs (for reward awareness)
    fee_rate: float = 0.001          # 0.1% per trade
    slippage: float = 0.0005         # 0.05% slippage
    
    # Reward scaling
    reward_scale: float = 100.0      # Scale P&L percentage to reward


@dataclass 
class TradeResult:
    """Container for trade results"""
    entry_price: float
    exit_price: float
    entry_step: int
    exit_step: int
    position_size: float
    gross_pnl: float
    fees_paid: float
    net_pnl: float
    pnl_pct: float
    hold_duration: int
    exit_reason: str  # 'agent', 'stop_loss', 'take_profit', 'timeout'


class TradeBasedMultiTimeframeEnv:
    """
    Trade-Based Multi-Timeframe Trading Environment
    
    Each episode = 1 complete trade
    Agent learns trade QUALITY, not step-by-step prediction
    
    Preserves all multi-timeframe features from original environment.
    """
    
    def __init__(
        self,
        dataframes: Dict[str, pd.DataFrame],
        features_dfs: Dict[str, pd.DataFrame],
        execution_timeframe: str = '5m',
        initial_balance: float = 10000,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        stop_loss: Optional[float] = 0.03,
        take_profit: Optional[float] = 0.06,
        window_size: int = 50,
        asset: str = 'BTC_USDT',
        selected_features: Optional[List[str]] = None,
        enable_short: bool = False,
        trade_config: Optional[TradeConfig] = None
    ):
        """
        Initialize Trade-Based Multi-Timeframe Environment
        
        Args:
            dataframes: Dict mapping timeframe to OHLCV dataframe
            features_dfs: Dict mapping timeframe to features dataframe
            execution_timeframe: Which timeframe to use for execution
            initial_balance: Starting capital
            fee_rate: Trading fee per trade
            slippage: Slippage per trade
            stop_loss: Stop loss as fraction (e.g., 0.03 = 3%)
            take_profit: Take profit as fraction
            window_size: Lookback window
            asset: Trading asset name
            selected_features: List of feature names to use
            enable_short: Allow short positions (future feature)
            trade_config: Configuration for trade-based episodes
        """
        # Store parameters
        self.dataframes = dataframes
        self.features_dfs = features_dfs
        self.execution_timeframe = execution_timeframe
        self.timeframes = sorted(dataframes.keys())
        self.asset = asset
        self.enable_short = enable_short
        
        # Validate inputs
        self._validate_inputs()
        
        # Synchronize timeframes (same as original)
        self.synchronized_data = self._synchronize_timeframes()
        
        # Environment parameters
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.window_size = window_size
        self.selected_features = selected_features
        
        # Trade-based configuration
        self.trade_config = trade_config or TradeConfig(
            fee_rate=fee_rate,
            slippage=slippage
        )
        
        # Filter features if specified
        if selected_features:
            for tf in self.timeframes:
                available = [f for f in selected_features if f in self.features_dfs[tf].columns]
                if available:
                    self.features_dfs[tf] = self.features_dfs[tf][available]
        
        # Asset encoding
        self.asset_encoding = self._encode_asset(asset)
        
        # ═══════════════════════════════════════════════════════════════
        # ACTION SPACE: Only 2 actions (meaning depends on phase)
        # ═══════════════════════════════════════════════════════════════
        self.action_space_n = 2  # 0=WAIT/HOLD, 1=ENTER/EXIT
        
        # ═══════════════════════════════════════════════════════════════
        # OBSERVATION SPACE: Same as original + phase info
        # ═══════════════════════════════════════════════════════════════
        self.observation_space_shape = self._get_observation_shape()
        
        # Pre-compute market observations (same as original)
        self.precompute_observations = True
        self.precomputed_obs = None
        if self.precompute_observations:
            self._precompute_all_observations()
        
        # Initialize episode state
        self.reset()
        
        # Statistics tracking
        self.episode_stats = {
            'total_episodes': 0,
            'successful_trades': 0,
            'timeout_searches': 0,
            'timeout_exits': 0,
            'stop_loss_exits': 0,
            'take_profit_exits': 0,
            'agent_exits': 0,
            'total_pnl': 0.0,
            'avg_hold_duration': 0.0,
            'avg_wait_duration': 0.0
        }
        
        logger.info(f"═══════════════════════════════════════════════════════════")
        logger.info(f" TRADE-BASED Multi-Timeframe Environment Initialized")
        logger.info(f"═══════════════════════════════════════════════════════════")
        logger.info(f"  Asset: {self.asset}")
        logger.info(f"  Timeframes: {self.timeframes}")
        logger.info(f"  Action Space: 2 (phase-dependent)")
        logger.info(f"  Episode Structure: 1 trade per episode")
        logger.info(f"  Max wait: {self.trade_config.max_wait_steps} steps")
        logger.info(f"  Max hold: {self.trade_config.max_hold_steps} steps")
        logger.info(f"  Stop Loss: {self.stop_loss*100:.1f}%")
        logger.info(f"  Take Profit: {self.take_profit*100:.1f}%")
        logger.info(f"═══════════════════════════════════════════════════════════")

    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION & SETUP (Same as original)
    # ═══════════════════════════════════════════════════════════════════════════
    
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
        """Synchronize all timeframes to execution timeframe"""
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

    def _encode_asset(self, asset: str) -> np.ndarray:
        """Encode asset as one-hot vector (5 dims)"""
        asset_map = {
            'ETH_USDT': 0, 'ETH/USDT': 0, 'ETH/USD': 0, 'ETHUSD': 0,
            'SOL_USDT': 1, 'SOL/USDT': 1, 'SOL/USD': 1, 'SOLUSD': 1,
            'DOT_USDT': 2, 'DOT/USDT': 2, 'DOT/USD': 2, 'DOTUSD': 2,
            'AVAX_USDT': 3, 'AVAX/USDT': 3, 'AVAX/USD': 3, 'AVAXUSD': 3,
            'ADA_USDT': 4, 'ADA/USDT': 4, 'ADA/USD': 4, 'ADAUSD': 4
        }
        
        encoding = np.zeros(5, dtype=np.float32)
        if asset in asset_map:
            encoding[asset_map[asset]] = 1.0
        return encoding

    def _encode_timeframe(self, timeframe: str) -> np.ndarray:
        """Encode timeframe as one-hot vector (6 dims)"""
        timeframe_map = {'1m': 0, '5m': 1, '15m': 2, '30m': 3, '1h': 4, '4h': 5, '1d': 6}
        
        encoding = np.zeros(6, dtype=np.float32)
        if timeframe in timeframe_map:
            encoding[timeframe_map[timeframe]] = 1.0
        return encoding

    # ═══════════════════════════════════════════════════════════════════════════
    # OBSERVATION SPACE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_observation_shape(self) -> Tuple[int]:
        """
        Calculate observation space dimensions
        
        Same as original PLUS:
        - Phase encoding (2 dims)
        - Phase-specific info (3 dims)
        """
        # Market features from all timeframes
        total_features = sum(len(self.features_dfs[tf].columns) for tf in self.timeframes)
        
        # Position info (5 dims) - same as original
        position_dims = 5
        
        # Account info (5 dims) - same as original  
        account_dims = 5
        
        # Asset encoding (5 dims) - same as original
        asset_dims = 5
        
        # Timeframe encodings
        encoding_dims = len(self.timeframes) * 6
        
        # NEW: Phase encoding (2 dims)
        phase_dims = 2
        
        # NEW: Phase-specific info (3 dims)
        # - wait_duration_normalized OR hold_duration_normalized
        # - unrealized_pnl_pct (0 if searching)
        # - time_pressure (how close to timeout)
        phase_info_dims = 3
        
        total = (total_features + position_dims + account_dims + 
                asset_dims + encoding_dims + phase_dims + phase_info_dims)
        
        return (total,)

    def _precompute_all_observations(self):
        """Pre-compute MARKET observations (phase info added dynamically)"""
        logger.info("   Pre-computing market observations...")
        start_time = time.time()
        
        n_steps = len(self.synchronized_data)
        
        # Calculate market observation size (without phase info)
        market_features = sum(len(self.features_dfs[tf].columns) for tf in self.timeframes)
        position_dims = 5
        account_dims = 5
        asset_dims = 5
        encoding_dims = len(self.timeframes) * 6
        market_obs_size = market_features + position_dims + account_dims + asset_dims + encoding_dims
        
        self.precomputed_obs = np.zeros((n_steps, market_obs_size), dtype=np.float32)
        
        for i in range(self.window_size + 1, n_steps):
            obs = self._calculate_market_observation(i)
            self.precomputed_obs[i] = obs
        
        elapsed = time.time() - start_time
        logger.info(f"   Pre-computed {n_steps - self.window_size:,} market observations in {elapsed:.2f}s")

    def _calculate_market_observation(self, step: int) -> np.ndarray:
        """Calculate market observation for a given step (NO phase info)"""
        obs_parts = []
        
        # Use PREVIOUS bar's data (t-1) to prevent look-ahead
        if step > 0:
            sync_row = self.synchronized_data.iloc[step - 1]
        else:
            sync_row = self.synchronized_data.iloc[step]
        
        # Features from each timeframe
        for tf in self.timeframes:
            tf_index = sync_row[f'{tf}_index']
            
            if tf_index in self.features_dfs[tf].index:
                features = self.features_dfs[tf].loc[tf_index].values
            else:
                features = np.zeros(len(self.features_dfs[tf].columns))
            
            obs_parts.append(features)
        
        # Position information (will be updated dynamically)
        position_info = np.zeros(5, dtype=np.float32)
        obs_parts.append(position_info)
        
        # Account information (will be updated dynamically)
        account_info = np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
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

    def _get_observation(self) -> np.ndarray:
        """
        Get full observation including phase information
        """
        # Get pre-computed market observation
        if self.precompute_observations and self.precomputed_obs is not None:
            market_obs = self.precomputed_obs[self.current_step].copy()
        else:
            market_obs = self._calculate_market_observation(self.current_step)
        
        # Update position info in market_obs
        market_obs = self._update_position_info(market_obs)
        
        # Update account info in market_obs
        market_obs = self._update_account_info(market_obs)
        
        # Add phase encoding (2 dims)
        phase_encoding = np.array([
            1.0 if self.phase == Phase.SEARCHING else 0.0,
            1.0 if self.phase == Phase.IN_TRADE else 0.0
        ], dtype=np.float32)
        
        # Add phase-specific info (3 dims)
        phase_info = self._get_phase_info()
        
        # Combine all
        full_obs = np.concatenate([market_obs, phase_encoding, phase_info])
        
        return full_obs

    def _update_position_info(self, obs: np.ndarray) -> np.ndarray:
        """Update position info section of observation"""
        # Calculate position info indices
        market_features = sum(len(self.features_dfs[tf].columns) for tf in self.timeframes)
        pos_start = market_features
        pos_end = pos_start + 5
        
        current_price = self._get_current_price()
        
        obs[pos_start:pos_end] = np.array([
            float(self.position),
            self.entry_price / current_price if self.entry_price > 0 else 0,
            self.position_size / self.initial_balance if self.position_size > 0 else 0,
            self.unrealized_pnl / self.initial_balance if self.position != Positions.FLAT else 0,
            self._get_hold_duration() / 100
        ], dtype=np.float32)
        
        return obs

    def _update_account_info(self, obs: np.ndarray) -> np.ndarray:
        """Update account info section of observation"""
        # Calculate account info indices
        market_features = sum(len(self.features_dfs[tf].columns) for tf in self.timeframes)
        acc_start = market_features + 5  # After position info
        acc_end = acc_start + 5
        
        portfolio_value = self._get_portfolio_value()
        
        obs[acc_start:acc_end] = np.array([
            self.balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            self.total_fees_paid / self.initial_balance
        ], dtype=np.float32)
        
        return obs

    def _get_phase_info(self) -> np.ndarray:
        """
        Get phase-specific information (3 dims)
        
        This helps the agent understand its current context:
        - How long it's been searching/holding
        - Current P&L if in trade
        - How close to timeout
        """
        config = self.trade_config
        
        if self.phase == Phase.SEARCHING:
            # Searching: duration, 0, time_pressure
            duration_normalized = self.steps_in_phase / config.max_wait_steps
            unrealized_pnl_pct = 0.0
            time_pressure = self.steps_in_phase / config.max_wait_steps
            
        else:  # IN_TRADE
            # In trade: duration, unrealized P&L, time_pressure
            duration_normalized = self.steps_in_phase / config.max_hold_steps
            
            current_price = self._get_current_price()
            if self.entry_price > 0:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl_pct = 0.0
            
            time_pressure = self.steps_in_phase / config.max_hold_steps
        
        return np.array([
            duration_normalized,
            unrealized_pnl_pct,
            time_pressure
        ], dtype=np.float32)

    # ═══════════════════════════════════════════════════════════════════════════
    # CORE ENVIRONMENT METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for new episode
        
        Each episode starts at a random point in the data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random starting position
        config = self.trade_config
        buffer = config.max_wait_steps + config.max_hold_steps + 100
        
        if len(self.synchronized_data) > buffer + self.window_size:
            min_start = self.window_size
            max_start = len(self.synchronized_data) - buffer
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = self.window_size
        
        self.episode_start_step = self.current_step
        
        # ═══════════════════════════════════════════════════════════════
        # Phase state - START IN SEARCHING PHASE
        # ═══════════════════════════════════════════════════════════════
        self.phase = Phase.SEARCHING
        self.steps_in_phase = 0
        
        # Position state
        self.position = Positions.FLAT
        self.entry_price = 0.0
        self.entry_step = 0
        self.position_size = 0.0
        self.entry_fee = 0.0
        
        # Account state
        self.balance = self.initial_balance
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0
        
        # Episode tracking
        self.done = False
        self.truncated = False
        self.trade_result = None
        
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action based on current phase
        
        SEARCHING phase (FLAT):
            action 0 = WAIT (do nothing, keep searching)
            action 1 = ENTER (open long position)
        
        IN_TRADE phase (LONG):
            action 0 = HOLD (keep position)
            action 1 = EXIT (close position)
        
        Returns:
            observation: Next state
            reward: Only non-zero when episode ends (trade completes)
            done: True when episode ends
            truncated: True if ended due to timeout
            info: Additional information
        """
        self.current_step += 1
        self.steps_in_phase += 1
        current_price = self._get_current_price()
        
        reward = 0.0
        info = {'phase': self.phase.name, 'action': action}
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE: SEARCHING FOR ENTRY
        # ═══════════════════════════════════════════════════════════════
        if self.phase == Phase.SEARCHING:
            
            if action == 1:  # ENTER
                # Execute entry
                self._execute_entry(current_price)
                
                # Switch to IN_TRADE phase
                self.phase = Phase.IN_TRADE
                self.steps_in_phase = 0
                
                info['action_name'] = 'ENTER'
                info['entry_price'] = self.entry_price
                info['wait_duration'] = self.current_step - self.episode_start_step
                
            else:  # WAIT
                info['action_name'] = 'WAIT'
                
                # Check for search timeout
                if self.steps_in_phase >= self.trade_config.max_wait_steps:
                    # Penalty for not finding a trade
                    reward = self.trade_config.no_trade_penalty
                    self.done = True
                    self.truncated = True
                    info['termination'] = 'search_timeout'
                    self.episode_stats['timeout_searches'] += 1
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE: IN TRADE - LOOKING FOR EXIT
        # ═══════════════════════════════════════════════════════════════
        elif self.phase == Phase.IN_TRADE:
            
            # First check stop-loss and take-profit
            exit_triggered, exit_reason = self._check_exit_conditions(current_price)
            
            if exit_triggered:
                # Stop-loss or take-profit triggered
                reward = self._execute_exit(current_price, exit_reason)
                self.done = True
                self.truncated = False
                info['termination'] = exit_reason
                info['action_name'] = exit_reason.upper()
                
                if exit_reason == 'stop_loss':
                    self.episode_stats['stop_loss_exits'] += 1
                else:
                    self.episode_stats['take_profit_exits'] += 1
                    
            elif action == 1:  # EXIT (agent decision)
                reward = self._execute_exit(current_price, 'agent')
                self.done = True
                self.truncated = False
                info['termination'] = 'agent_exit'
                info['action_name'] = 'EXIT'
                self.episode_stats['agent_exits'] += 1
                
            else:  # HOLD
                info['action_name'] = 'HOLD'
                
                # Update unrealized P&L
                self.unrealized_pnl = (current_price - self.entry_price) * self.position_size
                
                # Check for hold timeout
                if self.steps_in_phase >= self.trade_config.max_hold_steps:
                    # Force exit with penalty
                    reward = self._execute_exit(current_price, 'timeout')
                    reward += self.trade_config.timeout_exit_penalty
                    self.done = True
                    self.truncated = True
                    info['termination'] = 'hold_timeout'
                    self.episode_stats['timeout_exits'] += 1
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Add trade result to info if done
        if self.done and self.trade_result is not None:
            info['trade_result'] = self.trade_result.__dict__
            self.episode_stats['total_episodes'] += 1
            self.episode_stats['total_pnl'] += self.trade_result.net_pnl
            
            if self.trade_result.net_pnl > 0:
                self.episode_stats['successful_trades'] += 1
        
        return next_obs, reward, self.done, self.truncated, info

    def _execute_entry(self, price: float) -> None:
        """Execute trade entry"""
        # Calculate execution price with slippage
        execution_price = price * (1 + self.slippage)
        
        # Calculate position size (95% of balance)
        available_capital = self.balance * 0.95
        gross_position_size = available_capital / execution_price
        
        # Calculate entry fee
        self.entry_fee = available_capital * self.fee_rate
        coins_lost_to_fees = self.entry_fee / execution_price
        net_position_size = gross_position_size - coins_lost_to_fees
        
        # Update state
        self.position = Positions.LONG
        self.entry_price = execution_price
        self.entry_step = self.current_step
        self.position_size = net_position_size
        self.balance -= available_capital
        self.total_fees_paid += self.entry_fee

    def _execute_exit(self, price: float, exit_reason: str) -> float:
        """
        Execute trade exit and return reward
        
        This is the ONLY place reward is calculated!
        """
        # Calculate execution price with slippage
        execution_price = price * (1 - self.slippage)
        
        # Calculate proceeds and fees
        gross_proceeds = self.position_size * execution_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee
        
        # Calculate P&L
        gross_pnl = (execution_price - self.entry_price) * self.position_size
        total_fees = self.entry_fee + exit_fee
        net_pnl = gross_pnl - exit_fee  # Entry fee already accounted for in position size
        pnl_pct = net_pnl / self.initial_balance
        
        # Calculate hold duration
        hold_duration = self.current_step - self.entry_step
        
        # Store trade result
        self.trade_result = TradeResult(
            entry_price=self.entry_price,
            exit_price=execution_price,
            entry_step=self.entry_step,
            exit_step=self.current_step,
            position_size=self.position_size,
            gross_pnl=gross_pnl,
            fees_paid=total_fees,
            net_pnl=net_pnl,
            pnl_pct=pnl_pct,
            hold_duration=hold_duration,
            exit_reason=exit_reason
        )
        
        # Update account
        self.balance += net_proceeds
        self.realized_pnl += net_pnl
        self.total_fees_paid += exit_fee
        
        # Reset position
        self.position = Positions.FLAT
        self.entry_price = 0.0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        
        # ═══════════════════════════════════════════════════════════════
        # CALCULATE REWARD - The core of trade-based learning!
        # ═══════════════════════════════════════════════════════════════
        reward = self._calculate_trade_reward(pnl_pct, hold_duration, exit_reason)
        
        return reward

    def _calculate_trade_reward(self, pnl_pct: float, hold_duration: int, exit_reason: str) -> float:
        """
        Calculate reward for completed trade
        
        This is where the magic happens - reward structure that encourages:
        1. Profitable trades (obviously)
        2. Holding for appropriate duration
        3. Letting winners run (take-profit is good)
        4. Cutting losers (stop-loss is acceptable)
        """
        config = self.trade_config
        
        # Base reward: scaled P&L percentage
        base_reward = pnl_pct * config.reward_scale
        
        # ═══════════════════════════════════════════════════════════════
        # Duration-based adjustments
        # ═══════════════════════════════════════════════════════════════
        
        if hold_duration >= config.min_hold_for_bonus:
            # Held long enough - add hold bonus
            extra_steps = hold_duration - config.min_hold_for_bonus
            hold_bonus = min(extra_steps * config.hold_duration_bonus, 2.0)  # Cap at 2.0
            base_reward += hold_bonus
        else:
            # Exited too early
            duration_ratio = hold_duration / config.min_hold_for_bonus
            
            if pnl_pct > 0:
                # Profitable but short: reduce reward
                base_reward *= (0.3 + 0.7 * duration_ratio)  # 30-100% of reward
            else:
                # Loss AND short: extra penalty
                base_reward *= (1.0 + 0.5 * (1 - duration_ratio))  # 100-150% of loss
        
        # ═══════════════════════════════════════════════════════════════
        # Exit reason adjustments
        # ═══════════════════════════════════════════════════════════════
        
        if exit_reason == 'take_profit':
            # Take-profit is good! Small bonus
            base_reward += 0.5
            
        elif exit_reason == 'stop_loss':
            # Stop-loss is acceptable risk management
            # No additional penalty (the loss IS the penalty)
            pass
            
        elif exit_reason == 'timeout':
            # Forced exit - already has timeout_exit_penalty added in step()
            pass
            
        elif exit_reason == 'agent':
            # Agent chose to exit - reward based on quality of decision
            if pnl_pct > config.fee_rate * 2 + config.slippage * 2:
                # Profitable after costs - good decision
                base_reward += 0.2
        
        return base_reward

    def _check_exit_conditions(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if stop-loss or take-profit triggered
        
        Returns:
            (triggered: bool, reason: str)
        """
        if self.position == Positions.FLAT:
            return False, ''
        
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        # Stop-loss
        if self.stop_loss and pnl_pct <= -self.stop_loss:
            return True, 'stop_loss'
        
        # Take-profit
        if self.take_profit and pnl_pct >= self.take_profit:
            return True, 'take_profit'
        
        return False, ''

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_current_price(self) -> float:
        """Get current price from execution timeframe"""
        exec_index = self.synchronized_data.index[self.current_step]
        return self.dataframes[self.execution_timeframe].loc[exec_index, 'close']

    def _get_hold_duration(self) -> int:
        """Get current hold duration"""
        if self.phase == Phase.IN_TRADE:
            return self.steps_in_phase
        return 0

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.position == Positions.FLAT:
            return self.balance
        
        current_price = self._get_current_price()
        position_value = self.position_size * current_price
        return self.balance + position_value

    def render(self, mode: str = 'human') -> None:
        """Render current state"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step} | Phase: {self.phase.name}")
            print(f"Position: {self.position.name}")
            
            if self.phase == Phase.SEARCHING:
                print(f"Waiting for entry... ({self.steps_in_phase}/{self.trade_config.max_wait_steps} steps)")
            else:
                current_price = self._get_current_price()
                pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
                print(f"Entry: ${self.entry_price:.2f} | Current: ${current_price:.2f}")
                print(f"Unrealized P&L: {pnl_pct:+.2f}%")
                print(f"Hold duration: {self.steps_in_phase}/{self.trade_config.max_hold_steps} steps")
            
            print(f"Portfolio: ${self._get_portfolio_value():.2f}")
            print(f"{'='*60}")

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics across all episodes"""
        stats = self.episode_stats.copy()
        
        if stats['total_episodes'] > 0:
            stats['win_rate'] = stats['successful_trades'] / stats['total_episodes']
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['total_episodes']
        else:
            stats['win_rate'] = 0.0
            stats['avg_pnl_per_trade'] = 0.0
        
        return stats

    def get_feature_names(self) -> List[str]:
        """Get feature names for explainability"""
        feature_names = []
        
        # Market features
        for tf in self.timeframes:
            for feat in self.features_dfs[tf].columns:
                feature_names.append(f"{tf}_{feat}")
        
        # Position info
        feature_names.extend([
            'position_type', 'entry_price_normalized', 'position_size_normalized',
            'unrealized_pnl_normalized', 'hold_duration_normalized'
        ])
        
        # Account info
        feature_names.extend([
            'balance_normalized', 'portfolio_value_normalized', 'realized_pnl_normalized',
            'unrealized_pnl_account', 'total_fees_normalized'
        ])
        
        # Asset encoding
        feature_names.extend(['asset_eth', 'asset_sol', 'asset_dot', 'asset_avax', 'asset_ada'])
        
        # Timeframe encodings
        for tf in self.timeframes:
            feature_names.extend([
                f'{tf}_enc_1m', f'{tf}_enc_5m', f'{tf}_enc_15m',
                f'{tf}_enc_30m', f'{tf}_enc_1h', f'{tf}_enc_4h'
            ])
        
        # Phase info (NEW)
        feature_names.extend([
            'phase_searching', 'phase_in_trade',
            'duration_normalized', 'unrealized_pnl_pct', 'time_pressure'
        ])
        
        return feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" Trade-Based Multi-Timeframe Environment Test")
    print("=" * 70)
    
    # Create sample data for testing
    print("\nCreating sample data...")
    
    dates_5m = pd.date_range(start='2024-01-01', periods=5000, freq='5min')
    dates_15m = pd.date_range(start='2024-01-01', periods=1667, freq='15min')
    dates_1h = pd.date_range(start='2024-01-01', periods=417, freq='1h')
    
    np.random.seed(42)
    
    # Generate price data with trend
    base_price = 100
    returns_5m = np.random.randn(5000) * 0.002
    prices_5m = base_price * np.exp(np.cumsum(returns_5m))
    
    df_5m = pd.DataFrame({
        'open': prices_5m * (1 + np.random.randn(5000) * 0.001),
        'high': prices_5m * (1 + np.abs(np.random.randn(5000)) * 0.002),
        'low': prices_5m * (1 - np.abs(np.random.randn(5000)) * 0.002),
        'close': prices_5m,
        'volume': np.random.uniform(1000, 10000, 5000)
    }, index=dates_5m)
    
    # Resample for other timeframes
    df_15m = df_5m.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    df_1h = df_5m.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    
    dataframes = {'5m': df_5m, '15m': df_15m, '1h': df_1h}
    
    # Generate features
    features_dfs = {}
    for tf, df in dataframes.items():
        features_dfs[tf] = pd.DataFrame({
            'rsi': np.random.uniform(30, 70, len(df)),
            'macd': np.random.randn(len(df)) * 0.5,
            'bb_position': np.random.uniform(0, 1, len(df)),
            'volume_ratio': np.random.uniform(0.5, 2, len(df)),
            'momentum': np.random.randn(len(df)) * 0.02
        }, index=df.index)
    
    print(f"  5m data: {len(df_5m)} candles")
    print(f"  15m data: {len(df_15m)} candles")
    print(f"  1h data: {len(df_1h)} candles")
    
    # Create environment
    print("\nCreating Trade-Based Environment...")
    
    env = TradeBasedMultiTimeframeEnv(
        dataframes=dataframes,
        features_dfs=features_dfs,
        execution_timeframe='5m',
        initial_balance=10000,
        fee_rate=0.001,
        slippage=0.0005,
        stop_loss=0.02,
        take_profit=0.04,
        asset='ETH_USDT'
    )
    
    print(f"\nObservation shape: {env.observation_space_shape}")
    print(f"Action space: {env.action_space_n}")
    
    # Run test episodes
    print("\n" + "=" * 70)
    print(" Running Test Episodes")
    print("=" * 70)
    
    n_episodes = 10
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Simple strategy: random actions
            action = np.random.randint(0, 2)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if steps > 1000:  # Safety limit
                break
        
        # Print episode result
        termination = info.get('termination', 'unknown')
        
        if 'trade_result' in info:
            tr = info['trade_result']
            print(f"Episode {ep+1}: {termination:15s} | "
                  f"PnL: {tr['pnl_pct']*100:+6.2f}% | "
                  f"Hold: {tr['hold_duration']:3d} steps | "
                  f"Reward: {total_reward:+7.2f}")
        else:
            print(f"Episode {ep+1}: {termination:15s} | "
                  f"No trade | "
                  f"Reward: {total_reward:+7.2f}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print(" Episode Statistics")
    print("=" * 70)
    
    stats = env.get_episode_stats()
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Successful trades: {stats['successful_trades']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  Timeout searches: {stats['timeout_searches']}")
    print(f"  Timeout exits: {stats['timeout_exits']}")
    print(f"  Stop-loss exits: {stats['stop_loss_exits']}")
    print(f"  Take-profit exits: {stats['take_profit_exits']}")
    print(f"  Agent exits: {stats['agent_exits']}")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Avg P&L/trade: ${stats['avg_pnl_per_trade']:.2f}")
    
    print("\n" + "=" * 70)
    print(" Trade-Based Environment Ready!")
    print("=" * 70)