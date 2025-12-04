"""
Multi-Objective RL Trainer for Trade-Based Episodes

This trainer uses the Multi-Objective reward system where the agent
learns 5 separate objectives simultaneously:

1. PNL_QUALITY      - Maximize profit, minimize loss size
2. HOLD_DURATION    - Hold trades longer
3. WIN_ACHIEVED     - Win more trades
4. LOSS_CONTROL     - Cut losers early
5. RISK_REWARD      - Maintain good risk/reward ratios

The agent has separate Q-value heads for each objective and learns
to balance them for optimal trading decisions.

Author: Claude
Version: 1.0
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import torch

# Import our multi-objective components
from src.multi_objective_rewards import (
    MultiObjectiveConfig,
    MultiObjectiveRewardCalculator,
    RewardObjective,
)
from src.multi_objective_dqn import (
    MultiObjectiveAgent,
    MODQNConfig,
    OBJECTIVES,
    NUM_OBJECTIVES
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MOTradeConfig:
    """
    Configuration for Multi-Objective Trade-Based Training
    
    Aligned with trade_based_trainer.py configuration for consistency.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # EPISODE STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════════
    max_wait_steps: int = 200           # Max steps to find entry
    max_hold_steps: int = 300           # Max steps in trade
    min_hold_steps: int = 12            # Minimum hold before agent can exit
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRADING PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    stop_loss_pct: float = 0.03         # 3% stop-loss
    take_profit_pct: float = 0.03       # 3% take-profit
    fee_rate: float = 0.001             # 0.1% fee
    slippage: float = 0.0005            # 0.05% slippage
    initial_balance: float = 10000.0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-OBJECTIVE WEIGHTS
    # ═══════════════════════════════════════════════════════════════════════════
    weight_pnl_quality: float = 0.35
    weight_hold_duration: float = 0.25
    weight_win_achieved: float = 0.15
    weight_loss_control: float = 0.15
    weight_risk_reward: float = 0.10
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RL TRAINING PARAMETERS (aligned with trade_based_trainer)
    # ═══════════════════════════════════════════════════════════════════════════
    num_episodes: int = 8000
    batch_size: int = 512               # Same as trade_based (was 64)
    learning_rate: float = 0.001
    gamma: float = 0.99                 # Discount factor
    
    # Replay buffer
    memory_size: int = 25000            # Same as trade_based rl_memory_size
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    head_hidden_dim: int = 32           # Hidden dim for each objective head
    use_double_dqn: bool = True         # Double DQN for stability
    use_dueling_dqn: bool = True        # Dueling architecture per head
    
    # Target network
    target_update_frequency: int = 10   # Update target every N training steps
    tau: float = 0.005                  # Soft update coefficient (if using soft updates)
    use_soft_update: bool = True        # Use soft updates vs hard updates
    
    # Exploration (aligned with trade_based slower decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9998       # Same as trade_based (was 0.9995)
    
    # Validation
    validation_frequency: int = 200     # Check every N episodes
    validation_episodes: int = 50       # Episodes per validation
    enable_early_stopping: bool = False # Disabled by default
    early_stopping_patience: int = 9999
    early_stopping_min_delta: float = 0.1
    
    # Logging
    log_interval: int = 50              # Same as trade_based (was 100)
    save_interval: int = 200            # Same as trade_based (was 1000)
    
    def get_objective_weights(self) -> Dict[str, float]:
        return {
            'pnl_quality': self.weight_pnl_quality,
            'hold_duration': self.weight_hold_duration,
            'win_achieved': self.weight_win_achieved,
            'loss_control': self.weight_loss_control,
            'risk_reward': self.weight_risk_reward,
        }


class Phase:
    """Trading phases"""
    SEARCHING = 0
    IN_TRADE = 1


class MultiObjectiveTradingEnv:
    """
    Trading Environment that returns Multi-Objective rewards
    
    This is a simplified version focused on the multi-objective aspect.
    Integrates with the existing data pipeline.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        feature_data: pd.DataFrame,
        config: MOTradeConfig,
        asset: str = "UNKNOWN"
    ):
        self.price_data = price_data
        self.feature_data = feature_data
        self.config = config
        self.asset = asset
        
        # Validate data
        assert len(price_data) == len(feature_data), "Price and feature data must match"
        assert 'close' in price_data.columns, "Price data must have 'close' column"
        
        self.n_steps = len(price_data)
        self.feature_dim = feature_data.shape[1]
        
        # State dimension = features + position info
        # Position info: [in_trade, unrealized_pnl, steps_in_trade, distance_to_tp, distance_to_sl]
        self.position_info_dim = 5
        self.state_dim = self.feature_dim + self.position_info_dim
        
        # Action space: 2 actions (context-dependent meaning)
        # SEARCHING: 0=WAIT, 1=ENTER
        # IN_TRADE: 0=HOLD, 1=EXIT
        self.action_dim = 2
        
        # Reward calculator
        self.reward_calculator = MultiObjectiveRewardCalculator(
            MultiObjectiveConfig(
                stop_loss_pct=config.stop_loss_pct,
                weight_pnl_quality=config.weight_pnl_quality,
                weight_hold_duration=config.weight_hold_duration,
                weight_win_achieved=config.weight_win_achieved,
                weight_loss_control=config.weight_loss_control,
                weight_risk_reward=config.weight_risk_reward,
            )
        )
        
        # Episode tracking
        self.reset()
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment for new episode"""
        
        # Random start position (with buffer for warmup and episode length)
        buffer = 100
        max_start = self.n_steps - self.config.max_wait_steps - self.config.max_hold_steps - buffer
        
        if start_idx is not None:
            self.current_idx = start_idx
        else:
            self.current_idx = np.random.randint(buffer, max(buffer + 1, max_start))
        
        # Reset state
        self.phase = Phase.SEARCHING
        self.steps_in_phase = 0
        self.done = False
        
        # Trade state
        self.entry_price = 0.0
        self.entry_idx = 0
        self.position_size = 0.0
        self.unrealized_pnl = 0.0
        
        # Track excursions for risk/reward calculation
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
        
        # Episode stats
        self.episode_info = {
            'entry_wait_steps': 0,
            'hold_duration': 0,
            'pnl_pct': 0.0,
            'exit_reason': None,
        }
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation"""
        
        # Market features
        features = self.feature_data.iloc[self.current_idx].values.astype(np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Position information
        current_price = self.price_data.iloc[self.current_idx]['close']
        
        if self.phase == Phase.IN_TRADE and self.entry_price > 0:
            in_trade = 1.0
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            steps_in_trade = self.steps_in_phase / self.config.max_hold_steps
            
            # Distance to TP/SL (normalized)
            distance_to_tp = (self.config.take_profit_pct - unrealized_pnl) / self.config.take_profit_pct
            distance_to_sl = (unrealized_pnl + self.config.stop_loss_pct) / self.config.stop_loss_pct
        else:
            in_trade = 0.0
            unrealized_pnl = 0.0
            steps_in_trade = 0.0
            distance_to_tp = 1.0
            distance_to_sl = 1.0
        
        position_info = np.array([
            in_trade,
            unrealized_pnl * 10,  # Scale for visibility
            steps_in_trade,
            distance_to_tp,
            distance_to_sl,
        ], dtype=np.float32)
        
        # Combine
        state = np.concatenate([features, position_info])
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, float], bool, Dict]:
        """
        Execute action and return multi-objective rewards
        
        Returns:
            state: Next state
            rewards: Dict of rewards per objective
            done: Whether episode is finished
            info: Additional information
        """
        info = {'phase': self.phase, 'action': action}
        rewards = {obj: 0.0 for obj in OBJECTIVES}
        
        current_price = self.price_data.iloc[self.current_idx]['close']
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE: SEARCHING FOR ENTRY
        # ═══════════════════════════════════════════════════════════════
        if self.phase == Phase.SEARCHING:
            
            if action == 1:  # ENTER
                self._enter_trade(current_price)
                info['action_name'] = 'ENTER'
                self.episode_info['entry_wait_steps'] = self.steps_in_phase
                
            else:  # WAIT
                info['action_name'] = 'WAIT'
                self.steps_in_phase += 1
                self.current_idx += 1
                
                # Check timeout
                if self.steps_in_phase >= self.config.max_wait_steps:
                    self.done = True
                    info['exit_reason'] = 'search_timeout'
                    # Small penalty for not trading
                    rewards['pnl_quality'] = -0.5
                    rewards['win_achieved'] = -0.2
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE: IN TRADE
        # ═══════════════════════════════════════════════════════════════
        elif self.phase == Phase.IN_TRADE:
            
            # Update unrealized P&L
            self.unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            
            # Track excursions
            if self.unrealized_pnl > self.max_favorable_excursion:
                self.max_favorable_excursion = self.unrealized_pnl
            if self.unrealized_pnl < self.max_adverse_excursion:
                self.max_adverse_excursion = self.unrealized_pnl
            
            # Check stop-loss
            if self.unrealized_pnl <= -self.config.stop_loss_pct:
                rewards = self._exit_trade(-self.config.stop_loss_pct, 'stop_loss')
                self.done = True
                info['exit_reason'] = 'stop_loss'
                info['action_name'] = 'STOP_LOSS'
            
            # Check take-profit
            elif self.unrealized_pnl >= self.config.take_profit_pct:
                rewards = self._exit_trade(self.config.take_profit_pct, 'take_profit')
                self.done = True
                info['exit_reason'] = 'take_profit'
                info['action_name'] = 'TAKE_PROFIT'
            
            # Agent wants to exit
            elif action == 1:
                # Check minimum hold requirement
                if self.steps_in_phase < self.config.min_hold_steps:
                    # Can't exit yet - force hold
                    info['action_name'] = 'HOLD (forced)'
                    info['forced_hold'] = True
                    self.steps_in_phase += 1
                    self.current_idx += 1
                else:
                    # Can exit
                    rewards = self._exit_trade(self.unrealized_pnl, 'agent')
                    self.done = True
                    info['exit_reason'] = 'agent'
                    info['action_name'] = 'EXIT'
            
            # Hold
            else:
                info['action_name'] = 'HOLD'
                self.steps_in_phase += 1
                self.current_idx += 1
                
                # Check timeout
                if self.steps_in_phase >= self.config.max_hold_steps:
                    rewards = self._exit_trade(self.unrealized_pnl, 'timeout')
                    self.done = True
                    info['exit_reason'] = 'timeout'
        
        # Get next state
        next_state = self._get_state() if not self.done else np.zeros(self.state_dim)
        
        # Add episode info
        info['episode_info'] = self.episode_info
        
        return next_state, rewards, self.done, info
    
    def _enter_trade(self, price: float):
        """Enter a trade"""
        self.phase = Phase.IN_TRADE
        self.entry_price = price * (1 + self.config.slippage)  # Slippage on entry
        self.entry_idx = self.current_idx
        self.position_size = self.config.initial_balance / self.entry_price
        self.steps_in_phase = 0
        
        # Reset excursions
        self.max_favorable_excursion = 0.0
        self.max_adverse_excursion = 0.0
    
    def _exit_trade(self, pnl_pct: float, exit_reason: str) -> Dict[str, float]:
        """
        Exit trade and calculate multi-objective rewards
        
        Returns:
            Dictionary of rewards per objective
        """
        exit_price = self.entry_price * (1 + pnl_pct)
        
        # Account for fees
        total_fees = 2 * self.config.fee_rate  # Entry + exit
        net_pnl_pct = pnl_pct - total_fees
        
        # Store episode info
        self.episode_info['hold_duration'] = self.steps_in_phase
        self.episode_info['pnl_pct'] = net_pnl_pct
        self.episode_info['exit_reason'] = exit_reason
        
        # Calculate multi-objective rewards
        rewards = self.reward_calculator.calculate(
            pnl_pct=net_pnl_pct,
            hold_duration=self.steps_in_phase,
            exit_reason=exit_reason,
            entry_price=self.entry_price,
            exit_price=exit_price,
            max_favorable_excursion=self.max_favorable_excursion,
            max_adverse_excursion=self.max_adverse_excursion,
        )
        
        return rewards


class MultiObjectiveTrainer:
    """
    Trainer for Multi-Objective RL Trading Agent
    
    Handles:
    - Environment management
    - Agent training loop
    - Per-objective statistics
    - Logging and visualization
    - Model saving
    """
    
    def __init__(
        self,
        train_data: Dict[str, Dict[str, pd.DataFrame]],
        val_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        config: Optional[MOTradeConfig] = None,
        output_dir: str = "models/multi_objective"
    ):
        """
        Args:
            train_data: {asset: {'prices': df, 'features': df}}
            val_data: Same structure for validation
            config: Training configuration
            output_dir: Where to save models
        """
        self.train_data = train_data
        self.val_data = val_data
        self.config = config or MOTradeConfig()
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get dimensions from first asset
        first_asset = list(train_data.keys())[0]
        sample_features = train_data[first_asset]['features']
        feature_dim = sample_features.shape[1]
        self.state_dim = feature_dim + 5  # +5 for position info
        
        # Create agent config (pass all parameters from MOTradeConfig)
        dqn_config = MODQNConfig(
            state_dim=self.state_dim,
            action_dim=2,
            hidden_dims=self.config.hidden_dims,
            head_hidden_dim=self.config.head_hidden_dim,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            batch_size=self.config.batch_size,
            memory_size=self.config.memory_size,
            tau=self.config.tau,
            target_update_frequency=self.config.target_update_frequency,
            use_soft_update=self.config.use_soft_update,
            epsilon_start=self.config.epsilon_start,
            epsilon_end=self.config.epsilon_end,
            epsilon_decay=self.config.epsilon_decay,
            use_dueling=self.config.use_dueling_dqn,
            use_double_dqn=self.config.use_double_dqn,
            objective_weights=self.config.get_objective_weights(),
        )
        
        self.agent = MultiObjectiveAgent(
            state_dim=self.state_dim,
            action_dim=2,
            config=dqn_config
        )
        
        # Create environments for each asset
        self.train_envs = {}
        for asset, data in train_data.items():
            self.train_envs[asset] = MultiObjectiveTradingEnv(
                price_data=data['prices'],
                feature_data=data['features'],
                config=self.config,
                asset=asset
            )
        
        self.assets = list(self.train_envs.keys())
        
        # ═══════════════════════════════════════════════════════════════
        # TRACKING
        # ═══════════════════════════════════════════════════════════════
        self.episode_rewards = {obj: [] for obj in OBJECTIVES}
        self.episode_totals = []
        self.episode_pnls = []
        self.episode_hold_durations = []
        self.episode_win_rate = []
        
        self.objective_history = {obj: [] for obj in OBJECTIVES}
        
        logger.info(f"Multi-Objective Trainer initialized")
        logger.info(f"  Assets: {self.assets}")
        logger.info(f"  State dim: {self.state_dim}")
        logger.info(f"  Objectives: {OBJECTIVES}")
        logger.info(f"  Device: {self.agent.device}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Training results and statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-OBJECTIVE RL TRAINING")
        logger.info("=" * 70)
        logger.info(f"Training for {self.config.num_episodes} episodes...")
        
        start_time = datetime.now()
        
        # Training stats
        recent_pnls = []
        recent_holds = []
        recent_wins = []
        
        for episode in range(self.config.num_episodes):
            # Select random asset
            asset = np.random.choice(self.assets)
            env = self.train_envs[asset]
            
            # Run episode
            state = env.reset()
            episode_rewards = {obj: 0.0 for obj in OBJECTIVES}
            done = False
            
            while not done:
                # Select action
                action = self.agent.select_action(state, training=True)
                
                # Take step
                next_state, rewards, done, info = env.step(action)
                
                # Accumulate rewards
                for obj in OBJECTIVES:
                    episode_rewards[obj] += rewards[obj]
                
                # Store transition (only when episode ends with actual rewards)
                if done and info.get('exit_reason') != 'search_timeout':
                    self.agent.store_transition(state, action, rewards, next_state, done)
                
                state = next_state
            
            # Train
            if len(self.agent.memory) >= self.config.batch_size:
                losses = self.agent.train_step(self.config.batch_size)
            
            # Track episode results
            ep_info = env.episode_info
            
            if ep_info['exit_reason'] and ep_info['exit_reason'] != 'search_timeout':
                # Track per-objective rewards
                for obj in OBJECTIVES:
                    self.episode_rewards[obj].append(episode_rewards[obj])
                    self.objective_history[obj].append(episode_rewards[obj])
                
                # Track overall stats
                total_reward = sum(
                    episode_rewards[obj] * self.config.get_objective_weights()[obj]
                    for obj in OBJECTIVES
                )
                self.episode_totals.append(total_reward)
                
                pnl = ep_info['pnl_pct']
                hold = ep_info['hold_duration']
                
                self.episode_pnls.append(pnl)
                self.episode_hold_durations.append(hold)
                
                recent_pnls.append(pnl)
                recent_holds.append(hold)
                recent_wins.append(1 if pnl > 0 else 0)
                
                # Keep recent lists bounded
                if len(recent_pnls) > 100:
                    recent_pnls.pop(0)
                    recent_holds.pop(0)
                    recent_wins.pop(0)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                self._log_progress(
                    episode + 1,
                    recent_pnls,
                    recent_holds,
                    recent_wins
                )
            
            # Save checkpoint
            if (episode + 1) % self.config.save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        # Final save
        self._save_checkpoint(self.config.num_episodes, final=True)
        
        # Generate report
        duration = (datetime.now() - start_time).total_seconds()
        results = self._generate_final_report(duration)
        
        return results
    
    def _log_progress(
        self,
        episode: int,
        recent_pnls: List[float],
        recent_holds: List[int],
        recent_wins: List[int]
    ):
        """Log training progress"""
        
        if len(recent_pnls) == 0:
            return
        
        avg_pnl = np.mean(recent_pnls) * 100
        avg_hold = np.mean(recent_holds)
        win_rate = np.mean(recent_wins) * 100
        epsilon = self.agent.epsilon
        
        # Per-objective averages
        obj_avgs = {}
        for obj in OBJECTIVES:
            recent = self.objective_history[obj][-100:]
            obj_avgs[obj] = np.mean(recent) if recent else 0.0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Episode {episode}/{self.config.num_episodes}")
        logger.info(f"{'='*70}")
        logger.info(f"  Avg P&L:     {avg_pnl:+.3f}%")
        logger.info(f"  Avg Hold:    {avg_hold:.1f} steps")
        logger.info(f"  Win Rate:    {win_rate:.1f}%")
        logger.info(f"  Epsilon:     {epsilon:.4f}")
        logger.info(f"  ")
        logger.info(f"  PER-OBJECTIVE REWARDS (last 100 episodes):")
        
        for obj in OBJECTIVES:
            weight = self.config.get_objective_weights()[obj]
            logger.info(f"    {obj:15s}: {obj_avgs[obj]:+.3f} (weight: {weight:.2f})")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint"""
        suffix = "final" if final else f"ep{episode}"
        path = os.path.join(self.output_dir, f"mo_agent_{suffix}.pt")
        self.agent.save(path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _generate_final_report(self, duration: float) -> Dict[str, Any]:
        """Generate final training report"""
        
        report = {
            'training_duration_seconds': duration,
            'num_episodes': self.config.num_episodes,
            'num_trades': len(self.episode_pnls),
        }
        
        if len(self.episode_pnls) > 0:
            report['overall_stats'] = {
                'avg_pnl_pct': float(np.mean(self.episode_pnls) * 100),
                'total_pnl_pct': float(np.sum(self.episode_pnls) * 100),
                'win_rate': float(np.mean([1 if p > 0 else 0 for p in self.episode_pnls]) * 100),
                'avg_hold_duration': float(np.mean(self.episode_hold_durations)),
            }
            
            # Per-objective stats
            report['objective_stats'] = {}
            for obj in OBJECTIVES:
                rewards = self.objective_history[obj]
                if rewards:
                    report['objective_stats'][obj] = {
                        'mean': float(np.mean(rewards)),
                        'std': float(np.std(rewards)),
                        'min': float(np.min(rewards)),
                        'max': float(np.max(rewards)),
                        'final_100_mean': float(np.mean(rewards[-100:])),
                    }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_final_summary(report)
        
        return report
    
    def _print_final_summary(self, report: Dict[str, Any]):
        """Print final training summary"""
        
        print("\n" + "=" * 70)
        print("MULTI-OBJECTIVE TRAINING COMPLETE")
        print("=" * 70)
        
        print(f"\nDuration: {report['training_duration_seconds']:.1f}s")
        print(f"Episodes: {report['num_episodes']}")
        print(f"Trades: {report['num_trades']}")
        
        if 'overall_stats' in report:
            stats = report['overall_stats']
            print(f"\nOVERALL PERFORMANCE:")
            print(f"  Avg P&L:        {stats['avg_pnl_pct']:+.3f}%")
            print(f"  Total P&L:      {stats['total_pnl_pct']:+.2f}%")
            print(f"  Win Rate:       {stats['win_rate']:.1f}%")
            print(f"  Avg Hold:       {stats['avg_hold_duration']:.1f} steps")
        
        if 'objective_stats' in report:
            print(f"\nPER-OBJECTIVE PERFORMANCE:")
            print(f"{'Objective':<15} {'Mean':>8} {'Final 100':>10} {'Trend':>8}")
            print("-" * 45)
            
            for obj in OBJECTIVES:
                if obj in report['objective_stats']:
                    s = report['objective_stats'][obj]
                    trend = "↑" if s['final_100_mean'] > s['mean'] else "↓"
                    print(f"{obj:<15} {s['mean']:>+8.3f} {s['final_100_mean']:>+10.3f} {trend:>8}")
        
        print("\n" + "=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION WITH EXISTING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def convert_existing_data_format(
    dataframes: Dict[str, pd.DataFrame],
    features_dfs: Dict[str, pd.DataFrame],
    execution_timeframe: str = '5m'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert existing multi-timeframe data to simple format for MO training
    
    Args:
        dataframes: {timeframe: ohlcv_df}
        features_dfs: {timeframe: features_df}
        execution_timeframe: Which timeframe to use
        
    Returns:
        prices_df, features_df aligned
    """
    price_df = dataframes[execution_timeframe].copy()
    features = features_dfs[execution_timeframe].copy()
    
    # Ensure we have required price columns
    required_cols = ['close']
    optional_cols = ['high', 'low', 'open', 'volume']
    
    # Check close exists
    if 'close' not in price_df.columns:
        raise ValueError("Price data must have 'close' column")
    
    # Add optional columns if missing
    for col in optional_cols:
        if col not in price_df.columns:
            if col == 'high':
                price_df[col] = price_df['close'] * 1.001
            elif col == 'low':
                price_df[col] = price_df['close'] * 0.999
            elif col == 'open':
                price_df[col] = price_df['close']
            elif col == 'volume':
                price_df[col] = 1000
    
    # Align indices
    common_idx = price_df.index.intersection(features.index)
    prices = price_df.loc[common_idx][['close', 'high', 'low', 'open', 'volume']]
    features = features.loc[common_idx]
    
    return prices, features


def prepare_from_trade_based_data(
    asset_data: Dict[str, Any],
    execution_timeframe: str = '5m'
) -> Dict[str, pd.DataFrame]:
    """
    Convert data from trade_based_trainer format to MO format
    
    Args:
        asset_data: {'dataframes': {...}, 'features_dfs': {...}}
        
    Returns:
        {'prices': df, 'features': df}
    """
    prices, features = convert_existing_data_format(
        asset_data['dataframes'],
        asset_data['features_dfs'],
        execution_timeframe
    )
    
    return {'prices': prices, 'features': features}


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create synthetic data for testing
    print("Creating synthetic data for testing...")
    
    n_samples = 10000
    
    # Synthetic price data
    np.random.seed(42)
    returns = np.random.randn(n_samples) * 0.001
    prices = 100 * np.exp(np.cumsum(returns))
    
    price_df = pd.DataFrame({
        'close': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'open': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Synthetic features (50 features)
    feature_df = pd.DataFrame(
        np.random.randn(n_samples, 50),
        columns=[f'feature_{i}' for i in range(50)]
    )
    
    # Create training data structure
    train_data = {
        'BTC_USDT': {
            'prices': price_df,
            'features': feature_df
        }
    }
    
    # Create config
    config = MOTradeConfig(
        num_episodes=500,
        log_interval=50,
        save_interval=250,
    )
    
    # Create trainer
    trainer = MultiObjectiveTrainer(
        train_data=train_data,
        config=config,
        output_dir="models/mo_test"
    )
    
    # Train
    print("\nStarting training...")
    results = trainer.train()
    
    print("\nTraining complete!")
    print(f"Results saved to: models/mo_test/")