"""
MULTI-OBJECTIVE EXTENSION FOR TRADE-BASED TRAINER
==================================================

This module extends the existing trade_based_trainer with multi-objective rewards.

USAGE:
------
1. Copy this file to src/multi_objective_extension.py
2. In trade_based_trainer.py, add: from src.multi_objective_extension import ...
3. Enable multi-objective mode in config: 'use_multi_objective': True

The extension:
- Adds multi-objective reward calculation to TradeBasedMultiTimeframeEnv
- Creates MultiObjectiveDQNAgent that wraps existing DQNAgent
- Keeps ALL existing infrastructure (multi-timeframe, pre-computed obs, etc.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: MULTI-OBJECTIVE REWARD CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Objective names (5 objectives)
OBJECTIVES = ['pnl_quality', 'hold_duration', 'win_achieved', 'loss_control', 'risk_reward']


@dataclass
class MORewardConfig:
    """Configuration for multi-objective rewards"""
    
    # Objective weights (must sum to 1.0 for proper scaling)
    weight_pnl_quality: float = 0.40
    weight_hold_duration: float = 0.05
    weight_win_achieved: float = 0.15
    weight_loss_control: float = 0.20
    weight_risk_reward: float = 0.20
    
    # Trading parameters
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.03
    fee_rate: float = 0.001
    
    # Hold duration settings
    min_hold_for_bonus: int = 12      # 1 hour at 5m
    max_hold_steps: int = 300         # 25 hours at 5m
    target_hold_steps: int = 48       # 4 hours at 5m (ideal hold)
    
    # Reward scaling
    pnl_scale: float = 50.0           # Scale PnL rewards
    
    def get_weights(self) -> Dict[str, float]:
        return {
            'pnl_quality': self.weight_pnl_quality,
            'hold_duration': self.weight_hold_duration,
            'win_achieved': self.weight_win_achieved,
            'loss_control': self.weight_loss_control,
            'risk_reward': self.weight_risk_reward,
        }


class MultiObjectiveRewardCalculator:
    """
    Calculates 5 separate reward signals from trade outcomes
    
    Each objective teaches a different aspect of good trading:
    1. pnl_quality   - Maximize profit, minimize loss magnitude
    2. hold_duration - Hold trades for meaningful moves
    3. win_achieved  - Win more trades than you lose
    4. loss_control  - Cut losers early, before stop-loss
    5. risk_reward   - Achieve good risk/reward ratios
    """
    
    def __init__(self, config: MORewardConfig):
        self.config = config
    
    def calculate(
        self,
        pnl_pct: float,
        hold_duration: int,
        exit_reason: str,
        max_favorable_excursion: float = 0.0,
        max_adverse_excursion: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate multi-objective rewards for a completed trade
        
        Args:
            pnl_pct: Net P&L as percentage (e.g., 0.02 = 2%)
            hold_duration: Number of steps held
            exit_reason: 'agent', 'stop_loss', 'take_profit', 'timeout'
            max_favorable_excursion: Best unrealized P&L during trade
            max_adverse_excursion: Worst unrealized P&L during trade (negative)
        
        Returns:
            Dict with reward for each objective
        """
        rewards = {}
        
        # 1. PNL_QUALITY: Scaled P&L with asymmetric treatment
        rewards['pnl_quality'] = self._calc_pnl_quality(pnl_pct)
        
        # 2. HOLD_DURATION: Reward for holding appropriately
        rewards['hold_duration'] = self._calc_hold_duration(pnl_pct, hold_duration)
        
        # 3. WIN_ACHIEVED: Binary win/loss signal
        rewards['win_achieved'] = self._calc_win_achieved(pnl_pct)
        
        # 4. LOSS_CONTROL: Reward for cutting losers early
        rewards['loss_control'] = self._calc_loss_control(pnl_pct, exit_reason)
        
        # 5. RISK_REWARD: Based on excursions
        rewards['risk_reward'] = self._calc_risk_reward(
            pnl_pct, max_favorable_excursion, max_adverse_excursion
        )
        
        return rewards
    
    def _calc_pnl_quality(self, pnl_pct: float) -> float:
        """
        PNL Quality objective
        
        - Wins: Scaled reward (bigger wins = bigger rewards)
        - Losses: Scaled but capped (small losses punished, big losses capped)
        """
        if pnl_pct > 0:
            # Wins: aggressive scaling
            return pnl_pct * self.config.pnl_scale * 1.5
        else:
            # Losses: capped scaling
            scaled = pnl_pct * self.config.pnl_scale
            return max(scaled, -1.0)  # Cap at -1.0
    
    def _calc_hold_duration(self, pnl_pct: float, hold_duration: int) -> float:
        """
        Hold Duration objective
        
        - Reward for holding winning trades longer
        - Penalty for holding losing trades too long
        - Optimal is around target_hold_steps
        """
        cfg = self.config
        
        if pnl_pct > 0:
            # WINNING TRADE: reward longer holds up to target
            if hold_duration >= cfg.target_hold_steps:
                # Held long enough - full reward
                return 1.0
            elif hold_duration >= cfg.min_hold_for_bonus:
                # Partial reward
                progress = (hold_duration - cfg.min_hold_for_bonus) / (cfg.target_hold_steps - cfg.min_hold_for_bonus)
                return 0.3 + 0.7 * progress
            else:
                # Too short - reduced reward
                return 0.1 * (hold_duration / cfg.min_hold_for_bonus)
        else:
            # LOSING TRADE: quicker exit is better (up to a point)
            if hold_duration <= cfg.min_hold_for_bonus:
                # Cut quickly - good!
                return 0.5
            else:
                # Held too long - penalty
                excess = hold_duration - cfg.min_hold_for_bonus
                penalty = min(excess / cfg.target_hold_steps, 1.0)
                return -0.5 * penalty
    
    def _calc_win_achieved(self, pnl_pct: float) -> float:
        """
        Win Achieved objective
        
        Binary signal with magnitude based on how much you won/lost
        """
        if pnl_pct > 0:
            # Win - reward scales with size but capped
            return min(1.0, pnl_pct / self.config.take_profit_pct)
        else:
            # Loss - penalty scales with size but capped
            return max(-1.0, pnl_pct / self.config.stop_loss_pct)
    
    def _calc_loss_control(self, pnl_pct: float, exit_reason: str) -> float:
        """
        Loss Control objective
        
        Reward for cutting losers before stop-loss
        Penalty for hitting stop-loss or holding losers too long
        """
        cfg = self.config
        
        if pnl_pct > 0:
            # Winning trade - neutral for this objective
            return 0.0
        
        # Losing trade
        if exit_reason == 'agent':
            # Agent chose to exit - reward based on how much was saved
            loss_magnitude = abs(pnl_pct)
            if loss_magnitude < cfg.stop_loss_pct * 0.5:
                # Cut very early - great!
                return 1.0
            elif loss_magnitude < cfg.stop_loss_pct:
                # Cut before stop-loss - good
                saved = cfg.stop_loss_pct - loss_magnitude
                return 0.5 + 0.5 * (saved / cfg.stop_loss_pct)
            else:
                # Somehow lost more than stop-loss?
                return -0.5
        
        elif exit_reason == 'stop_loss':
            # Hit stop-loss - bad, could have cut earlier
            return -1.0
        
        elif exit_reason == 'timeout':
            # Timed out with a loss - also bad
            return -0.8
        
        return 0.0
    
    def _calc_risk_reward(
        self,
        pnl_pct: float,
        max_favorable: float,
        max_adverse: float
    ) -> float:
        """
        Risk/Reward objective
        
        Reward for achieving good risk/reward ratios
        """
        # Avoid division by zero
        if abs(max_adverse) < 0.001:
            max_adverse = -0.001
        
        if max_favorable <= 0:
            max_favorable = 0.001
        
        # Calculate realized risk/reward
        if pnl_pct > 0:
            # Won: reward = profit / max_drawdown
            rr_ratio = pnl_pct / abs(max_adverse)
            # Scale: 1.0 ratio = 0.5 reward, 2.0 ratio = 1.0 reward
            return min(1.0, rr_ratio / 2.0)
        else:
            # Lost: penalty based on how much profit was given back
            if max_favorable > 0:
                # Had profit but gave it back
                giveback = max_favorable - pnl_pct
                penalty = giveback / max_favorable
                return -min(1.0, penalty)
            else:
                # Never had profit - small penalty
                return -0.3
    
    def get_weighted_reward(self, rewards: Dict[str, float]) -> float:
        """Convert multi-objective rewards to single scalar using weights"""
        weights = self.config.get_weights()
        total = sum(rewards[obj] * weights[obj] for obj in OBJECTIVES)
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MULTI-HEAD DQN NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadDQNetwork(nn.Module):
    """
    DQN with separate heads for each objective
    
    Architecture:
    - Shared feature extraction layers
    - 5 separate output heads (one per objective)
    - Optional dueling architecture per head
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [128, 64],
        head_hidden_dim: int = 32,
        num_objectives: int = 5,
        use_dueling: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        self.use_dueling = use_dueling
        
        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.shared_out_dim = hidden_dims[-1]
        
        # Per-objective heads
        self.heads = nn.ModuleList()
        
        for _ in range(num_objectives):
            if use_dueling:
                head = DuelingHead(self.shared_out_dim, action_dim, head_hidden_dim)
            else:
                head = nn.Sequential(
                    nn.Linear(self.shared_out_dim, head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, action_dim)
                )
            self.heads.append(head)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning Q-values for each objective
        
        Returns:
            Dict mapping objective name to Q-values tensor [batch, actions]
        """
        shared_features = self.shared(state)
        
        q_values = {}
        for i, obj_name in enumerate(OBJECTIVES):
            q_values[obj_name] = self.heads[i](shared_features)
        
        return q_values
    
    def get_combined_q_values(
        self,
        state: torch.Tensor,
        weights: Dict[str, float]
    ) -> torch.Tensor:
        """
        Get weighted combination of Q-values for action selection
        
        Args:
            state: State tensor
            weights: Objective weights
        
        Returns:
            Combined Q-values tensor [batch, actions]
        """
        q_dict = self.forward(state)
        
        combined = torch.zeros_like(q_dict[OBJECTIVES[0]])
        for obj_name, q_vals in q_dict.items():
            combined += weights[obj_name] * q_vals
        
        return combined


class DuelingHead(nn.Module):
    """Dueling architecture head for one objective"""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: MULTI-OBJECTIVE REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class MOReplayBuffer:
    """
    Replay buffer that stores multi-objective rewards
    
    Each experience contains:
    - state, action, next_state, done (as usual)
    - rewards: Dict[objective, reward]
    """
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Rewards per objective
        self.rewards = {obj: np.zeros(capacity, dtype=np.float32) for obj in OBJECTIVES}
        
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        rewards: Dict[str, float],
        next_state: np.ndarray,
        done: bool
    ):
        """Add experience to buffer"""
        idx = self.position
        
        self.states[idx] = state
        self.actions[idx] = action
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        for obj in OBJECTIVES:
            self.rewards[obj][idx] = rewards.get(obj, 0.0)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch"""
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        
        rewards = {
            obj: torch.FloatTensor(self.rewards[obj][indices])
            for obj in OBJECTIVES
        }
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: MULTI-OBJECTIVE DQN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MODQNConfig:
    """Configuration for Multi-Objective DQN Agent"""
    
    # Dimensions
    state_dim: int = 188
    action_dim: int = 2
    
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    head_hidden_dim: int = 32
    use_dueling: bool = True
    use_double_dqn: bool = True
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 256
    gamma: float = 0.99
    
    # Memory
    memory_size: int = 50000
    min_memory_size: int = 1000
    
    # Updates
    update_every: int = 4
    target_update_every: int = 1000
    tau: float = 0.005
    use_soft_update: bool = True
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9998
    
    # Objective weights
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'pnl_quality': 0.40,
        'hold_duration': 0.05,
        'win_achieved': 0.15,
        'loss_control': 0.20,
        'risk_reward': 0.20,
    })


class MultiObjectiveDQNAgent:
    """
    DQN Agent with multi-objective learning
    
    Uses same interface as regular DQNAgent but:
    - Stores multi-objective rewards
    - Trains separate heads for each objective
    - Combines Q-values using weights for action selection
    """
    
    def __init__(self, config: MODQNConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.q_network = MultiHeadDQNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            head_hidden_dim=config.head_hidden_dim,
            num_objectives=len(OBJECTIVES),
            use_dueling=config.use_dueling
        ).to(self.device)
        
        self.target_network = MultiHeadDQNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            head_hidden_dim=config.head_hidden_dim,
            num_objectives=len(OBJECTIVES),
            use_dueling=config.use_dueling
        ).to(self.device)
        
        # Copy weights to target
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Memory
        self.memory = MOReplayBuffer(config.memory_size, config.state_dim)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Step counter
        self.step_count = 0
        self.update_count = 0
        
        # Loss tracking
        self.recent_losses = {obj: deque(maxlen=100) for obj in OBJECTIVES}
        
        logger.info(f"Multi-Objective DQN Agent initialized on {self.device}")
        logger.info(f"  State dim: {config.state_dim}, Action dim: {config.action_dim}")
        logger.info(f"  Objectives: {OBJECTIVES}")
        logger.info(f"  Weights: {config.objective_weights}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy with weighted Q-values
        
        Same interface as regular DQNAgent.act()
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            combined_q = self.q_network.get_combined_q_values(
                state_tensor,
                self.config.objective_weights
            )
            return combined_q.argmax(dim=1).item()
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,  # Scalar reward (for compatibility)
        next_state: np.ndarray,
        done: bool,
        mo_rewards: Optional[Dict[str, float]] = None
    ):
        """
        Store experience in replay buffer
        
        Args:
            state, action, reward, next_state, done: Standard experience
            mo_rewards: Optional multi-objective rewards dict
                       If None, scalar reward is used for all objectives
        """
        if mo_rewards is None:
            # Fall back to scalar reward for all objectives
            mo_rewards = {obj: reward for obj in OBJECTIVES}
        
        self.memory.push(state, action, mo_rewards, next_state, done)
    
    def replay(self) -> Dict[str, float]:
        """
        Train on batch from replay buffer
        
        Returns dict of losses per objective
        """
        if len(self.memory) < self.config.min_memory_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        rewards = {obj: r.to(self.device) for obj, r in rewards.items()}
        
        # Get current Q-values for all objectives
        current_q_dict = self.q_network(states)
        
        # Get next Q-values from target
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select actions
                next_q_online = self.q_network.get_combined_q_values(
                    next_states, self.config.objective_weights
                )
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                
                next_q_target = self.target_network(next_states)
            else:
                next_q_target = self.target_network(next_states)
        
        # Calculate loss for each objective
        total_loss = 0.0
        losses = {}
        
        for obj in OBJECTIVES:
            # Current Q for taken actions
            current_q = current_q_dict[obj].gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q
            if self.config.use_double_dqn:
                next_q = next_q_target[obj].gather(1, next_actions).squeeze(1)
            else:
                next_q = next_q_target[obj].max(dim=1)[0]
            
            target_q = rewards[obj] + self.config.gamma * next_q * (1 - dones)
            
            # Loss
            loss = nn.MSELoss()(current_q, target_q)
            
            # Weight the loss by objective weight
            weighted_loss = loss * self.config.objective_weights[obj]
            total_loss += weighted_loss
            
            losses[obj] = loss.item()
            self.recent_losses[obj].append(loss.item())
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.config.use_soft_update:
            self._soft_update()
        elif self.update_count % self.config.target_update_every == 0:
            self._hard_update()
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return losses
    
    def _soft_update(self):
        """Soft update target network"""
        tau = self.config.tau
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def _hard_update(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get average losses per objective"""
        return {
            obj: np.mean(list(losses)) if losses else 0.0
            for obj, losses in self.recent_losses.items()
        }
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'config': self.config,
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"Agent loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: ENVIRONMENT EXTENSION
# ═══════════════════════════════════════════════════════════════════════════════

def add_multi_objective_to_env(env_class):
    """
    Decorator/mixin to add multi-objective reward calculation to environment
    
    Usage:
        from trade_based_mtf_env import TradeBasedMultiTimeframeEnv
        
        # Add multi-objective support
        env = TradeBasedMultiTimeframeEnv(...)
        add_mo_support(env, mo_config)
    """
    pass  # Not needed - we'll calculate MO rewards in the trainer


def calculate_mo_rewards_from_trade(
    trade_result,  # TradeResult dataclass
    mo_config: MORewardConfig
) -> Dict[str, float]:
    """
    Calculate multi-objective rewards from a TradeResult
    
    This is called in the trainer after each trade completes.
    
    Args:
        trade_result: TradeResult from environment
        mo_config: Multi-objective reward configuration
    
    Returns:
        Dict of rewards per objective
    """
    calculator = MultiObjectiveRewardCalculator(mo_config)
    
    # Extract trade info
    pnl_pct = trade_result.pnl_pct
    hold_duration = trade_result.hold_duration
    exit_reason = trade_result.exit_reason
    
    # For MFE/MAE, we'd need to track these in the environment
    # For now, estimate based on P&L
    if pnl_pct > 0:
        max_favorable = pnl_pct * 1.2  # Assume some giveback
        max_adverse = -pnl_pct * 0.3   # Assume some drawdown
    else:
        max_favorable = abs(pnl_pct) * 0.2  # Small profit at some point
        max_adverse = pnl_pct * 1.1   # Slightly worse than final
    
    return calculator.calculate(
        pnl_pct=pnl_pct,
        hold_duration=hold_duration,
        exit_reason=exit_reason,
        max_favorable_excursion=max_favorable,
        max_adverse_excursion=max_adverse
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: USAGE EXAMPLE / INTEGRATION GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

"""
INTEGRATION INTO trade_based_trainer.py:
========================================

1. Add imports at top:
   ```python
   from src.multi_objective_extension import (
       MultiObjectiveDQNAgent, MODQNConfig, MORewardConfig,
       calculate_mo_rewards_from_trade, OBJECTIVES
   )
   ```

2. Add config option:
   ```python
   'use_multi_objective': True,  # Enable multi-objective mode
   'mo_weights': {
       'pnl_quality': 0.35,
       'hold_duration': 0.25,
       'win_achieved': 0.15,
       'loss_control': 0.15,
       'risk_reward': 0.10,
   },
   ```

3. In _train_rl_agent(), when creating agent:
   ```python
   if self.config.get('use_multi_objective', False):
       mo_config = MODQNConfig(
           state_dim=state_dim,
           action_dim=action_dim,
           hidden_dims=self.config['rl_hidden_dims'],
           use_dueling=self.config['use_dueling_dqn'],
           use_double_dqn=self.config['use_double_dqn'],
           batch_size=self.config.get('rl_batch_size', 256),
           memory_size=self.config.get('rl_memory_size', 50000),
           epsilon_start=self.config.get('epsilon_start', 1.0),
           epsilon_end=self.config.get('epsilon_end', 0.05),
           epsilon_decay=self.config.get('epsilon_decay', 0.9998),
           objective_weights=self.config.get('mo_weights', {}),
       )
       self.rl_agent = MultiObjectiveDQNAgent(config=mo_config)
   else:
       # Regular DQNAgent (existing code)
       self.rl_agent = DQNAgent(config=rl_config)
   ```

4. In _run_trade_based_episode(), after trade completes:
   ```python
   # Store experience with multi-objective rewards
   if self.config.get('use_multi_objective', False) and env.trade_result:
       mo_rewards = calculate_mo_rewards_from_trade(
           env.trade_result,
           MORewardConfig(
               stop_loss_pct=self.config.get('stop_loss', 0.03),
               take_profit_pct=self.config.get('take_profit', 0.03),
               **self.config.get('mo_weights', {})
           )
       )
       self.rl_agent.remember(state, action, reward, next_state, done, mo_rewards=mo_rewards)
   else:
       self.rl_agent.remember(state, action, reward, next_state, done)
   ```

5. Logging - add per-objective stats:
   ```python
   if self.config.get('use_multi_objective', False):
       loss_stats = self.rl_agent.get_loss_stats()
       for obj, loss in loss_stats.items():
           logger.info(f"    {obj}: loss={loss:.4f}")
   ```
"""