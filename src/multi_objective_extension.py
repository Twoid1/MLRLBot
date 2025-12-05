"""
MULTI-OBJECTIVE EXTENSION FOR TRADE-BASED TRAINER (v4.3 - FIXED)
=========================================================================

This module extends the existing trade_based_trainer with multi-objective rewards.

v4.3 FIXES:
- pnl_quality: Removed 0.5 loser penalty - now negative P&L = negative reward (guaranteed)
- risk_reward: Added caps to prevent value explosions when max_adverse is tiny

v4.1 FEATURES (preserved):
- ALL rewards scale with trade size
- NO floors or caps that break proportionality (except risk_reward caps for stability)
- Bigger losses = proportionally bigger penalties
- Everything scales linearly with P&L magnitude

USAGE:
------
1. Copy this file to src/multi_objective_extension.py
2. In trade_based_trainer.py, add: from src.multi_objective_extension import ...
3. Enable multi-objective mode in config: 'use_multi_objective': True
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
# PART 1: MULTI-OBJECTIVE REWARD CALCULATOR (v4.3 - FIXED)
# ═══════════════════════════════════════════════════════════════════════════════

# Objective names (5 objectives)
OBJECTIVES = ['pnl_quality', 'hold_duration', 'win_achieved', 'loss_control', 'risk_reward']


@dataclass
class MORewardConfig:
    """
    Configuration for multi-objective rewards (v4.3)
    
    All parameter names preserved from original for backward compatibility.
    """
    
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
    
    # Hold duration settings (ORIGINAL NAMES PRESERVED)
    min_hold_for_bonus: int = 12      # Below this = penalty (1 hour at 5m)
    max_hold_steps: int = 300         # Timeout threshold
    target_hold_steps: int = 48       # Optimal hold (4 hours at 5m)
    
    # Reward scaling
    pnl_scale: float = 50.0           # Base P&L multiplier (1% = 0.5 base reward)
    
    # v4.3: Risk-reward caps to prevent explosions
    max_rr_ratio: float = 5.0         # Cap R:R ratio
    max_rr_magnitude: float = 2.0     # Cap magnitude factor
    
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
    Calculates 5 separate reward signals from trade outcomes (v4.3 - FIXED)
    
    Each objective teaches a different aspect of good trading:
    1. pnl_quality   - Maximize profit, minimize loss magnitude
    2. hold_duration - Hold trades for meaningful moves
    3. win_achieved  - Win more trades than you lose
    4. loss_control  - Cut losers early, before stop-loss
    5. risk_reward   - Achieve good risk/reward ratios
    
    v4.3 FIXES:
    - pnl_quality: Full penalty for losers (no 0.5 factor)
      → Ensures negative P&L = negative pnl_quality reward
    - risk_reward: Caps on rr_ratio and magnitude to prevent explosions
      → Prevents +50 rewards from tiny max_adverse values
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
        
        ALL rewards scale proportionally with trade size.
        
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
        is_winner = pnl_pct > 0
        
        # Pre-calculate magnitude factor used by multiple objectives
        magnitude = abs(pnl_pct) * self.config.pnl_scale  # e.g., 2% × 50 = 1.0
        
        # 1. PNL_QUALITY: Pure P&L signal (v4.3: full penalty for losers)
        rewards['pnl_quality'] = self._calc_pnl_quality(pnl_pct, hold_duration, is_winner)
        
        # 2. HOLD_DURATION: Scaled with both duration and magnitude
        rewards['hold_duration'] = self._calc_hold_duration(pnl_pct, hold_duration, is_winner, magnitude)
        
        # 3. WIN_ACHIEVED: Scaled by P&L ratio to targets
        rewards['win_achieved'] = self._calc_win_achieved(pnl_pct, hold_duration, is_winner)
        
        # 4. LOSS_CONTROL: Fully scaled with loss magnitude
        rewards['loss_control'] = self._calc_loss_control(pnl_pct, hold_duration, exit_reason, is_winner, magnitude)
        
        # 5. RISK_REWARD: Scaled with caps to prevent explosions (v4.3 fix)
        rewards['risk_reward'] = self._calc_risk_reward(
            pnl_pct, hold_duration, max_favorable_excursion, max_adverse_excursion, is_winner, magnitude
        )
        
        return rewards
    
    def _calc_pnl_quality(self, pnl_pct: float, hold_duration: int, is_winner: bool) -> float:
        """
        PNL Quality objective (v4.3 - FIXED)
        
        FORMULA: pnl × scale (same for winners AND losers)
        
        v4.3 FIX: Removed 0.5 loser penalty
        - Before: losers got pnl × scale × 0.5 (half penalty)
        - After: losers get pnl × scale (full penalty)
        
        This ensures:
        - Negative avg P&L → Negative avg pnl_quality (guaranteed)
        - Positive avg P&L → Positive avg pnl_quality (guaranteed)
        """
        cfg = self.config
        
        # v4.3: Same formula for winners AND losers
        # Full penalty ensures reward accurately reflects P&L direction
        return pnl_pct * cfg.pnl_scale
    
    def _calc_hold_duration(self, pnl_pct: float, hold_duration: int, is_winner: bool, magnitude: float) -> float:
        """
        Hold Duration objective (v4.1 - FULLY SCALED)
        
        FORMULA: direction × hold_quality × magnitude_factor
        
        - direction: +1 (winners), -1 (losers)
        - hold_quality: How well you held (0.1 to 1.5 for winners, 0.7 to 1.3 for losers)
        - magnitude_factor: Scales with trade size (no cap)
        """
        cfg = self.config
        
        # Direction: +1 for winners, -1 for losers
        direction = 1.0 if is_winner else -1.0
        
        # Hold quality (how well you held relative to optimal)
        hold_quality = self._get_hold_multiplier(hold_duration, is_winner)
        
        # Magnitude factor - scales linearly with trade size (no cap!)
        magnitude_factor = magnitude / 3.0  # Normalized so 3% trade = 1.0
        
        return direction * hold_quality * magnitude_factor
    
    def _calc_win_achieved(self, pnl_pct: float, hold_duration: int, is_winner: bool) -> float:
        """
        Win Achieved objective (v4.1 - PURE)
        
        FORMULA: pnl_ratio (NO hold duration influence!)
        
        This is a PURE win/loss signal:
        - Wins scale with how much you won relative to take-profit
        - Losses scale with how much you lost relative to stop-loss
        - Hold duration has NO effect here
        """
        cfg = self.config
        
        if is_winner:
            # Ratio to take-profit (e.g., 3% win / 3% TP = 1.0, 6% win = 2.0)
            return pnl_pct / cfg.take_profit_pct
        else:
            # Ratio to stop-loss (e.g., -1.5% / -3% SL = -0.5, -3% = -1.0)
            return pnl_pct / cfg.stop_loss_pct  # Already negative
    
    def _calc_loss_control(self, pnl_pct: float, hold_duration: int, exit_reason: str, 
                           is_winner: bool, magnitude: float) -> float:
        """
        Loss Control objective (v4.1 - FULLY SCALED)
        
        FORMULA: control_quality × magnitude_factor
        
        - control_quality: How well you managed the loss (-1 to +1)
        - magnitude_factor: Scales with loss size (bigger loss = bigger reward/penalty)
        
        For WINNERS: Returns 0 (neutral)
        For LOSERS: Rewards cutting early, penalizes letting losses run
        """
        cfg = self.config
        
        if is_winner:
            return 0.0
        
        loss_magnitude = abs(pnl_pct)
        
        # Calculate control quality (-1 to +1)
        if exit_reason == 'agent':
            # Agent chose to exit - how much did they save?
            saved_ratio = (cfg.stop_loss_pct - loss_magnitude) / cfg.stop_loss_pct
            control_quality = saved_ratio
            
        elif exit_reason == 'stop_loss':
            control_quality = -0.2
            
        elif exit_reason == 'timeout':
            control_quality = -0.5
            
        else:
            control_quality = 0.0
        
        # Magnitude factor - bigger losses make this more important
        magnitude_factor = loss_magnitude / cfg.stop_loss_pct
        
        return control_quality * magnitude_factor
    
    def _calc_risk_reward(
        self,
        pnl_pct: float,
        hold_duration: int,
        max_favorable: float,
        max_adverse: float,
        is_winner: bool,
        magnitude: float
    ) -> float:
        """
        Risk/Reward objective (v4.3 - FIXED WITH CAPS)
        
        FORMULA: rr_quality × magnitude_factor (both capped)
        
        v4.3 FIX: Added caps to prevent explosions
        - Before: rr_ratio could be 200+ when max_adverse was tiny (0.01%)
        - After: rr_ratio capped at 5.0, magnitude_factor capped at 2.0
        
        This prevents single trades from dominating the average.
        """
        cfg = self.config
        
        # Avoid division by zero with small defaults
        if abs(max_adverse) < 0.0001:
            max_adverse = -0.0001
        if max_favorable < 0.0001:
            max_favorable = 0.0001
        
        if is_winner:
            # Risk/Reward ratio: profit / max drawdown
            raw_rr_ratio = pnl_pct / abs(max_adverse)
            
            # v4.3 FIX: Cap to prevent explosions
            rr_ratio = min(raw_rr_ratio, cfg.max_rr_ratio)
            
            # Quality: compare to target RR of 2.0
            rr_quality = rr_ratio / 2.0
            
        else:
            # For losers: how much profit did you give back?
            if max_favorable > abs(pnl_pct) * 0.1:
                total_swing = max_favorable - pnl_pct
                raw_giveback = total_swing / max_favorable
                
                # v4.3 FIX: Cap giveback ratio
                giveback_ratio = min(raw_giveback, cfg.max_rr_ratio)
                rr_quality = -giveback_ratio
            else:
                # Never had meaningful profit - scale with loss
                rr_quality = pnl_pct / cfg.stop_loss_pct
        
        # v4.3 FIX: Cap magnitude factor
        magnitude_factor = min(magnitude / 2.0, cfg.max_rr_magnitude)
        
        return rr_quality * magnitude_factor
    
    def _get_hold_multiplier(self, hold_duration: int, is_winner: bool) -> float:
        """
        Calculate hold duration multiplier (shared by multiple objectives)
        
        Winners: Longer hold = higher multiplier (reward patience)
        Losers: Shorter hold = lower multiplier (reward cutting)
        
        Returns: 0.1 to 1.5 for winners, 0.7 to 1.3 for losers
        """
        cfg = self.config
        
        min_hold = cfg.min_hold_for_bonus
        target_hold = cfg.target_hold_steps
        extended_hold = min(cfg.max_hold_steps - 50, 200)
        
        if is_winner:
            if hold_duration < min_hold:
                t = hold_duration / min_hold
                mult = 0.1 + t * 0.4
                
            elif hold_duration < target_hold:
                t = (hold_duration - min_hold) / (target_hold - min_hold)
                mult = 0.5 + t * 0.5
                
            elif hold_duration < extended_hold:
                t = (hold_duration - target_hold) / (extended_hold - target_hold)
                mult = 1.0 + t * 0.5
                
            else:
                mult = 1.5
        else:
            if hold_duration < min_hold:
                mult = 0.7
                
            elif hold_duration < target_hold:
                t = (hold_duration - min_hold) / (target_hold - min_hold)
                mult = 0.7 + t * 0.3
                
            elif hold_duration < extended_hold:
                t = (hold_duration - target_hold) / (extended_hold - target_hold)
                mult = 1.0 + t * 0.3
                
            else:
                mult = 1.3
        
        return mult
    
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
        """
        if mo_rewards is None:
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
            current_q = current_q_dict[obj].gather(1, actions.unsqueeze(1)).squeeze(1)
            
            if self.config.use_double_dqn:
                next_q = next_q_target[obj].gather(1, next_actions).squeeze(1)
            else:
                next_q = next_q_target[obj].max(dim=1)[0]
            
            target_q = rewards[obj] + self.config.gamma * next_q * (1 - dones)
            
            loss = nn.MSELoss()(current_q, target_q)
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

def calculate_mo_rewards_from_trade(
    trade_result,  # TradeResult dataclass
    mo_config: MORewardConfig
) -> Dict[str, float]:
    """
    Calculate multi-objective rewards from a TradeResult
    """
    calculator = MultiObjectiveRewardCalculator(mo_config)
    
    pnl_pct = trade_result.pnl_pct
    hold_duration = trade_result.hold_duration
    exit_reason = trade_result.exit_reason
    
    # Estimate MFE/MAE if not tracked
    if pnl_pct > 0:
        max_favorable = pnl_pct * 1.2
        max_adverse = -pnl_pct * 0.3
    else:
        max_favorable = abs(pnl_pct) * 0.2
        max_adverse = pnl_pct * 1.1
    
    return calculator.calculate(
        pnl_pct=pnl_pct,
        hold_duration=hold_duration,
        exit_reason=exit_reason,
        max_favorable_excursion=max_favorable,
        max_adverse_excursion=max_adverse
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_v43_fixes():
    """Test v4.3 fixes for pnl_quality and risk_reward"""
    
    print("=" * 100)
    print("MULTI-OBJECTIVE REWARDS v4.3 - FIXED")
    print("=" * 100)
    print()
    print("v4.3 FIXES:")
    print("  1. pnl_quality: Full penalty for losers (removed 0.5 factor)")
    print("     → Negative P&L = Negative reward (guaranteed)")
    print("  2. risk_reward: Caps on rr_ratio (5.0) and magnitude (2.0)")
    print("     → Prevents explosion from tiny max_adverse values")
    print()
    
    config = MORewardConfig()
    calculator = MultiObjectiveRewardCalculator(config)
    
    # Test 1: pnl_quality fix - verify negative P&L = negative reward
    print("-" * 100)
    print("TEST 1: pnl_quality fix - negative P&L must give negative reward")
    print("-" * 100)
    
    test_pnl_cases = [
        (0.03, "Winner +3%"),
        (0.015, "Winner +1.5%"),
        (-0.015, "Loser -1.5%"),
        (-0.03, "Loser -3%"),
    ]
    
    for pnl, desc in test_pnl_cases:
        reward = calculator._calc_pnl_quality(pnl, 30, pnl > 0)
        sign = "Y" if (pnl > 0 and reward > 0) or (pnl < 0 and reward < 0) else "N"
        print(f"  {desc:20s}: pnl_quality = {reward:+.3f} {sign}")
    
    # Simulate mixed portfolio
    print()
    print("  Simulation: 46% win rate, -0.28% avg P&L")
    np.random.seed(42)
    n_trades = 1000
    pnl_list = []
    reward_list = []
    
    for i in range(n_trades):
        if i < 460:
            pnl = np.random.uniform(0.005, 0.025)
        else:
            pnl = -np.random.uniform(0.01, 0.026)
        pnl_list.append(pnl)
        reward_list.append(calculator._calc_pnl_quality(pnl, 30, pnl > 0))
    
    avg_pnl = np.mean(pnl_list)
    avg_reward = np.mean(reward_list)
    sign_match = (avg_pnl > 0 and avg_reward > 0) or (avg_pnl < 0 and avg_reward < 0)
    
    print(f"    Avg P&L: {avg_pnl*100:+.3f}%")
    print(f"    Avg pnl_quality: {avg_reward:+.3f}")
    print(f"    Sign match: {' CORRECT' if sign_match else ' WRONG'}")
    
    # Test 2: risk_reward fix - verify no explosions
    print()
    print("-" * 100)
    print("TEST 2: risk_reward fix - capped to prevent explosions")
    print("-" * 100)
    
    test_rr_cases = [
        # (pnl, mfe, mae, description)
        (0.02, 0.025, -0.0001, "Winner +2%, tiny drawdown (-0.01%)"),
        (0.02, 0.025, -0.005, "Winner +2%, normal drawdown (-0.5%)"),
        (0.02, 0.025, -0.01, "Winner +2%, larger drawdown (-1%)"),
        (-0.02, 0.01, -0.025, "Loser -2%, had +1% potential"),
        (-0.02, 0.0, -0.025, "Loser -2%, no potential"),
    ]
    
    for pnl, mfe, mae, desc in test_rr_cases:
        is_winner = pnl > 0
        magnitude = abs(pnl) * config.pnl_scale
        reward = calculator._calc_risk_reward(pnl, 30, mfe, mae, is_winner, magnitude)
        capped = "CAPPED" if abs(reward) <= 5.0 else "EXPLODED!"
        print(f"  {desc}")
        print(f"    risk_reward = {reward:+.3f} ({capped})")
    
    # Test 3: Full scenario simulation
    print()
    print("-" * 100)
    print("TEST 3: Full scenario - 46% win rate, -0.28% avg P&L")
    print("-" * 100)
    
    np.random.seed(123)
    n_trades = 100
    n_winners = 46
    
    all_pnl = []
    all_rewards = {obj: [] for obj in OBJECTIVES}
    
    for i in range(n_trades):
        if i < n_winners:
            pnl = np.random.uniform(0.005, 0.025)
            mfe = pnl * np.random.uniform(1.0, 1.2)
            mae = -np.random.uniform(0.001, 0.008)
            exit_reason = 'agent'
        else:
            pnl = -np.random.uniform(0.01, 0.026)
            mfe = np.random.uniform(0, 0.005)
            mae = pnl * 1.1
            exit_reason = np.random.choice(['agent', 'stop_loss'], p=[0.7, 0.3])
        
        all_pnl.append(pnl)
        
        rewards = calculator.calculate(
            pnl_pct=pnl,
            hold_duration=30,
            exit_reason=exit_reason,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae
        )
        
        for obj in OBJECTIVES:
            all_rewards[obj].append(rewards[obj])
    
    # Use default weights
    weights = config.get_weights()
    
    print(f"  Avg P&L: {np.mean(all_pnl)*100:.2f}%")
    print(f"  Win Rate: {n_winners}%")
    print()
    
    print("  PER-OBJECTIVE AVERAGES:")
    total = 0
    for obj in OBJECTIVES:
        avg = np.mean(all_rewards[obj])
        w = weights[obj]
        contrib = avg * w
        total += contrib
        print(f"    {obj:15s}: {avg:+.3f} × {w:.2f} = {contrib:+.4f}")
    
    print(f"    {'─' * 45}")
    print(f"    {'TOTAL':15s}: {total:+.4f}")
    print()
    
    if total > 0 and np.mean(all_pnl) < 0:
        print("   WARNING: Positive reward on negative P&L!")
    elif total < 0 and np.mean(all_pnl) < 0:
        print("   CORRECT: Negative reward on negative P&L!")
    
    print()
    print("=" * 100)


def test_scaled_mo_rewards():
    """Test the scaled multi-objective reward system"""
    
    print("=" * 100)
    print("MULTI-OBJECTIVE REWARDS v4.3 - FULLY SCALED")
    print("=" * 100)
    print()
    
    config = MORewardConfig()
    calculator = MultiObjectiveRewardCalculator(config)
    
    test_cases = [
        # Small wins
        (0.005, 3, 'agent', 0.006, -0.001, "Tiny win +0.5%, quick exit"),
        (0.005, 48, 'agent', 0.006, -0.001, "Tiny win +0.5%, target hold"),
        
        # Medium wins
        (0.02, 3, 'agent', 0.024, -0.006, "Medium win +2%, quick exit"),
        (0.02, 48, 'agent', 0.024, -0.006, "Medium win +2%, target hold"),
        (0.02, 100, 'take_profit', 0.02, -0.004, "Medium win +2%, extended"),
        
        # Big wins
        (0.04, 48, 'take_profit', 0.04, -0.008, "Big win +4%, target hold"),
        (0.06, 48, 'take_profit', 0.06, -0.01, "Huge win +6%, target hold"),
        
        # Small losses
        (-0.005, 3, 'agent', 0.001, -0.006, "Tiny loss -0.5%, quick cut"),
        (-0.005, 48, 'stop_loss', 0.001, -0.005, "Tiny loss -0.5%, stop-loss"),
        
        # Medium losses  
        (-0.015, 3, 'agent', 0.003, -0.016, "Medium loss -1.5%, quick cut"),
        (-0.015, 48, 'stop_loss', 0.003, -0.015, "Medium loss -1.5%, stop-loss"),
        
        # Full losses
        (-0.03, 3, 'agent', 0.006, -0.033, "Full loss -3%, quick cut"),
        (-0.03, 48, 'stop_loss', 0.006, -0.03, "Full loss -3%, stop-loss"),
        (-0.03, 150, 'timeout', 0.006, -0.033, "Full loss -3%, timeout"),
    ]
    
    print(f"{'Description':<32} | {'pnl_q':>8} | {'hold_d':>8} | {'win_a':>8} | {'loss_c':>8} | {'rr':>8} | {'TOTAL':>8}")
    print("-" * 110)
    
    for pnl_pct, hold, exit_reason, mfe, mae, desc in test_cases:
        rewards = calculator.calculate(pnl_pct, hold, exit_reason, mfe, mae)
        total = calculator.get_weighted_reward(rewards)
        
        print(f"{desc:<32} | {rewards['pnl_quality']:>+8.3f} | {rewards['hold_duration']:>+8.3f} | "
              f"{rewards['win_achieved']:>+8.3f} | {rewards['loss_control']:>+8.3f} | "
              f"{rewards['risk_reward']:>+8.3f} | {total:>+8.3f}")
    
    print()
    print("=" * 100)


if __name__ == "__main__":
    test_v43_fixes()
    print("\n\n")
    test_scaled_mo_rewards()