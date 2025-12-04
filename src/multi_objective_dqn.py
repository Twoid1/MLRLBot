"""
Multi-Objective Deep Q-Network (MO-DQN)

A neural network architecture with:
- Shared feature extraction layers
- Separate output heads for each reward objective

This allows the agent to learn different Q-functions for each objective,
then combine them for action selection.

Author: Claude
Version: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# Objective names (must match MultiObjectiveRewardCalculator)
OBJECTIVES = [
    'pnl_quality',
    'hold_duration', 
    'win_achieved',
    'loss_control',
    'risk_reward',
]

NUM_OBJECTIVES = len(OBJECTIVES)


@dataclass
class MODQNConfig:
    """Configuration for Multi-Objective DQN"""
    
    # Network architecture
    state_dim: int = 64                    # Input dimension
    action_dim: int = 2                    # Number of actions
    hidden_dims: List[int] = None         # Shared layer sizes
    head_hidden_dim: int = 32              # Hidden dim in each objective head
    
    # Objective weights (for combining Q-values)
    objective_weights: Dict[str, float] = None
    
    # Training
    learning_rate: float = 0.001
    gamma: float = 0.99                    # Discount factor
    batch_size: int = 256                  # Batch size for training
    
    # Replay buffer
    memory_size: int = 50000               # Replay buffer capacity
    
    # Target network
    tau: float = 0.005                     # Soft update coefficient
    target_update_frequency: int = 10      # Hard update frequency (if not using soft)
    use_soft_update: bool = True           # Use soft updates vs hard updates
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.9998          # Aligned with trade_based_trainer
    
    # Advanced
    use_dueling: bool = True               # Dueling architecture per head
    use_double_dqn: bool = True            # Double DQN for stability
    use_noisy: bool = False                # Noisy networks
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        
        if self.objective_weights is None:
            self.objective_weights = {
                'pnl_quality': 0.35,
                'hold_duration': 0.10,
                'win_achieved': 0.20,
                'loss_control': 0.15,
                'risk_reward': 0.20,
            }


class ObjectiveHead(nn.Module):
    """
    Single objective head - outputs Q-values for one objective
    
    Can optionally use dueling architecture where:
    - Value stream estimates state value V(s)
    - Advantage stream estimates action advantages A(s,a)
    - Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        action_dim: int,
        hidden_dim: int = 32,
        use_dueling: bool = True
    ):
        super().__init__()
        
        self.use_dueling = use_dueling
        self.action_dim = action_dim
        
        if use_dueling:
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
        else:
            # Simple Q-value output
            self.q_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for this objective
        
        Args:
            features: Shared features [batch, feature_dim]
            
        Returns:
            Q-values [batch, action_dim]
        """
        if self.use_dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine: Q = V + (A - mean(A))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_layer(features)
        
        return q_values


class MultiObjectiveDQN(nn.Module):
    """
    Multi-Objective Deep Q-Network
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         INPUT (state)                           │
    │                              │                                  │
    │                    ┌─────────┴─────────┐                        │
    │                    │  SHARED LAYERS    │                        │
    │                    │  (feature extraction)                      │
    │                    └─────────┬─────────┘                        │
    │                              │                                  │
    │      ┌───────────┬───────────┼───────────┬───────────┐          │
    │      ▼           ▼           ▼           ▼           ▼          │
    │  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐     │
    │  │ HEAD  │   │ HEAD  │   │ HEAD  │   │ HEAD  │   │ HEAD  │     │
    │  │ pnl   │   │ hold  │   │ win   │   │ loss  │   │ risk  │     │
    │  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘     │
    │      │           │           │           │           │          │
    │      ▼           ▼           ▼           ▼           ▼          │
    │   Q_pnl       Q_hold      Q_win      Q_loss      Q_risk        │
    │      │           │           │           │           │          │
    │      └───────────┴───────────┼───────────┴───────────┘          │
    │                              ▼                                  │
    │                    WEIGHTED COMBINATION                         │
    │                         Q_combined                              │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: MODQNConfig):
        super().__init__()
        
        self.config = config
        self.objectives = OBJECTIVES
        self.num_objectives = NUM_OBJECTIVES
        
        # ═══════════════════════════════════════════════════════════════
        # SHARED FEATURE EXTRACTION LAYERS
        # ═══════════════════════════════════════════════════════════════
        layers = []
        input_dim = config.state_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        self.feature_dim = config.hidden_dims[-1]
        
        # ═══════════════════════════════════════════════════════════════
        # OBJECTIVE-SPECIFIC HEADS
        # ═══════════════════════════════════════════════════════════════
        self.heads = nn.ModuleDict({
            obj: ObjectiveHead(
                input_dim=self.feature_dim,
                action_dim=config.action_dim,
                hidden_dim=config.head_hidden_dim,
                use_dueling=config.use_dueling
            )
            for obj in self.objectives
        })
        
        # ═══════════════════════════════════════════════════════════════
        # OBJECTIVE WEIGHTS (learnable or fixed)
        # ═══════════════════════════════════════════════════════════════
        weights = [config.objective_weights.get(obj, 0.2) for obj in self.objectives]
        self.register_buffer('objective_weights', torch.tensor(weights, dtype=torch.float32))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        state: torch.Tensor,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass
        
        Args:
            state: Input state [batch, state_dim]
            return_all: If True, return Q-values for all objectives
            
        Returns:
            combined_q: Weighted combination of Q-values [batch, action_dim]
            objective_qs: Dict of Q-values per objective (if return_all=True)
        """
        # Extract shared features
        features = self.shared_layers(state)
        
        # Compute Q-values for each objective
        objective_qs = {
            obj: head(features)
            for obj, head in self.heads.items()
        }
        
        # Combine with weights
        # Stack: [batch, num_objectives, action_dim]
        q_stack = torch.stack([objective_qs[obj] for obj in self.objectives], dim=1)
        
        # Weights: [num_objectives, 1] for broadcasting
        weights = self.objective_weights.unsqueeze(1)
        
        # Weighted sum: [batch, action_dim]
        combined_q = (q_stack * weights).sum(dim=1)
        
        if return_all:
            return combined_q, objective_qs
        else:
            return combined_q, None
    
    def get_q_values(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get Q-values for all objectives separately"""
        features = self.shared_layers(state)
        
        return {
            obj: head(features)
            for obj, head in self.heads.items()
        }
    
    def get_combined_q(self, state: torch.Tensor) -> torch.Tensor:
        """Get combined Q-values for action selection"""
        combined_q, _ = self.forward(state, return_all=False)
        return combined_q


class MultiObjectiveReplayBuffer:
    """
    Replay buffer that stores multi-objective rewards
    
    Instead of storing a single reward scalar, stores a dict of rewards
    for each objective.
    """
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = 0
        self.size = 0
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Rewards for each objective
        self.rewards = {
            obj: np.zeros(capacity, dtype=np.float32)
            for obj in OBJECTIVES
        }
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        rewards: Dict[str, float],
        next_state: np.ndarray,
        done: bool
    ):
        """Store a transition with multi-objective rewards"""
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
        """Sample a batch of transitions"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
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


class MultiObjectiveAgent:
    """
    RL Agent that uses Multi-Objective DQN
    
    Key features:
    - Separate Q-networks for each objective
    - Trains each objective with its own reward signal
    - Combines objectives for action selection
    - Supports Double DQN for each objective
    
    All parameters are configurable via MODQNConfig.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        config: Optional[MODQNConfig] = None,
        device: str = 'auto'
    ):
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create config
        if config is None:
            config = MODQNConfig(state_dim=state_dim, action_dim=action_dim)
        else:
            config.state_dim = state_dim
            config.action_dim = action_dim
        
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.objectives = OBJECTIVES
        
        # ═══════════════════════════════════════════════════════════════
        # NETWORKS
        # ═══════════════════════════════════════════════════════════════
        self.q_network = MultiObjectiveDQN(config).to(self.device)
        self.target_network = MultiObjectiveDQN(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # ═══════════════════════════════════════════════════════════════
        # OPTIMIZER
        # ═══════════════════════════════════════════════════════════════
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # ═══════════════════════════════════════════════════════════════
        # REPLAY BUFFER (uses config.memory_size)
        # ═══════════════════════════════════════════════════════════════
        self.memory = MultiObjectiveReplayBuffer(
            capacity=config.memory_size,
            state_dim=state_dim
        )
        
        # ═══════════════════════════════════════════════════════════════
        # EXPLORATION (uses config values)
        # ═══════════════════════════════════════════════════════════════
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # ═══════════════════════════════════════════════════════════════
        # TRAINING STATS
        # ═══════════════════════════════════════════════════════════════
        self.training_step = 0
        self.objective_losses = {obj: [] for obj in self.objectives}
        self.objective_q_values = {obj: [] for obj in self.objectives}
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy with combined Q-values
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network.get_combined_q(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        rewards: Dict[str, float],
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition with multi-objective rewards"""
        self.memory.push(state, action, rewards, next_state, done)
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Train on a batch - trains EACH objective separately
        
        Returns:
            Dictionary of losses per objective
        """
        if len(self.memory) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Move rewards to device
        rewards = {obj: r.to(self.device) for obj, r in rewards.items()}
        
        # Get current Q-values for all objectives
        current_qs = self.q_network.get_q_values(states)
        
        # Get next Q-values from target network
        with torch.no_grad():
            # Double DQN: use online network to select action, target to evaluate
            next_q_online = self.q_network.get_combined_q(next_states)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_qs_target = self.target_network.get_q_values(next_states)
        
        # Calculate loss for each objective
        total_loss = 0.0
        losses = {}
        
        for obj in self.objectives:
            # Current Q-value for taken action
            current_q = current_qs[obj].gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-value (Double DQN style)
            next_q = next_qs_target[obj].gather(1, next_actions).squeeze(1)
            target_q = rewards[obj] + self.config.gamma * next_q * (1 - dones)
            
            # Loss for this objective
            loss = F.smooth_l1_loss(current_q, target_q)
            losses[obj] = loss.item()
            
            # Weight loss by objective importance
            weight = self.config.objective_weights.get(obj, 0.2)
            total_loss += weight * loss
            
            # Track stats
            self.objective_losses[obj].append(loss.item())
            self.objective_q_values[obj].append(current_q.mean().item())
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        # Update exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1
        
        # Soft update target network
        if self.training_step % 10 == 0:
            self._soft_update_target()
        
        return losses
    
    def _soft_update_target(self):
        """Soft update target network"""
        tau = self.config.tau
        for target_param, param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def hard_update_target(self):
        """Hard update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_objective_stats(self) -> Dict[str, Dict[str, float]]:
        """Get training statistics per objective"""
        stats = {}
        
        for obj in self.objectives:
            if len(self.objective_losses[obj]) > 0:
                recent_losses = self.objective_losses[obj][-100:]
                recent_qs = self.objective_q_values[obj][-100:]
                
                stats[obj] = {
                    'mean_loss': np.mean(recent_losses),
                    'mean_q': np.mean(recent_qs),
                    'std_q': np.std(recent_qs),
                }
            else:
                stats[obj] = {'mean_loss': 0.0, 'mean_q': 0.0, 'std_q': 0.0}
        
        return stats
    
    def print_objective_stats(self):
        """Pretty print objective statistics"""
        stats = self.get_objective_stats()
        
        print("\n" + "=" * 60)
        print("MULTI-OBJECTIVE TRAINING STATISTICS")
        print("=" * 60)
        print(f"{'Objective':<15} {'Loss':>10} {'Mean Q':>10} {'Std Q':>10}")
        print("-" * 60)
        
        for obj in self.objectives:
            s = stats[obj]
            print(f"{obj:<15} {s['mean_loss']:>10.4f} {s['mean_q']:>10.3f} {s['std_q']:>10.3f}")
        
        print("=" * 60)
        print(f"Epsilon: {self.epsilon:.4f}  |  Training Steps: {self.training_step}")
        print("=" * 60 + "\n")
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create agent
    state_dim = 64
    action_dim = 2
    
    config = MODQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[128, 64],
        objective_weights={
            'pnl_quality': 0.35,
            'hold_duration': 0.25,
            'win_achieved': 0.15,
            'loss_control': 0.15,
            'risk_reward': 0.10,
        }
    )
    
    agent = MultiObjectiveAgent(state_dim, action_dim, config)
    
    print("Multi-Objective DQN Agent Created!")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Objectives: {agent.objectives}")
    print(f"  Device: {agent.device}")
    
    # Simulate some training
    print("\nSimulating 100 training steps...")
    
    for i in range(100):
        # Fake transition
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(state_dim).astype(np.float32)
        
        # Multi-objective rewards
        rewards = {
            'pnl_quality': np.random.randn() * 0.5,
            'hold_duration': np.random.randn() * 0.3,
            'win_achieved': np.random.choice([1.0, -0.5]),
            'loss_control': np.random.randn() * 0.2,
            'risk_reward': np.random.randn() * 0.2,
        }
        
        agent.store_transition(state, action, rewards, next_state, done=False)
        
        if len(agent.memory) >= 32:
            losses = agent.train_step(batch_size=32)
    
    # Print stats
    agent.print_objective_stats()