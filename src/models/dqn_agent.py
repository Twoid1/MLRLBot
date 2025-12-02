"""
DQN Agent Module
Deep Q-Network implementation for trading decisions
Built from scratch with PyTorch - no external RL libraries
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque, namedtuple
import random
import pickle
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class DQNConfig:
    """Configuration for DQN Agent"""
    # Network architecture
    state_dim: int = 63
    action_dim: int = 3  # Hold, Buy, Sell
    hidden_dims: List[int] = None  # Default: [256, 256, 128]
    
    # Training hyperparameters
    learning_rate: float = 0.0001
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.001  # Soft update parameter
    
    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    
    # Memory parameters
    memory_size: int = 100000
    min_memory_size: int = 1000
    
    # Training parameters
    update_every: int = 4
    target_update_every: int = 1000
    
    # Advanced features
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_prioritized_replay: bool = False
    use_noisy_networks: bool = False
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture with optional dueling architecture
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int],
                 dueling: bool = False,
                 noisy: bool = False):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            dueling: Use dueling architecture
            noisy: Use noisy networks for exploration
        """
        super(DQNetwork, self).__init__()
        
        self.dueling = dueling
        self.noisy = noisy
        
        # Build shared layers with a compatible name
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            if noisy:
                layers.append(NoisyLinear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Use 'layers' instead of 'shared_layers' for compatibility
        self.layers = nn.Sequential(*layers)
        
        # Dueling architecture
        if dueling:
            # Value stream
            if noisy:
                self.value_head = nn.Sequential(
                    NoisyLinear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    NoisyLinear(hidden_dims[-1], 1)
                )
            else:
                self.value_head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], 1)
                )
            
            # Advantage stream
            if noisy:
                self.advantage_head = nn.Sequential(
                    NoisyLinear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    NoisyLinear(hidden_dims[-1], action_dim)
                )
            else:
                self.advantage_head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], action_dim)
                )
        else:
            # Standard Q-network
            if noisy:
                self.q_head = nn.Sequential(
                    NoisyLinear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    NoisyLinear(hidden_dims[-1], action_dim)
                )
            else:
                self.q_head = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Linear(hidden_dims[-1], action_dim)
                )
    
    def forward(self, x):
        """Forward pass through network"""
        x = self.layers(x)  # Changed from self.shared_layers
        
        if self.dueling:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            # Combine value and advantage streams
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(x)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers"""
        if self.noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for exploration (Noisy Networks)
    """
    
    def __init__(self, in_features: int, out_features: int, sigma: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int):
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer
    """
    
    def __init__(self, capacity: int, state_dim: int = None, alpha: float = 0.6, seed: int = 42):
        """
        Initialize prioritized replay buffer
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension (for compatibility)
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            seed: Random seed
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.epsilon = 1e-6
        random.seed(seed)
        
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience to buffer with proper handling"""
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        experience = Experience(state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample batch with priorities - returns tensors for compatibility
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples. Have {self.size}, need {batch_size}")
        
        # Calculate probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance weights
        total = self.size
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Return as tensors for compatibility with test script
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return self.size


class ReplayBuffer:
    """
    Standard Experience Replay buffer
    """
    
    def __init__(self, capacity: int, state_dim: int = None, seed: int = 42):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
            state_dim: State dimension (for compatibility)
            seed: Random seed
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.buffer = deque(maxlen=capacity)
        random.seed(seed)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample random batch from buffer - returns tensors"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples. Have {len(self.buffer)}, need {batch_size}")
        
        experiences = random.sample(self.buffer, batch_size)
        
        # Return as tensors for compatibility
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for trading
    """
    
    def __init__(self, 
                 state_dim: int = None,
                 action_dim: int = None,
                 config: Optional[DQNConfig] = None,
                 **kwargs):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: State dimension (overrides config)
            action_dim: Action dimension (overrides config) 
            config: Configuration object
            **kwargs: Additional config parameters
        """
        # Create config if not provided
        if config is None:
            config = DQNConfig()
        
        # Override config with direct parameters if provided
        if state_dim is not None:
            config.state_dim = state_dim
        if action_dim is not None:
            config.action_dim = action_dim
        
        # Map common kwargs to config attributes
        param_mapping = {
            'learning_rate': 'learning_rate',
            'gamma': 'gamma',
            'epsilon': 'epsilon_start',
            'epsilon_decay': 'epsilon_decay',
            'epsilon_min': 'epsilon_end',
            'buffer_size': 'memory_size',
            'batch_size': 'batch_size',
            'update_frequency': 'update_every',
            'target_update_frequency': 'target_update_every',
            'hidden_dims': 'hidden_dims',
            'track_metrics': 'track_metrics'
        }
        
        # Apply kwargs to config
        for key, value in kwargs.items():
            if key in param_mapping:
                setattr(config, param_mapping[key], value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        # Add track_metrics if not present
        if not hasattr(config, 'track_metrics'):
            config.track_metrics = kwargs.get('track_metrics', False)
        
        self.config = config
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            dueling=self.config.use_dueling_dqn,
            noisy=self.config.use_noisy_networks
        ).to(self.device)
        
        self.target_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            dueling=self.config.use_dueling_dqn,
            noisy=self.config.use_noisy_networks
        ).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        
        # Memory - updated initialization
        if self.config.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                self.config.memory_size,
                self.config.state_dim,
                alpha=getattr(self.config, 'alpha', 0.6)
            )
        else:
            self.memory = ReplayBuffer(
                self.config.memory_size,
                self.config.state_dim
            )
        
        # Training variables
        self.epsilon = self.config.epsilon_start
        self.step_count = 0
        self.episode_count = 0
        self.update_count = 0
        self.training_history = []
        
        # Performance tracking
        self.losses = []
        self.rewards = []
        self.q_values = []
        
        # Compatibility attributes for test script
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        self.epsilon_decay = self.config.epsilon_decay
        self.epsilon_min = self.config.epsilon_end
        self.buffer_size = self.config.memory_size
        self.track_metrics = self.config.track_metrics if hasattr(self.config, 'track_metrics') else False

        if hasattr(self, 'device'):
            self.q_network = self.q_network.to(self.device)
            self.target_network = self.target_network.to(self.device)

    @property
    def layers(self):
        """Compatibility property for test script"""
        # Return the first layer of the shared layers for compatibility
        if hasattr(self.q_network, 'shared_layers') and len(self.q_network.shared_layers) > 0:
            return self.q_network.shared_layers
        return None
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration (not used with noisy networks)
        if training and not self.config.use_noisy_networks:
            if random.random() < self.epsilon:
                return random.randrange(self.config.action_dim)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q-values
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            if hasattr(self, 'device'):
                state_tensor = state_tensor.to(self.device)
            self.q_values.append(q_values.cpu().numpy())
        
        # Select action with highest Q-value
        return np.argmax(q_values.cpu().numpy())
    
    def remember(self, 
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
        """
        if self.config.use_prioritized_replay:
            # For prioritized replay, calculate initial priority based on reward
            priority = abs(reward) + 1e-6
            self.memory.push(state, action, reward, next_state, done, priority)
        else:
            self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size: Optional[int] = None, beta: float = 0.4) -> Optional[float]:
        """
        Train on batch from replay buffer - FIXED VERSION
        
        Args:
            batch_size: Batch size (optional, uses config if not provided)
            beta: Importance sampling parameter for prioritized replay
            
        Returns:
            Loss value or None if not enough samples
        """
        # Use provided batch_size or fall back to config
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.memory) < max(batch_size, self.config.min_memory_size):
            return None
        
        # Sample batch
        if self.config.use_prioritized_replay:
            # Prioritized replay returns: (experiences, indices, weights)
            tensors, indices, weights = self.memory.sample(batch_size, beta)
            weights = torch.FloatTensor(weights).to(self.device)
            
            # Unpack the tuple of tensors directly
            states, actions, rewards, next_states, dones = tensors
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
        else:
            # Standard replay buffer returns: (states, actions, rewards, next_states, dones) - already tensors!
            result = self.memory.sample(batch_size)
            
            # Check if result is a tuple of tensors (standard buffer) or list of experiences
            if isinstance(result, tuple) and len(result) == 5:
                # Already tensors from ReplayBuffer.sample()
                states, actions, rewards, next_states, dones = result
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                next_states = next_states.to(self.device)
                dones = dones.to(self.device)
            else:
                # List of Experience namedtuples (shouldn't happen with current code, but handle it)
                states = torch.FloatTensor([e.state for e in result]).to(self.device)
                actions = torch.LongTensor([e.action for e in result]).to(self.device)
                rewards = torch.FloatTensor([e.reward for e in result]).to(self.device)
                next_states = torch.FloatTensor([e.next_state for e in result]).to(self.device)
                dones = torch.FloatTensor([e.done for e in result]).to(self.device)
            
            weights = torch.ones(batch_size).to(self.device)
            indices = None
        
        # Current Q values
        self.q_network.train()
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states).max(1)[0]
            
            # Calculate target values
            target_q_values = rewards + (self.config.gamma * next_q_values * (1 - dones))
        
        # Calculate loss with importance weighting
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Update priorities for prioritized replay
        if self.config.use_prioritized_replay and indices is not None:
            priorities = loss.detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Reset noise if using noisy networks
        if self.config.use_noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        # Update counter
        self.update_count += 1
        
        return weighted_loss.item()
    
    def update_target_network(self, soft: bool = False):
        """
        Update target network weights
        
        Args:
            soft: Use soft update (Polyak averaging)
        """
        if soft:
            # Soft update: Î¸' = Ï„*Î¸ + (1-Ï„)*Î¸'
            for target_param, local_param in zip(
                self.target_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * local_param.data + 
                    (1.0 - self.config.tau) * target_param.data
                )
        else:
            # Hard update: copy all weights
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_step(self,
                state: np.ndarray,
                action: int,
                reward: float,
                next_state: np.ndarray,
                done: bool) -> float:
        """
        Single training step
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            
        Returns:
            Loss value
        """
        # Store experience
        self.remember(state, action, reward, next_state, done)
        
        # Update step count
        self.step_count += 1
        
        # Train if enough samples
        loss = 0.0
        if self.step_count % self.config.update_every == 0:
            loss_value = self.replay()
            loss = loss_value if loss_value is not None else 0.0
        
        # Update target network
        if self.step_count % self.config.target_update_every == 0:
            self.update_target_network(soft=True)
        
        # Decay epsilon
        if not self.config.use_noisy_networks:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
        
        return loss
    
    def train_episode(self, env, max_steps: int = 500, random_start: bool = True) -> Dict[str, Any]:
        """
        Train agent for one episode
        
        Args:
            env: Trading environment
            max_steps: Maximum steps per episode
            random_start: If True, start episodes at random positions (prevents overfitting!)
            
        Returns:
            Episode statistics
        
        UPDATED: Now uses random episode starts by default for better generalization!
        """
        state = env.reset(random_start=random_start, max_steps=max_steps)
        total_reward = 0
        total_loss = 0
        steps = 0
        
        for step in range(max_steps):
            # Select action
            action = self.act(state, training=True)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # Train
            loss = self.train_step(state, action, reward, next_state, done)
            
            # Track metrics
            total_reward += reward
            total_loss += loss
            self.rewards.append(reward)
            
            # Update state
            state = next_state
            steps += 1
            
            if done:
                break
        
        # Update episode count
        self.episode_count += 1
        
        # ========== FIX: ADD THESE LINES ==========
        # Extract trades from environment
        trades_list = []
        if hasattr(env, 'trades'):
            for t in env.trades:
                # Skip if not a dict or missing required fields
                if not isinstance(t, dict):
                    continue
                    
                trades_list.append({
                    'step': t.get('step', 0),              # ✅ Safe access with default
                    'timestamp': str(t.get('timestamp', '')),
                    'action': t.get('action', 'UNKNOWN'),
                    'price': float(t.get('price', 0)),
                    'size': float(t.get('size', 0)),
                    'fees': float(t.get('fees', 0)),
                    'pnl': float(t['pnl']) if t.get('pnl') is not None else None,
                    'duration': t.get('duration'),
                    'balance': float(t.get('balance', 0)),
                    'equity': float(t.get('equity', 0))
                })
        # ==========================================
        
        # Calculate statistics
        stats = {
            'episode': self.episode_count,
            'total_reward': total_reward,
            'average_reward': total_reward / steps,
            'total_loss': total_loss,
            'average_loss': total_loss / steps if steps > 0 else 0,
            'steps': steps,
            'epsilon': self.epsilon,
            'portfolio_value': info.get('portfolio_value', 0),
            'win_rate': info.get('win_rate', 0),
            'sharpe_ratio': info.get('sharpe_ratio', 0),
            'num_trades': sum(1 for t in trades_list if t.get('pnl') is not None),
            'winning_trades': sum(1 for t in trades_list if t.get('pnl') is not None and t['pnl'] > 0),  # ✅ CORRECT
            'losing_trades': sum(1 for t in trades_list if t.get('pnl') is not None and t['pnl'] < 0),
            'max_drawdown': info.get('max_drawdown', 0),
            
            # ========== FIX: ADD TRADES LIST ==========
            'trades': trades_list  # â† ADD THIS LINE
            # ==========================================
        }
        
        self.training_history.append(stats)
        
        return stats

    def evaluate(self, env, n_episodes: int = 10, random_start: bool = True, max_steps: int = 900) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            env: Trading environment
            n_episodes: Number of evaluation episodes
            random_start: If True, use random episode starts (recommended for diverse evaluation)
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation metrics
        
        UPDATED: Now supports random starts for more robust evaluation!
        """
        total_rewards = []
        portfolio_values = []
        win_rates = []
        sharpe_ratios = []
        
        for episode in range(n_episodes):
            state = env.reset(random_start=random_start, max_steps=max_steps)
            episode_reward = 0
            done = False
            
            while not done:
                # Select action (no exploration)
                action = self.act(state, training=False)
                
                # Take action
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                state = next_state
            
            # Store episode metrics
            total_rewards.append(episode_reward)
            portfolio_values.append(info.get('portfolio_value', 0))
            win_rates.append(info.get('win_rate', 0))
            sharpe_ratios.append(info.get('sharpe_ratio', 0))
        
        # Calculate statistics
        metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'max_reward': np.max(total_rewards),
            'min_reward': np.min(total_rewards),
            'mean_portfolio_value': np.mean(portfolio_values),
            'mean_win_rate': np.mean(win_rates),
            'mean_sharpe_ratio': np.mean(sharpe_ratios),
            'success_rate': np.mean([r > 0 for r in total_rewards])
        }
        
        return metrics
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for a state
        
        Args:
            state: State to evaluate
            
        Returns:
            Q-values for all actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            if hasattr(self, 'device'):
                state_tensor = state_tensor.to(self.device)
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().squeeze()
    
    def save(self, filepath: str):
        """
        Save agent to disk
        
        Args:
            filepath: Path to save agent
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dictionary for safer serialization
        config_dict = {
            'state_dim': self.config.state_dim,
            'action_dim': self.config.action_dim,
            'hidden_dims': self.config.hidden_dims,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'gamma': self.config.gamma,
            'tau': self.config.tau,
            'epsilon_start': self.config.epsilon_start,
            'epsilon_end': self.config.epsilon_end,
            'epsilon_decay': self.config.epsilon_decay,
            'memory_size': self.config.memory_size,
            'min_memory_size': self.config.min_memory_size,
            'update_every': self.config.update_every,
            'target_update_every': self.config.target_update_every,
            'use_double_dqn': self.config.use_double_dqn,
            'use_dueling_dqn': self.config.use_dueling_dqn,
            'use_prioritized_replay': self.config.use_prioritized_replay,
            'use_noisy_networks': self.config.use_noisy_networks,
        }
        
        checkpoint = {
            'config_dict': config_dict,  # Save as dict instead of dataclass
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'update_count': self.update_count,
            'training_history': self.training_history,
            'losses': self.losses[-1000:],  # Keep last 1000 losses
            'rewards': self.rewards[-1000:]  # Keep last 1000 rewards
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """
        Load agent from disk
        
        Args:
            filepath: Path to load agent from
        """
        # Load with weights_only=False for compatibility
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Reconstruct config from dictionary
        if 'config_dict' in checkpoint:
            config_dict = checkpoint['config_dict']
            self.config = DQNConfig(**config_dict)
        elif 'config' in checkpoint:
            # Handle old format
            self.config = checkpoint['config']
        
        # Reinitialize networks with loaded config
        self.q_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            dueling=self.config.use_dueling_dqn,
            noisy=self.config.use_noisy_networks
        ).to(self.device)
        
        self.target_network = DQNetwork(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dims,
            dueling=self.config.use_dueling_dqn,
            noisy=self.config.use_noisy_networks
        ).to(self.device)
        
        # Load network states
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        
        # Reinitialize optimizer and load state
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load other attributes
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.update_count = checkpoint.get('update_count', 0)
        self.training_history = checkpoint.get('training_history', [])
        self.losses = checkpoint.get('losses', [])
        self.rewards = checkpoint.get('rewards', [])
        
        # Update compatibility attributes
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.epsilon_min = self.config.epsilon_end
        
        logger.info(f"Agent loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress
        
        Returns:
            Training summary statistics
        """
        if not self.training_history:
            return {}
        
        recent_history = self.training_history[-100:]
        
        return {
            'total_episodes': self.episode_count,
            'total_steps': self.step_count,
            'current_epsilon': self.epsilon,
            'recent_avg_reward': np.mean([h['total_reward'] for h in recent_history]),
            'recent_avg_loss': np.mean([h['average_loss'] for h in recent_history]),
            'recent_avg_portfolio': np.mean([h['portfolio_value'] for h in recent_history]),
            'recent_avg_sharpe': np.mean([h['sharpe_ratio'] for h in recent_history]),
            'best_episode_reward': max([h['total_reward'] for h in self.training_history]),
            'memory_size': len(self.memory)
        }
    
    def plot_training_progress(self):
        """Plot training progress (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.training_history:
                logger.warning("No training history to plot")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            
            # Rewards
            episodes = [h['episode'] for h in self.training_history]
            rewards = [h['total_reward'] for h in self.training_history]
            axes[0, 0].plot(episodes, rewards, alpha=0.6)
            axes[0, 0].plot(episodes, pd.Series(rewards).rolling(20).mean(), linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            
            # Loss
            losses = [h['average_loss'] for h in self.training_history]
            axes[0, 1].plot(episodes, losses, alpha=0.6)
            axes[0, 1].plot(episodes, pd.Series(losses).rolling(20).mean(), linewidth=2)
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Loss')
            
            # Portfolio Value
            portfolio = [h['portfolio_value'] for h in self.training_history]
            axes[0, 2].plot(episodes, portfolio)
            axes[0, 2].set_title('Portfolio Value')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Value')
            
            # Win Rate
            win_rates = [h['win_rate'] for h in self.training_history]
            axes[1, 0].plot(episodes, win_rates)
            axes[1, 0].set_title('Win Rate')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            
            # Sharpe Ratio
            sharpe = [h['sharpe_ratio'] for h in self.training_history]
            axes[1, 1].plot(episodes, sharpe)
            axes[1, 1].set_title('Sharpe Ratio')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Sharpe')
            axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
            
            # Epsilon
            epsilons = [h['epsilon'] for h in self.training_history]
            axes[1, 2].plot(episodes, epsilons)
            axes[1, 2].set_title('Exploration Rate (Epsilon)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Epsilon')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    def decay_epsilon(self):
        """Decay epsilon for exploration"""
        if not self.config.use_noisy_networks:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class DoubleDQNAgent(DQNAgent):
    """
    Double DQN Agent
    Uses main network to select actions and target network to evaluate them
    """
    
    def __init__(self, state_dim: int = None, action_dim: int = None, 
                 config: Optional[DQNConfig] = None, **kwargs):
        """Initialize Double DQN with forced double DQN setting"""
        if config is None:
            config = DQNConfig()
        config.use_double_dqn = True
        
        # Pass all parameters to parent, not just config
        super().__init__(state_dim=state_dim, action_dim=action_dim, config=config, **kwargs)


class DuelingDQNAgent(DQNAgent):
    """
    Dueling DQN Agent
    Uses dueling network architecture
    """
    
    def __init__(self, state_dim: int = None, action_dim: int = None,
                 config: Optional[DQNConfig] = None, **kwargs):
        """Initialize Dueling DQN with forced dueling architecture"""
        if config is None:
            config = DQNConfig()
        config.use_dueling_dqn = True
        
        # Pass all parameters to parent, not just config
        super().__init__(state_dim=state_dim, action_dim=action_dim, config=config, **kwargs)


class RainbowDQNAgent(DQNAgent):
    """
    Rainbow DQN Agent
    Combines: Double DQN, Dueling, Noisy Nets, Prioritized Replay
    """
    
    def __init__(self, state_dim: int = None, action_dim: int = None,
                 config: Optional[DQNConfig] = None, **kwargs):
        """Initialize Rainbow DQN with all features enabled"""
        if config is None:
            config = DQNConfig()
            
        # Enable all Rainbow components
        config.use_double_dqn = True
        config.use_dueling_dqn = True
        config.use_prioritized_replay = True
        config.use_noisy_networks = True
        
        # Add Rainbow-specific defaults
        if 'num_atoms' in kwargs:
            config.num_atoms = kwargs.pop('num_atoms')
        if 'v_min' in kwargs:
            config.v_min = kwargs.pop('v_min')
        if 'v_max' in kwargs:
            config.v_max = kwargs.pop('v_max')
        
        # Pass all parameters to parent
        super().__init__(state_dim=state_dim, action_dim=action_dim, config=config, **kwargs)
        
        # Rainbow uses noisy networks, so disable epsilon-greedy
        self.epsilon = 0.0
        self.epsilon_min = 0.0


class DQNAgentFactory:
    """Factory for creating DQN agents"""
    
    @staticmethod
    def create_agent(agent_type: str,
                    state_dim: int,
                    action_dim: int,
                    config: Optional[Dict[str, Any]] = None) -> DQNAgent:
        """
        Create a DQN agent
        
        Args:
            agent_type: Type of agent ('dqn', 'double', 'dueling', 'rainbow')
            state_dim: State dimension
            action_dim: Action dimension
            config: Configuration dictionary
            
        Returns:
            DQN agent instance
        """
        if config is None:
            config = {}
        
        agent_classes = {
            'dqn': DQNAgent,
            'double': DoubleDQNAgent,
            'dueling': DuelingDQNAgent,
            'rainbow': RainbowDQNAgent
        }
        
        if agent_type not in agent_classes:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agent_classes.keys())}")
        
        # Create config object from dict
        agent_config = DQNConfig()
        agent_config.state_dim = state_dim
        agent_config.action_dim = action_dim
        
        # Apply config dict to config object
        for key, value in config.items():
            if hasattr(agent_config, key):
                setattr(agent_config, key, value)
        
        # Create agent with appropriate class
        return agent_classes[agent_type](
            state_dim=state_dim,
            action_dim=action_dim,
            config=agent_config
        )
    
    @staticmethod
    def get_default_config(agent_type: str) -> Dict[str, Any]:
        """Get default configuration for agent type"""
        base_config = {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'buffer_size': 10000,
            'memory_size': 10000,
            'batch_size': 32,
            'update_every': 4,
            'target_update_every': 100,
            'hidden_dims': [256, 128],
            'track_metrics': True
        }
        
        if agent_type == 'rainbow':
            base_config.update({
                'learning_rate': 1e-4,
                'num_atoms': 51,
                'v_min': -10,
                'v_max': 10,
                'alpha': 0.6,
                'beta_start': 0.4,
                'beta_frames': 100000,
                'use_noisy_networks': True,
                'use_prioritized_replay': True,
                'use_double_dqn': True,
                'use_dueling_dqn': True
            })
        elif agent_type == 'double':
            base_config['use_double_dqn'] = True
        elif agent_type == 'dueling':
            base_config['use_dueling_dqn'] = True
        
        return base_config


# Testing function
if __name__ == "__main__":
    print("=== DQN Agent Test ===\n")
    
    # Test configuration
    config = DQNConfig(
        state_dim=63,
        action_dim=3,
        hidden_dims=[256, 256, 128],
        use_double_dqn=True,
        use_dueling_dqn=True,
        use_prioritized_replay=True,
        use_noisy_networks=False
    )
    
    # Initialize agent
    agent = DQNAgent(config)
    print(f"Agent initialized on device: {agent.device}")
    print(f"Network architecture: {agent.config.hidden_dims}")
    print(f"Using Double DQN: {agent.config.use_double_dqn}")
    print(f"Using Dueling DQN: {agent.config.use_dueling_dqn}")
    print(f"Using Prioritized Replay: {agent.config.use_prioritized_replay}")
    print(f"Using Noisy Networks: {agent.config.use_noisy_networks}\n")
    
    # Test forward pass
    print("Testing network forward pass...")
    dummy_state = np.random.randn(config.state_dim)
    q_values = agent.get_q_values(dummy_state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}\n")
    
    # Test action selection
    print("Testing action selection...")
    for i in range(5):
        action = agent.act(dummy_state, training=True)
        print(f"Step {i+1}: Action = {action}, Epsilon = {agent.epsilon:.4f}")
    
    # Test memory and replay
    print("\nTesting experience replay...")
    for i in range(100):
        state = np.random.randn(config.state_dim)
        action = np.random.randint(0, config.action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(config.state_dim)
        done = np.random.random() > 0.9
        
        agent.remember(state, action, reward, next_state, done)
    
    print(f"Memory size: {len(agent.memory)}")
    
    # Test training step
    print("\nTesting training...")
    for i in range(5):
        loss = agent.replay()
        print(f"Training step {i+1}: Loss = {loss:.6f}")
    
    # Test save/load
    print("\nTesting save/load...")
    agent.save("test_agent.pt")
    print("Agent saved successfully")
    
    new_agent = DQNAgent(config)
    new_agent.load("test_agent.pt")
    print("Agent loaded successfully")
    
    # Clean up test file
    import os
    if os.path.exists("test_agent.pt"):
        os.remove("test_agent.pt")
    
    print("\n=== DQN Agent Ready for Trading! ===")