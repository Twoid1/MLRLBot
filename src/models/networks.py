"""
Neural Network Architectures for DQN Agent
Built from scratch for complete control over the RL component
Includes standard DQN, Dueling DQN, and Noisy Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math


class DQNetwork(nn.Module):
    """
    Standard Deep Q-Network architecture
    
    Architecture:
        Input (state_dim) → FC(256) → ReLU → FC(256) → ReLU → FC(128) → ReLU → Output (action_dim)
    
    Designed for trading with:
    - State: market features + ML predictions + position info
    - Actions: HOLD (0), BUY (1), SELL (2)
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 dropout_rate: float = 0.0,
                 activation: str = 'relu'):
        """
        Initialize DQN
        
        Args:
            state_dim: Dimension of state space (~63-100 for our features)
            action_dim: Number of actions (3: HOLD, BUY, SELL)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability (0 = no dropout)
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(DQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Choose activation function
        self.activation = self._get_activation(activation)
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        x = state
        
        # Pass through hidden layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)  # Dropout or other layers
        
        # Output Q-values (no activation)
        q_values = self.output_layer(x)
        
        return q_values
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'tanh': torch.tanh
        }
        return activations.get(activation, F.relu)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # Output layer with smaller initialization
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture
    Splits Q-value into Value and Advantage streams
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    
    Better for environments where action choice doesn't always matter
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 value_dims: List[int] = [128],
                 advantage_dims: List[int] = [128],
                 dropout_rate: float = 0.0):
        """
        Initialize Dueling DQN
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Shared layer dimensions
            value_dims: Value stream dimensions
            advantage_dims: Advantage stream dimensions
            dropout_rate: Dropout probability
        """
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            if dropout_rate > 0:
                self.shared_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        shared_output_dim = prev_dim
        
        # Value stream
        self.value_layers = nn.ModuleList()
        prev_dim = shared_output_dim
        for value_dim in value_dims:
            self.value_layers.append(nn.Linear(prev_dim, value_dim))
            if dropout_rate > 0:
                self.value_layers.append(nn.Dropout(dropout_rate))
            prev_dim = value_dim
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Advantage stream
        self.advantage_layers = nn.ModuleList()
        prev_dim = shared_output_dim
        for advantage_dim in advantage_dims:
            self.advantage_layers.append(nn.Linear(prev_dim, advantage_dim))
            if dropout_rate > 0:
                self.advantage_layers.append(nn.Dropout(dropout_rate))
            prev_dim = advantage_dim
        self.advantage_head = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        # Shared network
        x = state
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x)
        
        # Value stream
        value = x
        for layer in self.value_layers:
            if isinstance(layer, nn.Linear):
                value = F.relu(layer(value))
            else:
                value = layer(value)
        value = self.value_head(value)
        
        # Advantage stream
        advantage = x
        for layer in self.advantage_layers:
            if isinstance(layer, nn.Linear):
                advantage = F.relu(layer(advantage))
            else:
                advantage = layer(advantage)
        advantage = self.advantage_head(advantage)
        
        # Combine streams (subtracting mean for stability)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in [self.shared_layers, self.value_layers, self.advantage_layers]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
        
        # Output layers with smaller initialization
        nn.init.xavier_uniform_(self.value_head.weight, gain=0.1)
        nn.init.constant_(self.value_head.bias, 0)
        nn.init.xavier_uniform_(self.advantage_head.weight, gain=0.1)
        nn.init.constant_(self.advantage_head.bias, 0)


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for NoisyNet-DQN
    Adds learnable noise to weights for exploration
    Better than epsilon-greedy in some cases
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize NoisyLinear layer
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            std_init: Initial standard deviation for noise
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise (call at each step)"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """
    DQN with NoisyNet layers for exploration
    Replaces epsilon-greedy with parameter noise
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256, 128],
                 std_init: float = 0.5):
        """
        Initialize Noisy DQN
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            std_init: Initial noise standard deviation
        """
        super(NoisyDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network with noisy layers
        self.layers = nn.ModuleList()
        
        # First layer (standard)
        self.layers.append(nn.Linear(state_dim, hidden_dims[0]))
        
        # Hidden layers (noisy)
        for i in range(len(hidden_dims) - 1):
            self.layers.append(NoisyLinear(hidden_dims[i], hidden_dims[i + 1], std_init))
        
        # Output layer (noisy)
        self.output_layer = NoisyLinear(hidden_dims[-1], action_dim, std_init)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = F.relu(self.layers[0](state))
        
        for layer in self.layers[1:]:
            x = F.relu(layer(x))
        
        return self.output_layer(x)
    
    def reset_noise(self):
        """Reset all noisy layers"""
        for layer in self.layers[1:]:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        self.output_layer.reset_noise()


class CategoricalDQN(nn.Module):
    """
    Categorical DQN (C51) for distributional RL
    Models the full distribution of returns instead of just expected value
    More robust for financial environments with heavy-tailed distributions
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 hidden_dims: List[int] = [256, 256, 128]):
        """
        Initialize Categorical DQN
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_atoms: Number of atoms for distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
            hidden_dims: Hidden layer dimensions
        """
        super(CategoricalDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support of the distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta = (v_max - v_min) / (num_atoms - 1)
        
        # Network layers
        self.layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        # Output: distribution over atoms for each action
        self.output_layer = nn.Linear(prev_dim, action_dim * num_atoms)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            Distribution logits [batch_size, action_dim, num_atoms]
        """
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        
        logits = self.output_layer(x)
        logits = logits.view(-1, self.action_dim, self.num_atoms)
        
        # Apply log softmax over atoms dimension
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values by taking expectation over distribution
        
        Args:
            state: State tensor
            
        Returns:
            Q-values for each action
        """
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=2)
        return q_values


class RainbowDQN(nn.Module):
    """
    Rainbow DQN combining multiple improvements:
    - Double DQN (handled in agent)
    - Dueling architecture
    - Noisy networks
    - Categorical distribution
    - Multi-step learning (handled in agent)
    - Prioritized replay (handled in buffer)
    
    State-of-the-art DQN variant for complex environments
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_atoms: int = 51,
                 v_min: float = -10.0,
                 v_max: float = 10.0,
                 hidden_dims: List[int] = [256, 256],
                 std_init: float = 0.5):
        """
        Initialize Rainbow DQN
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_atoms: Number of atoms for distribution
            v_min: Minimum value of support
            v_max: Maximum value of support
            hidden_dims: Hidden layer dimensions
            std_init: Noise initialization
        """
        super(RainbowDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        # Shared layers with noise
        self.shared_layers = nn.ModuleList()
        self.shared_layers.append(nn.Linear(state_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.shared_layers.append(NoisyLinear(hidden_dims[i], hidden_dims[i + 1], std_init))
        
        # Value stream (noisy)
        self.value_stream = NoisyLinear(hidden_dims[-1], num_atoms, std_init)
        
        # Advantage stream (noisy)
        self.advantage_stream = NoisyLinear(hidden_dims[-1], action_dim * num_atoms, std_init)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            Distribution log probabilities [batch_size, action_dim, num_atoms]
        """
        # Shared network
        x = F.relu(self.shared_layers[0](state))
        for layer in self.shared_layers[1:]:
            x = F.relu(layer(x))
        
        # Value and advantage streams
        value = self.value_stream(x).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(-1, self.action_dim, self.num_atoms)
        
        # Combine streams (dueling)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from distribution"""
        log_probs = self.forward(state)
        probs = log_probs.exp()
        q_values = (probs * self.support).sum(dim=2)
        return q_values
    
    def reset_noise(self):
        """Reset noise in all noisy layers"""
        for layer in self.shared_layers[1:]:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()


class NetworkFactory:
    """
    Factory class for creating different network architectures
    """
    
    @staticmethod
    def create_network(network_type: str,
                      state_dim: int,
                      action_dim: int,
                      config: Dict[str, Any] = None) -> nn.Module:
        """
        Create a network of specified type
        
        Args:
            network_type: Type of network ('dqn', 'dueling', 'noisy', 'categorical', 'rainbow')
            state_dim: State space dimension
            action_dim: Action space dimension
            config: Additional configuration
            
        Returns:
            Neural network module
        """
        config = config or {}
        
        if network_type == 'dqn':
            return DQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('hidden_dims', [256, 256, 128]),
                dropout_rate=config.get('dropout_rate', 0.0),
                activation=config.get('activation', 'relu')
            )
        
        elif network_type == 'dueling':
            return DuelingDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('hidden_dims', [256, 256]),
                value_dims=config.get('value_dims', [128]),
                advantage_dims=config.get('advantage_dims', [128]),
                dropout_rate=config.get('dropout_rate', 0.0)
            )
        
        elif network_type == 'noisy':
            return NoisyDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.get('hidden_dims', [256, 256, 128]),
                std_init=config.get('std_init', 0.5)
            )
        
        elif network_type == 'categorical':
            return CategoricalDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                num_atoms=config.get('num_atoms', 51),
                v_min=config.get('v_min', -10.0),
                v_max=config.get('v_max', 10.0),
                hidden_dims=config.get('hidden_dims', [256, 256, 128])
            )
        
        elif network_type == 'rainbow':
            return RainbowDQN(
                state_dim=state_dim,
                action_dim=action_dim,
                num_atoms=config.get('num_atoms', 51),
                v_min=config.get('v_min', -10.0),
                v_max=config.get('v_max', 10.0),
                hidden_dims=config.get('hidden_dims', [256, 256]),
                std_init=config.get('std_init', 0.5)
            )
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")


# Testing
if __name__ == "__main__":
    import time
    
    print("=== Testing Neural Network Architectures ===\n")
    
    # Test parameters
    state_dim = 63  # Typical state dimension for our trading env
    action_dim = 3  # HOLD, BUY, SELL
    batch_size = 32
    
    # Create sample input
    sample_state = torch.randn(batch_size, state_dim)
    
    # Test each network type
    networks = {
        'Standard DQN': DQNetwork(state_dim, action_dim),
        'Dueling DQN': DuelingDQN(state_dim, action_dim),
        'Noisy DQN': NoisyDQN(state_dim, action_dim),
        'Categorical DQN': CategoricalDQN(state_dim, action_dim),
        'Rainbow DQN': RainbowDQN(state_dim, action_dim)
    }
    
    for name, network in networks.items():
        print(f"Testing {name}...")
        
        # Forward pass
        start_time = time.time()
        
        if name == 'Categorical DQN' or name == 'Rainbow DQN':
            output = network.get_q_values(sample_state)
        else:
            output = network(sample_state)
        
        forward_time = time.time() - start_time
        
        # Count parameters
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Forward pass time: {forward_time*1000:.2f}ms")
        
        # Test gradient flow
        if output.requires_grad:
            loss = output.mean()
            loss.backward()
            print(f"  Gradient flow: Y")
        
        print()
    
    # Test NetworkFactory
    print("Testing NetworkFactory...")
    factory_net = NetworkFactory.create_network(
        'dueling', 
        state_dim, 
        action_dim,
        config={'hidden_dims': [128, 128], 'dropout_rate': 0.1}
    )
    output = factory_net(sample_state)
    print(f"Factory network output shape: {output.shape}")
    print(f"Factory network type: {type(factory_net).__name__}")
    
    print("\n=== All Networks Tested Successfully ===")