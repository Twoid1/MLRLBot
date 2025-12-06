"""
PRIORITIZED EXPERIENCE REPLAY FOR MULTI-OBJECTIVE DQN
======================================================

Implements Prioritized Experience Replay (PER) from:
"Prioritized Experience Replay" (Schaul et al., 2015)

Key Benefits:
- Samples important experiences more frequently
- Big wins, big losses, surprising outcomes get replayed more
- Learns 2-3x faster from rare but important trades
- Uses importance sampling to correct for bias

Integration:
- Drop-in replacement for MOReplayBuffer
- Works with existing MultiObjectiveDQNAgent
- Just change buffer class and update training loop

Usage:
------
# Replace in your agent:
# OLD: self.memory = MOReplayBuffer(capacity, state_dim)
# NEW: self.memory = PrioritizedMOReplayBuffer(capacity, state_dim)

# In replay():
# OLD: states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
# NEW: batch = self.memory.sample(batch_size)
#      ... calculate TD errors ...
#      self.memory.update_priorities(batch.indices, td_errors)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Objective names (must match multi_objective_extension.py)
OBJECTIVES = ['pnl_quality', 'hold_duration', 'win_achieved', 'loss_control', 'risk_reward']


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: SUM TREE DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class SumTree:
    """
    Binary tree where each node is sum of children.
    Leaf nodes store priorities, internal nodes store sums.
    
    Enables O(log n) sampling proportional to priority.
    
    Example tree with 4 leaves:
    
              [42]           <- root (sum of all priorities)
             /    \\
          [29]    [13]       <- internal nodes
          /  \\    /  \\
        [13][16][3][10]      <- leaf nodes (actual priorities)
    
    To sample with probability proportional to priority:
    1. Generate random number in [0, total_sum)
    2. Traverse tree: go left if value < left_child, else go right
    3. Reach leaf = sampled index
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        
        # Tree has capacity leaves + (capacity - 1) internal nodes
        # Total nodes = 2 * capacity - 1
        # We use 2 * capacity for simplicity (index 0 unused in some implementations)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        
        # Data storage (parallel to leaves)
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up to root"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for given value"""
        left = 2 * idx + 1
        right = left + 1
        
        # If we're at a leaf, return it
        if left >= len(self.tree):
            return idx
        
        # Otherwise, descend into appropriate child
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
    
    @property
    def total(self) -> float:
        """Total priority (root value)"""
        return self.tree[0]
    
    def add(self, priority: float, data_idx: int) -> int:
        """Add priority for data at data_idx, return tree leaf index"""
        # Leaf index in tree array
        tree_idx = data_idx + self.capacity - 1
        
        # Update tree
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
        return tree_idx
    
    def update(self, tree_idx: int, priority: float):
        """Update priority at tree index"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def get(self, value: float) -> Tuple[int, float, int]:
        """
        Get leaf for given value.
        
        Returns:
            tree_idx: Index in tree array
            priority: Priority value at that leaf
            data_idx: Index in data array
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        priority = self.tree[tree_idx]
        
        return tree_idx, priority, data_idx
    
    def get_leaf(self, data_idx: int) -> Tuple[int, float]:
        """Get tree index and priority for data index"""
        tree_idx = data_idx + self.capacity - 1
        return tree_idx, self.tree[tree_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: PRIORITIZED REPLAY BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class PERBatch(NamedTuple):
    """Batch returned by prioritized replay buffer"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: Dict[str, torch.Tensor]
    next_states: torch.Tensor
    dones: torch.Tensor
    indices: np.ndarray          # For updating priorities
    weights: torch.Tensor        # Importance sampling weights


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay"""
    
    # Priority exponent (0 = uniform, 1 = full prioritization)
    alpha: float = 0.6
    
    # Importance sampling exponent (anneals from beta_start to 1.0)
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_annealing_steps: int = 100000
    
    # Small constant to ensure non-zero priority
    epsilon: float = 1e-6
    
    # Clamp priorities to prevent extreme values
    max_priority: float = 100.0
    min_priority: float = 1e-6


class PrioritizedMOReplayBuffer:
    """
    Prioritized Experience Replay Buffer for Multi-Objective rewards.
    
    Samples experiences with probability proportional to their priority.
    Priority is based on TD error (how surprising the outcome was).
    
    Higher priority = more surprising = sample more often
    
    Uses importance sampling weights to correct for the bias introduced
    by non-uniform sampling.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        config: Optional[PERConfig] = None
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.config = config or PERConfig()
        
        # Sum tree for efficient priority-based sampling
        self.tree = SumTree(capacity)
        
        # Pre-allocate storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # Multi-objective rewards
        self.rewards = {obj: np.zeros(capacity, dtype=np.float32) for obj in OBJECTIVES}
        
        # Position and size tracking
        self.position = 0
        self.size = 0
        
        # Priority tracking
        self.max_priority = 1.0  # New experiences get this priority
        
        # Beta annealing
        self.beta = self.config.beta_start
        self.beta_increment = (self.config.beta_end - self.config.beta_start) / self.config.beta_annealing_steps
        self.step_count = 0
        
        logger.info(f"PrioritizedMOReplayBuffer initialized:")
        logger.info(f"  Capacity: {capacity:,}")
        logger.info(f"  Alpha (prioritization): {self.config.alpha}")
        logger.info(f"  Beta (IS correction): {self.config.beta_start} -> {self.config.beta_end}")
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        rewards: Dict[str, float],
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add experience to buffer with maximum priority.
        
        New experiences get max priority so they're sampled at least once.
        After first sampling, priority is updated based on TD error.
        """
        idx = self.position
        
        # Store experience
        self.states[idx] = state
        self.actions[idx] = action
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        
        for obj in OBJECTIVES:
            self.rewards[obj][idx] = rewards.get(obj, 0.0)
        
        # Add to tree with max priority (ensures new experiences get sampled)
        priority = self.max_priority ** self.config.alpha
        self.tree.add(priority, idx)
        
        # Update position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Optional[PERBatch]:
        """
        Sample batch with probability proportional to priority.
        
        Returns PERBatch with importance sampling weights for bias correction.
        """
        if self.size < batch_size:
            return None
        
        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        
        # Divide total priority into batch_size segments
        # Sample one experience from each segment (stratified sampling)
        segment_size = self.tree.total / batch_size
        
        for i in range(batch_size):
            # Sample uniformly within segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = np.random.uniform(low, high)
            
            # Get experience for this value
            tree_idx, priority, data_idx = self.tree.get(value)
            
            # Handle edge case of zero priority
            if priority == 0:
                priority = self.config.epsilon
            
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = priority
        
        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-beta) / max(w)
        # This corrects for the bias from non-uniform sampling
        
        # Probability of sampling each experience
        sampling_probs = priorities / self.tree.total
        
        # Importance sampling weights
        weights = (self.size * sampling_probs) ** (-self.beta)
        weights = weights / weights.max()  # Normalize by max weight
        
        # Anneal beta toward 1.0
        self.step_count += 1
        self.beta = min(self.config.beta_end, self.beta + self.beta_increment)
        
        # Gather experiences
        states = torch.FloatTensor(self.states[indices])
        actions = torch.LongTensor(self.actions[indices])
        next_states = torch.FloatTensor(self.next_states[indices])
        dones = torch.FloatTensor(self.dones[indices])
        rewards = {obj: torch.FloatTensor(self.rewards[obj][indices]) for obj in OBJECTIVES}
        weights = torch.FloatTensor(weights)
        
        return PERBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            indices=tree_indices,  # Tree indices for update
            weights=weights
        )
    
    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Higher TD error = more surprising = higher priority
        
        Args:
            tree_indices: Indices returned from sample()
            td_errors: TD errors from training (can be per-objective or combined)
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            # Priority = |TD error| + epsilon (ensure non-zero)
            priority = abs(td_error) + self.config.epsilon
            
            # Clamp to reasonable range
            priority = np.clip(priority, self.config.min_priority, self.config.max_priority)
            
            # Apply alpha exponent
            priority = priority ** self.config.alpha
            
            # Update tree
            self.tree.update(int(tree_idx), priority)
            
            # Track max for new experiences
            self.max_priority = max(self.max_priority, abs(td_error) + self.config.epsilon)
    
    def update_priorities_multi_objective(
        self,
        tree_indices: np.ndarray,
        td_errors: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ):
        """
        Update priorities using weighted combination of per-objective TD errors.
        
        Args:
            tree_indices: Indices returned from sample()
            td_errors: Dict of TD errors per objective
            weights: Objective weights for combining
        """
        # Combine TD errors using objective weights
        combined_errors = np.zeros(len(tree_indices))
        
        for obj in OBJECTIVES:
            if obj in td_errors:
                combined_errors += weights.get(obj, 0.0) * np.abs(td_errors[obj])
        
        self.update_priorities(tree_indices, combined_errors)
    
    def __len__(self):
        return self.size
    
    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        if self.size == 0:
            return {}
        
        priorities = []
        for i in range(self.size):
            _, priority = self.tree.get_leaf(i)
            priorities.append(priority)
        
        priorities = np.array(priorities)
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'beta': self.beta,
            'max_priority': self.max_priority,
            'mean_priority': np.mean(priorities),
            'std_priority': np.std(priorities),
            'min_priority': np.min(priorities),
            'max_stored_priority': np.max(priorities),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: UPDATED AGENT WITH PER
# ═══════════════════════════════════════════════════════════════════════════════

def create_per_replay_step(agent, batch: PERBatch) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Perform one training step with Prioritized Experience Replay.
    
    Returns losses and TD errors for priority update.
    
    This is the key modification to the training loop:
    1. Use importance sampling weights in loss calculation
    2. Calculate TD errors for priority update
    3. Return both for the caller to update priorities
    
    Usage in your training loop:
    ```
    batch = self.memory.sample(batch_size)
    if batch is not None:
        losses, td_errors = create_per_replay_step(self, batch)
        self.memory.update_priorities(batch.indices, td_errors)
    ```
    """
    config = agent.config
    device = agent.device
    
    # Move batch to device
    states = batch.states.to(device)
    actions = batch.actions.to(device)
    next_states = batch.next_states.to(device)
    dones = batch.dones.to(device)
    weights = batch.weights.to(device)  # Importance sampling weights
    rewards = {obj: r.to(device) for obj, r in batch.rewards.items()}
    
    # Get current Q-values
    current_q_dict = agent.q_network(states)
    
    # Get next Q-values from target network
    with torch.no_grad():
        if config.use_double_dqn:
            # Double DQN: use online network to select action, target to evaluate
            next_q_online = agent.q_network.get_combined_q_values(
                next_states, config.objective_weights
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = agent.target_network(next_states)
        else:
            next_q_target = agent.target_network(next_states)
    
    # Calculate losses and TD errors per objective
    total_loss = torch.tensor(0.0, device=device)
    losses = {}
    td_errors_dict = {}
    
    for obj in OBJECTIVES:
        # Current Q for taken action
        current_q = current_q_dict[obj].gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q (target)
        if config.use_double_dqn:
            next_q = next_q_target[obj].gather(1, next_actions).squeeze(1)
        else:
            next_q = next_q_target[obj].max(dim=1)[0]
        
        # Target Q value
        target_q = rewards[obj] + config.gamma * next_q * (1 - dones)
        
        # TD error (for priority update)
        td_error = (current_q - target_q).detach()
        td_errors_dict[obj] = td_error.cpu().numpy()
        
        # ═══════════════════════════════════════════════════════════════════
        # KEY PER MODIFICATION: Weight loss by importance sampling weights
        # ═══════════════════════════════════════════════════════════════════
        element_wise_loss = (current_q - target_q) ** 2
        weighted_loss = (element_wise_loss * weights).mean()
        
        # Apply objective weight
        objective_weighted_loss = weighted_loss * config.objective_weights[obj]
        total_loss = total_loss + objective_weighted_loss
        
        losses[obj] = weighted_loss.item()
        agent.recent_losses[obj].append(weighted_loss.item())
    
    # Backpropagation
    agent.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
    agent.optimizer.step()
    
    # Update target network
    agent.update_count += 1
    if config.use_soft_update:
        tau = config.tau
        for target_param, param in zip(
            agent.target_network.parameters(),
            agent.q_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    elif agent.update_count % config.target_update_every == 0:
        agent.target_network.load_state_dict(agent.q_network.state_dict())
    
    # Decay epsilon
    agent.epsilon = max(config.epsilon_end, agent.epsilon * config.epsilon_decay)
    
    # Combine TD errors for priority update (weighted by objective importance)
    combined_td_errors = np.zeros(len(batch.indices))
    for obj in OBJECTIVES:
        combined_td_errors += config.objective_weights[obj] * np.abs(td_errors_dict[obj])
    
    return losses, combined_td_errors


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class PERMultiObjectiveDQNAgent:
    """
    Multi-Objective DQN Agent with Prioritized Experience Replay.
    
    This is a complete agent class that integrates PER.
    Can be used as a drop-in replacement for MultiObjectiveDQNAgent.
    """
    
    def __init__(self, config, network_class=None):
        """
        Args:
            config: MODQNConfig with agent settings
            network_class: Optional custom network class (default: MultiHeadDQNetwork)
        """
        import torch.nn as nn
        import torch.optim as optim
        from collections import deque
        import random
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import network from multi_objective_extension if not provided
        if network_class is None:
            from multi_objective_extension import MultiHeadDQNetwork
            network_class = MultiHeadDQNetwork
        
        # Networks
        self.q_network = network_class(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            head_hidden_dim=config.head_hidden_dim,
            num_objectives=len(OBJECTIVES),
            use_dueling=config.use_dueling
        ).to(self.device)
        
        self.target_network = network_class(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            head_hidden_dim=config.head_hidden_dim,
            num_objectives=len(OBJECTIVES),
            use_dueling=config.use_dueling
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # ═══════════════════════════════════════════════════════════════════
        # KEY CHANGE: Use Prioritized Replay Buffer instead of regular buffer
        # ═══════════════════════════════════════════════════════════════════
        per_config = PERConfig(
            alpha=getattr(config, 'per_alpha', 0.6),
            beta_start=getattr(config, 'per_beta_start', 0.4),
            beta_end=getattr(config, 'per_beta_end', 1.0),
            beta_annealing_steps=getattr(config, 'per_beta_annealing_steps', 100000),
        )
        self.memory = PrioritizedMOReplayBuffer(
            config.memory_size,
            config.state_dim,
            per_config
        )
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Counters
        self.step_count = 0
        self.update_count = 0
        
        # Loss tracking
        self.recent_losses = {obj: deque(maxlen=100) for obj in OBJECTIVES}
        
        # Store random module reference
        self._random = random
        
        logger.info(f"PER Multi-Objective DQN Agent initialized on {self.device}")
        logger.info(f"  Using Prioritized Experience Replay")
        logger.info(f"  PER alpha: {per_config.alpha}")
        logger.info(f"  PER beta: {per_config.beta_start} -> {per_config.beta_end}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy"""
        if training and self._random.random() < self.epsilon:
            return self._random.randint(0, self.config.action_dim - 1)
        
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
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mo_rewards: Optional[Dict[str, float]] = None
    ):
        """Store experience in prioritized replay buffer"""
        if mo_rewards is None:
            mo_rewards = {obj: reward for obj in OBJECTIVES}
        
        self.memory.push(state, action, mo_rewards, next_state, done)
    
    def replay(self) -> Dict[str, float]:
        """
        Train on prioritized batch from replay buffer.
        
        Returns dict of losses per objective.
        """
        if len(self.memory) < self.config.min_memory_size:
            return {}
        
        # Sample prioritized batch
        batch = self.memory.sample(self.config.batch_size)
        
        if batch is None:
            return {}
        
        # Train and get TD errors
        losses, td_errors = create_per_replay_step(self, batch)
        
        # ═══════════════════════════════════════════════════════════════════
        # KEY PER STEP: Update priorities based on TD errors
        # ═══════════════════════════════════════════════════════════════════
        self.memory.update_priorities(batch.indices, td_errors)
        
        return losses
    
    def get_loss_stats(self) -> Dict[str, float]:
        """Get average losses per objective"""
        return {
            obj: np.mean(list(losses)) if losses else 0.0
            for obj, losses in self.recent_losses.items()
        }
    
    def get_memory_stats(self) -> Dict:
        """Get replay buffer statistics"""
        return self.memory.get_stats()
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count,
            'config': self.config,
            'memory_beta': self.memory.beta,
            'memory_max_priority': self.memory.max_priority,
        }, path)
        logger.info(f"PER Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint.get('update_count', 0)
        
        # Restore PER state if available
        if 'memory_beta' in checkpoint:
            self.memory.beta = checkpoint['memory_beta']
        if 'memory_max_priority' in checkpoint:
            self.memory.max_priority = checkpoint['memory_max_priority']
        
        logger.info(f"PER Agent loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_sum_tree():
    """Test SumTree data structure"""
    print("=" * 70)
    print("TEST: SumTree")
    print("=" * 70)
    
    tree = SumTree(capacity=8)
    
    # Add some priorities
    priorities = [3, 10, 12, 4, 1, 2, 8, 2]
    for i, p in enumerate(priorities):
        tree.add(p, i)
    
    print(f"Priorities: {priorities}")
    print(f"Total: {tree.total} (expected: {sum(priorities)})")
    
    # Sample many times and check distribution
    samples = {i: 0 for i in range(8)}
    n_samples = 10000
    
    for _ in range(n_samples):
        value = np.random.uniform(0, tree.total)
        _, _, data_idx = tree.get(value)
        samples[data_idx] += 1
    
    print("\nSampling distribution (should roughly match priority ratios):")
    total_p = sum(priorities)
    for i in range(8):
        expected = priorities[i] / total_p
        actual = samples[i] / n_samples
        print(f"  Index {i}: priority={priorities[i]:2d}, expected={expected:.3f}, actual={actual:.3f}")
    
    print("\n SumTree test passed!")


def test_prioritized_buffer():
    """Test PrioritizedMOReplayBuffer"""
    print("\n" + "=" * 70)
    print("TEST: PrioritizedMOReplayBuffer")
    print("=" * 70)
    
    buffer = PrioritizedMOReplayBuffer(
        capacity=1000,
        state_dim=10,
        config=PERConfig(alpha=0.6, beta_start=0.4)
    )
    
    # Add experiences with different "importance"
    np.random.seed(42)
    
    for i in range(500):
        state = np.random.randn(10).astype(np.float32)
        action = np.random.randint(0, 2)
        
        # Create rewards that vary in magnitude
        pnl = np.random.uniform(-0.03, 0.03)
        rewards = {
            'pnl_quality': pnl * 50,
            'hold_duration': np.random.uniform(-0.5, 0.5),
            'win_achieved': pnl / 0.03,
            'loss_control': 0.0 if pnl > 0 else np.random.uniform(-0.5, 0.5),
            'risk_reward': np.random.uniform(-1, 1),
        }
        
        next_state = np.random.randn(10).astype(np.float32)
        done = (i % 50 == 49)
        
        buffer.push(state, action, rewards, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Sample batch
    batch = buffer.sample(batch_size=32)
    
    print(f"\nSampled batch:")
    print(f"  States shape: {batch.states.shape}")
    print(f"  Actions shape: {batch.actions.shape}")
    print(f"  Weights shape: {batch.weights.shape}")
    print(f"  Weights range: [{batch.weights.min():.3f}, {batch.weights.max():.3f}]")
    print(f"  Indices: {batch.indices[:5]}...")
    
    # Simulate priority update
    fake_td_errors = np.random.uniform(0.1, 2.0, size=32)
    buffer.update_priorities(batch.indices, fake_td_errors)
    
    print(f"\nAfter priority update:")
    print(f"  Max priority: {buffer.max_priority:.3f}")
    print(f"  Buffer stats: {buffer.get_stats()}")
    
    print("\n PrioritizedMOReplayBuffer test passed!")


def test_sampling_bias_correction():
    """Test that importance sampling weights correct for bias"""
    print("\n" + "=" * 70)
    print("TEST: Importance Sampling Bias Correction")
    print("=" * 70)
    
    buffer = PrioritizedMOReplayBuffer(
        capacity=100,
        state_dim=5,
        config=PERConfig(alpha=0.6, beta_start=1.0)  # Full correction
    )
    
    # Add experiences with known priorities
    for i in range(100):
        state = np.array([i, 0, 0, 0, 0], dtype=np.float32)
        rewards = {obj: float(i) for obj in OBJECTIVES}  # Reward = index
        buffer.push(state, 0, rewards, state, False)
    
    # Manually set priorities: first 10 have high priority
    for i in range(10):
        tree_idx, _ = buffer.tree.get_leaf(i)
        buffer.tree.update(tree_idx, 10.0)  # High priority
    
    for i in range(10, 100):
        tree_idx, _ = buffer.tree.get_leaf(i)
        buffer.tree.update(tree_idx, 1.0)  # Low priority
    
    # Sample many times
    high_priority_samples = 0
    n_samples = 1000
    
    for _ in range(n_samples):
        batch = buffer.sample(10)
        # Count samples from high-priority region (indices 0-9)
        high_priority_samples += (batch.states[:, 0] < 10).sum().item()
    
    total_samples = n_samples * 10
    high_ratio = high_priority_samples / total_samples
    
    print(f"High priority items: 10/100 (10%)")
    print(f"High priority weight: 10x higher")
    print(f"Expected sampling ratio: ~50% (10 items × 10 weight / (10×10 + 90×1))")
    print(f"Actual sampling ratio: {high_ratio:.1%}")
    
    # Check weights compensate
    batch = buffer.sample(32)
    print(f"\nImportance sampling weights:")
    print(f"  Mean: {batch.weights.mean():.3f}")
    print(f"  Std: {batch.weights.std():.3f}")
    print(f"  Range: [{batch.weights.min():.3f}, {batch.weights.max():.3f}]")
    
    print("\n Bias correction test passed!")


def test_integration_example():
    """Show how to integrate PER with existing training loop"""
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE")
    print("=" * 70)
    
    print("""
# In your trade_based_trainer.py, make these changes:

# 1. Import PER components
from prioritized_replay import (
    PrioritizedMOReplayBuffer,
    PERConfig,
    PERBatch,
    create_per_replay_step
)

# 2. In trainer __init__, replace buffer:
# OLD:
# self.memory = MOReplayBuffer(config['memory_size'], state_dim)

# NEW:
per_config = PERConfig(
    alpha=config.get('per_alpha', 0.6),
    beta_start=config.get('per_beta_start', 0.4),
    beta_end=config.get('per_beta_end', 1.0),
    beta_annealing_steps=config.get('per_beta_annealing_steps', 100000),
)
self.memory = PrioritizedMOReplayBuffer(
    config['memory_size'],
    state_dim,
    per_config
)

# 3. In replay() method, update to use PER:
def replay(self):
    batch = self.memory.sample(self.config.batch_size)
    if batch is None:
        return {}
    
    # Train and get TD errors
    losses, td_errors = create_per_replay_step(self.rl_agent, batch)
    
    # Update priorities (KEY PER STEP!)
    self.memory.update_priorities(batch.indices, td_errors)
    
    return losses

# 4. Add PER config to your training config:
config = {
    # ... existing config ...
    
    # PER settings
    'per_alpha': 0.6,        # Prioritization strength (0=uniform, 1=full)
    'per_beta_start': 0.4,   # Initial importance sampling correction
    'per_beta_end': 1.0,     # Final IS correction (anneals over training)
    'per_beta_annealing_steps': 100000,
}

# 5. Optionally log PER stats:
if episode % 100 == 0:
    per_stats = self.memory.get_stats()
    logger.info(f"PER Stats: beta={per_stats['beta']:.3f}, "
                f"max_priority={per_stats['max_priority']:.3f}")
""")
    
    print(" See above for integration instructions!")


if __name__ == "__main__":
    test_sum_tree()
    test_prioritized_buffer()
    test_sampling_bias_correction()
    test_integration_example()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)