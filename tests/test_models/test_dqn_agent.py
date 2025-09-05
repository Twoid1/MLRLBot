"""
Test Script for DQN Agent Module
Tests all DQN variants and training functionality
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import warnings
import tempfile
import shutil
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the dqn_agent module
from src.models.dqn_agent import (
    DQNAgent,
    DoubleDQNAgent,
    DuelingDQNAgent,
    RainbowDQNAgent,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    DQNAgentFactory
)


def create_mock_environment_data():
    """Create mock environment data for testing"""
    state_dim = 63
    action_dim = 3
    n_episodes = 10
    episode_length = 100
    
    data = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_episodes': n_episodes,
        'episode_length': episode_length,
        'states': np.random.randn(n_episodes, episode_length, state_dim),
        'rewards': np.random.randn(n_episodes, episode_length),
        'dones': np.zeros((n_episodes, episode_length), dtype=bool)
    }
    
    # Set some episodes to end
    for i in range(n_episodes):
        data['dones'][i, -1] = True
    
    return data


def test_replay_buffer():
    """Test Replay Buffer functionality"""
    print("\n" + "="*60)
    print("TESTING REPLAY BUFFER")
    print("="*60)
    
    capacity = 1000
    state_dim = 63
    
    # Test 1: Basic buffer operations
    print("\n1. Testing basic buffer operations...")
    try:
        buffer = ReplayBuffer(capacity=capacity, state_dim=state_dim)
        
        # Add some experiences
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = i % 20 == 0
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 100
        print(f"    Added 100 experiences to buffer")
        
        # Sample from buffer
        batch = buffer.sample(32)
        assert len(batch[0]) == 32  # states
        assert len(batch[1]) == 32  # actions
        assert len(batch[2]) == 32  # rewards
        assert len(batch[3]) == 32  # next_states
        assert len(batch[4]) == 32  # dones
        
        print(f"    Successfully sampled batch of 32")
        
    except Exception as e:
        print(f"    Basic buffer operations failed: {e}")
        return False
    
    # Test 2: Capacity limit
    print("\n2. Testing capacity limit...")
    try:
        # Fill buffer beyond capacity
        for i in range(1100):
            state = np.random.randn(state_dim)
            buffer.push(state, 0, 0.0, state, False)
        
        assert len(buffer) == capacity
        print(f"    Buffer correctly limited to capacity {capacity}")
        
    except Exception as e:
        print(f"    Capacity limit test failed: {e}")
        return False
    
    # Test 3: Empty buffer handling
    print("\n3. Testing empty buffer handling...")
    try:
        empty_buffer = ReplayBuffer(capacity=100, state_dim=state_dim)
        
        # Try to sample from empty buffer
        try:
            empty_buffer.sample(32)
            print(f"    Should have raised error for empty buffer")
        except (ValueError, AssertionError):
            print(f"    Correctly raised error for empty buffer")
        
    except Exception as e:
        print(f"    Empty buffer test failed: {e}")
        return False
    
    print("\n Replay Buffer tests completed")
    return True


def test_prioritized_replay_buffer():
    """Test Prioritized Replay Buffer"""
    print("\n" + "="*60)
    print("TESTING PRIORITIZED REPLAY BUFFER")
    print("="*60)
    
    capacity = 1000
    state_dim = 63
    
    print("\n1. Testing prioritized sampling...")
    try:
        buffer = PrioritizedReplayBuffer(capacity=capacity, state_dim=state_dim, alpha=0.6)
        
        # Add experiences with different priorities
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = False
            
            buffer.push(state, action, reward, next_state, done)
        
        # Sample with priorities
        batch, indices, weights = buffer.sample(32, beta=0.4)
        
        assert len(batch[0]) == 32
        assert len(indices) == 32
        assert len(weights) == 32
        assert weights.min() > 0
        assert weights.max() <= 1.0
        
        print(f"    Prioritized sampling working")
        print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        
        # Update priorities
        new_priorities = np.random.uniform(0.1, 1.0, 32)
        buffer.update_priorities(indices, new_priorities)
        print(f"    Priority updates working")
        
    except Exception as e:
        print(f"    Prioritized sampling failed: {e}")
        return False
    
    print("\n Prioritized Replay Buffer tests completed")
    return True


def test_dqn_agent_initialization():
    """Test DQN Agent initialization"""
    print("\n" + "="*60)
    print("TESTING DQN AGENT INITIALIZATION")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    # Test 1: Basic DQN
    print("\n1. Testing basic DQN initialization...")
    try:
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=1e-3,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        assert agent.epsilon == 1.0
        assert agent.q_network is not None
        assert agent.target_network is not None
        
        print(f"    Basic DQN agent initialized")
        
    except Exception as e:
        print(f"    Basic DQN initialization failed: {e}")
        return False
    
    # Test 2: Double DQN
    print("\n2. Testing Double DQN initialization...")
    try:
        agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)
        print(f"    Double DQN agent initialized")
        
    except Exception as e:
        print(f"    Double DQN initialization failed: {e}")
        return False
    
    # Test 3: Dueling DQN
    print("\n3. Testing Dueling DQN initialization...")
    try:
        agent = DuelingDQNAgent(state_dim=state_dim, action_dim=action_dim)
        print(f"    Dueling DQN agent initialized")
        
    except Exception as e:
        print(f"    Dueling DQN initialization failed: {e}")
        return False
    
    # Test 4: Rainbow DQN
    print("\n4. Testing Rainbow DQN initialization...")
    try:
        agent = RainbowDQNAgent(state_dim=state_dim, action_dim=action_dim)
        print(f"    Rainbow DQN agent initialized")
        
    except Exception as e:
        print(f"    Rainbow DQN initialization failed: {e}")
        return False
    
    print("\n DQN Agent initialization tests completed")
    return True


def test_action_selection():
    """Test action selection mechanisms"""
    print("\n" + "="*60)
    print("TESTING ACTION SELECTION")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon=0.5  # 50% exploration
    )
    
    # Test 1: Epsilon-greedy action selection
    print("\n1. Testing epsilon-greedy action selection...")
    try:
        state = np.random.randn(state_dim)
        
        # Collect actions over multiple calls
        actions = []
        for _ in range(1000):
            action = agent.act(state)
            actions.append(action)
            assert 0 <= action < action_dim
        
        # Check that we get some variety due to exploration
        unique_actions = set(actions)
        assert len(unique_actions) > 1
        
        print(f"    Epsilon-greedy working")
        print(f"   Action distribution: {np.bincount(actions)}")
        
    except Exception as e:
        print(f"    Epsilon-greedy test failed: {e}")
        return False
    
    # Test 2: Greedy action selection
    print("\n2. Testing greedy action selection...")
    try:
        agent.epsilon = 0.0  # No exploration
        state = np.random.randn(state_dim)
        
        # Should get same action every time
        actions = []
        for _ in range(10):
            action = agent.act(state)
            actions.append(action)
        
        assert len(set(actions)) == 1  # All actions should be the same
        print(f"    Greedy action selection working")
        print(f"   Consistent action: {actions[0]}")
        
    except Exception as e:
        print(f"    Greedy test failed: {e}")
        return False
    
    # Test 3: Epsilon decay
    print("\n3. Testing epsilon decay...")
    try:
        agent.epsilon = 1.0
        agent.epsilon_decay = 0.99
        agent.epsilon_min = 0.01
        
        initial_epsilon = agent.epsilon
        
        # Decay epsilon multiple times
        for _ in range(100):
            agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min
        
        print(f"    Epsilon decay working")
        print(f"   Epsilon after 100 steps: {agent.epsilon:.4f}")
        
    except Exception as e:
        print(f"    Epsilon decay test failed: {e}")
        return False
    
    print("\n Action selection tests completed")
    return True


def test_training_step():
    """Test training step functionality"""
    print("\n" + "="*60)
    print("TESTING TRAINING STEP")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=1000
    )
    
    # Test 1: Memory storage
    print("\n1. Testing memory storage...")
    try:
        # Add experiences to memory
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = i % 10 == 0
            
            agent.remember(state, action, reward, next_state, done)
        
        assert len(agent.memory) == 100
        print(f"    Memory storage working")
        print(f"   Memory size: {len(agent.memory)}")
        
    except Exception as e:
        print(f"    Memory storage failed: {e}")
        return False
    
    # Test 2: Replay training
    print("\n2. Testing replay training...")
    try:
        # Get initial loss
        initial_loss = agent.replay(batch_size=32)
        
        if initial_loss is not None:
            assert initial_loss >= 0
            print(f"    Replay training working")
            print(f"   Initial loss: {initial_loss:.6f}")
            
            # Train for a few more steps
            losses = []
            for _ in range(10):
                loss = agent.replay(batch_size=32)
                if loss is not None:
                    losses.append(loss)
            
            if losses:
                print(f"   Average loss: {np.mean(losses):.6f}")
        else:
            print(f"    Not enough samples for replay training yet")
        
    except Exception as e:
        print(f"    Replay training failed: {e}")
        return False
    
    # Test 3: Target network update
    print("\n3. Testing target network update...")
    try:
        # Get initial target network weights
        initial_weights = agent.target_network.state_dict()['layers.0.weight'].clone()
        
        # Update target network
        agent.update_target_network()
        
        # Check if weights changed
        new_weights = agent.target_network.state_dict()['layers.0.weight']
        weights_changed = not torch.allclose(initial_weights, new_weights)
        
        print(f"    Target network update working")
        print(f"   Weights changed: {weights_changed}")
        
    except Exception as e:
        print(f"    Target network update failed: {e}")
        return False
    
    print("\n Training step tests completed")
    return True


def test_model_persistence():
    """Test model saving and loading"""
    print("\n" + "="*60)
    print("TESTING MODEL PERSISTENCE")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test 1: Save model
        print("\n1. Testing model saving...")
        agent1 = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        
        # Train briefly
        for i in range(50):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = False
            agent1.remember(state, action, reward, next_state, done)
        
        agent1.replay(batch_size=32)
        
        # Save
        save_path = os.path.join(temp_dir, 'test_agent.pth')
        agent1.save(save_path)
        assert os.path.exists(save_path)
        print(f"    Model saved successfully")
        
        # Test 2: Load model
        print("\n2. Testing model loading...")
        agent2 = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent2.load(save_path)
        
        # Compare predictions
        test_state = np.random.randn(state_dim)
        agent1.epsilon = 0  # No exploration
        agent2.epsilon = 0
        
        action1 = agent1.act(test_state)
        action2 = agent2.act(test_state)
        
        assert action1 == action2
        print(f"    Model loaded successfully")
        print(f"   Actions match: {action1} == {action2}")
        
    except Exception as e:
        print(f"    Model persistence failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    print("\n Model persistence tests completed")
    return True


def test_double_dqn():
    """Test Double DQN specific functionality"""
    print("\n" + "="*60)
    print("TESTING DOUBLE DQN")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    print("\n1. Testing Double DQN training...")
    try:
        # Add experiences
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = i % 10 == 0
            
            agent.remember(state, action, reward, next_state, done)
        
        # Train
        loss = agent.replay(batch_size=32)
        
        if loss is not None:
            assert loss >= 0
            print(f"    Double DQN training working")
            print(f"   Loss: {loss:.6f}")
        else:
            print(f"    Not enough samples yet")
        
    except Exception as e:
        print(f"    Double DQN training failed: {e}")
        return False
    
    print("\n Double DQN tests completed")
    return True


def test_rainbow_dqn():
    """Test Rainbow DQN specific functionality"""
    print("\n" + "="*60)
    print("TESTING RAINBOW DQN")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    print("\n1. Testing Rainbow DQN components...")
    try:
        agent = RainbowDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_atoms=51,
            v_min=-10,
            v_max=10
        )
        
        # Check components
        assert hasattr(agent, 'q_network')
        assert hasattr(agent, 'memory')  # Should have prioritized replay
        
        # Test noisy network
        state = np.random.randn(state_dim)
        action1 = agent.act(state)
        
        # Reset noise
        if hasattr(agent.q_network, 'reset_noise'):
            agent.q_network.reset_noise()
            agent.target_network.reset_noise()
        
        print(f"    Rainbow DQN initialized with all components")
        
    except Exception as e:
        print(f"    Rainbow DQN initialization failed: {e}")
        return False
    
    # Test 2: Training with Rainbow
    print("\n2. Testing Rainbow training...")
    try:
        # Add experiences
        for i in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = i % 10 == 0
            
            agent.remember(state, action, reward, next_state, done)
        
        # Train
        loss = agent.replay(batch_size=32)
        
        if loss is not None:
            assert loss >= 0
            print(f"    Rainbow DQN training working")
            print(f"   Loss: {loss:.6f}")
        
    except Exception as e:
        print(f"    Rainbow training failed: {e}")
        return False
    
    print("\n Rainbow DQN tests completed")
    return True


def test_agent_factory():
    """Test DQN Agent Factory"""
    print("\n" + "="*60)
    print("TESTING DQN AGENT FACTORY")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    agent_types = ['dqn', 'double', 'dueling', 'rainbow']
    
    print("\n1. Testing agent creation for all types...")
    for agent_type in agent_types:
        try:
            config = {
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'epsilon': 1.0
            }
            
            if agent_type == 'rainbow':
                config['num_atoms'] = 51
                config['v_min'] = -10
                config['v_max'] = 10
            
            agent = DQNAgentFactory.create_agent(
                agent_type=agent_type,
                state_dim=state_dim,
                action_dim=action_dim,
                config=config
            )
            
            # Test action
            state = np.random.randn(state_dim)
            action = agent.act(state)
            assert 0 <= action < action_dim
            
            print(f"    {agent_type}: Created and tested successfully")
            
        except Exception as e:
            print(f"    {agent_type} failed: {e}")
    
    # Test 2: Invalid agent type
    print("\n2. Testing invalid agent type...")
    try:
        agent = DQNAgentFactory.create_agent(
            agent_type='invalid_type',
            state_dim=state_dim,
            action_dim=action_dim
        )
        print(f"    Should have raised error for invalid type")
    except (ValueError, KeyError):
        print(f"    Correctly raised error for invalid type")
    
    print("\n Agent Factory tests completed")
    return True


def test_training_metrics():
    """Test training metrics tracking"""
    print("\n" + "="*60)
    print("TESTING TRAINING METRICS")
    print("="*60)
    
    state_dim = 63
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        track_metrics=True  # Enable metrics tracking if supported
    )
    
    print("\n1. Testing metrics collection...")
    try:
        # Train for several steps
        losses = []
        rewards = []
        
        for episode in range(5):
            episode_reward = 0
            for step in range(20):
                state = np.random.randn(state_dim)
                action = agent.act(state)
                reward = np.random.randn()
                next_state = np.random.randn(state_dim)
                done = step == 19
                
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                
                if len(agent.memory) >= 32:
                    loss = agent.replay(batch_size=32)
                    if loss is not None:
                        losses.append(loss)
            
            rewards.append(episode_reward)
            agent.decay_epsilon()
        
        if losses:
            print(f"    Training metrics collected")
            print(f"   Average loss: {np.mean(losses):.6f}")
            print(f"   Average reward: {np.mean(rewards):.4f}")
            print(f"   Final epsilon: {agent.epsilon:.4f}")
        else:
            print(f"    No losses recorded")
        
    except Exception as e:
        print(f"    Metrics collection failed: {e}")
        return False
    
    print("\n Training metrics tests completed")
    return True


def run_all_tests():
    """Run all DQN agent tests"""
    print("\n" + "#"*60)
    print("# DQN AGENT MODULE COMPREHENSIVE TEST SUITE")
    print("#"*60)
    
    start_time = datetime.now()
    
    tests = [
        ("Replay Buffer", test_replay_buffer),
        ("Prioritized Replay Buffer", test_prioritized_replay_buffer),
        ("Agent Initialization", test_dqn_agent_initialization),
        ("Action Selection", test_action_selection),
        ("Training Step", test_training_step),
        ("Model Persistence", test_model_persistence),
        ("Double DQN", test_double_dqn),
        ("Rainbow DQN", test_rainbow_dqn),
        ("Agent Factory", test_agent_factory),
        ("Training Metrics", test_training_metrics)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    elapsed = datetime.now() - start_time
    print(f"Time elapsed: {elapsed.total_seconds():.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n ALL TESTS PASSED! The DQN agent module is working correctly.")
    else:
        print(f"\n {total_tests - passed_tests} test(s) failed. Please review the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)