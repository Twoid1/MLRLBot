"""
Find the EXACT bottleneck causing slow training
This will show you where the 270 seconds per episode is being spent
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import time
from src.environment.trading_env import TradingEnvironment
from src.models.dqn_agent import DQNAgent, DQNConfig

print("="*80)
print("TRAINING BOTTLENECK ANALYSIS")
print("="*80)

# Create realistic data (similar size to your actual training)
print("\n1. Creating test data (similar to your training data)...")
n_samples = 35000  # ~5 years of hourly data
test_data = pd.DataFrame({
    'open': np.random.uniform(40000, 45000, n_samples),
    'high': np.random.uniform(45000, 46000, n_samples),
    'low': np.random.uniform(39000, 40000, n_samples),
    'close': np.random.uniform(40000, 45000, n_samples),
    'volume': np.random.uniform(100, 1000, n_samples)
})

features = pd.DataFrame(
    np.random.randn(n_samples, 50),
    columns=[f'feature_{i}' for i in range(50)]
)

print(f"   Data: {n_samples:,} samples")
print(f"   Features: {features.shape[1]} dimensions")

# Create environment WITH pre-computation
print("\n2. Creating environment...")
t_start = time.time()
env = TradingEnvironment(
    df=test_data,
    features_df=features,
    initial_balance=10000,
    precompute_observations=True
)
t_env = time.time() - t_start
print(f"    Environment created in {t_env:.2f}s")

# Create DQN agent
print("\n3. Creating DQN agent...")
t_start = time.time()
state_dim = env.observation_space_shape[0]
action_dim = env.action_space_n

config = DQNConfig(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=[256, 256, 128],
    batch_size=256,
    memory_size=50000,
    update_every=10,
    use_double_dqn=True,
    use_dueling_dqn=True,
    use_prioritized_replay=False
)

agent = DQNAgent(config=config)
t_agent = time.time() - t_start
print(f"    Agent created in {t_agent:.2f}s")
print(f"   Device: {agent.device}")
print(f"   Batch size: {config.batch_size}")
print(f"   Update every: {config.update_every} steps")

# Run ONE episode with detailed timing
print("\n4. Running ONE episode with detailed timing...")
print("="*80)

state = env.reset()
episode_reward = 0
steps = 0
done = False

# Timing trackers
times = {
    'act': [],
    'step': [],
    'remember': [],
    'replay': [],
    'other': []
}

episode_start = time.time()
replay_count = 0

print("   Running episode (will take up to 1000 steps)...")

while not done and steps < 1000:
    # Time: agent.act()
    t = time.time()
    action = agent.act(state, training=True)
    times['act'].append(time.time() - t)
    
    # Time: env.step()
    t = time.time()
    next_state, reward, done, truncated, info = env.step(action)
    times['step'].append(time.time() - t)
    
    # Time: agent.remember()
    t = time.time()
    agent.remember(state, action, reward, next_state, done)
    times['remember'].append(time.time() - t)
    
    # Time: agent.replay() (only when triggered)
    if len(agent.memory) > 1000 and steps % config.update_every == 0:
        t = time.time()
        loss = agent.replay(batch_size=config.batch_size)
        times['replay'].append(time.time() - t)
        replay_count += 1
    
    state = next_state
    episode_reward += reward
    steps += 1
    
    # Print progress
    if steps % 100 == 0:
        elapsed = time.time() - episode_start
        print(f"      Step {steps}: {elapsed:.1f}s elapsed, {steps/elapsed:.1f} steps/sec")

episode_time = time.time() - episode_start

print("\n" + "="*80)
print("DETAILED TIMING BREAKDOWN")
print("="*80)

# Calculate statistics
def stats(times_list):
    if not times_list:
        return 0, 0, 0, 0
    arr = np.array(times_list)
    return arr.sum(), arr.mean(), arr.min(), arr.max()

print(f"\nEpisode completed:")
print(f"  Total time: {episode_time:.2f}s")
print(f"  Steps: {steps}")
print(f"  Avg time per step: {episode_time/steps*1000:.1f}ms")
print(f"  Steps per second: {steps/episode_time:.2f}")

print(f"\n{'Operation':<20} {'Total (s)':<12} {'Calls':<8} {'Avg (ms)':<12} {'% of Time':<12}")
print("-"*80)

total_accounted = 0

# agent.act()
total, avg, min_t, max_t = stats(times['act'])
pct = (total / episode_time) * 100
total_accounted += total
print(f"{'agent.act()':<20} {total:<12.3f} {len(times['act']):<8} {avg*1000:<12.1f} {pct:<12.1f}%")

# env.step()
total, avg, min_t, max_t = stats(times['step'])
pct = (total / episode_time) * 100
total_accounted += total
print(f"{'env.step()':<20} {total:<12.3f} {len(times['step']):<8} {avg*1000:<12.1f} {pct:<12.1f}%")

# agent.remember()
total, avg, min_t, max_t = stats(times['remember'])
pct = (total / episode_time) * 100
total_accounted += total
print(f"{'agent.remember()':<20} {total:<12.3f} {len(times['remember']):<8} {avg*1000:<12.1f} {pct:<12.1f}%")

# agent.replay()
total, avg, min_t, max_t = stats(times['replay'])
pct = (total / episode_time) * 100
total_accounted += total
replay_info = f" ({replay_count} training updates)"
print(f"{'agent.replay()':<20} {total:<12.3f} {replay_count:<8} {avg*1000 if replay_count > 0 else 0:<12.1f} {pct:<12.1f}%{replay_info}")

# Other (overhead, logging, etc.)
other_time = episode_time - total_accounted
pct = (other_time / episode_time) * 100
print(f"{'Other (overhead)':<20} {other_time:<12.3f} {'-':<8} {'-':<12} {pct:<12.1f}%")

print("-"*80)
print(f"{'TOTAL':<20} {episode_time:<12.3f}")

# Identify the slowest operation
print("\n" + "="*80)
print("BOTTLENECK ANALYSIS")
print("="*80)

operations = {
    'agent.act()': sum(times['act']),
    'env.step()': sum(times['step']),
    'agent.remember()': sum(times['remember']),
    'agent.replay()': sum(times['replay']) if times['replay'] else 0,
    'Other': other_time
}

slowest = max(operations.items(), key=lambda x: x[1])

print(f"\n SLOWEST OPERATION: {slowest[0]}")
print(f"   Takes: {slowest[1]:.2f}s ({slowest[1]/episode_time*100:.1f}% of episode time)")

if slowest[0] == 'agent.replay()':
    print(f"\n DIAGNOSIS: Neural network training is the bottleneck")
    print(f"   Current batch size: {config.batch_size}")
    print(f"   Training frequency: every {config.update_every} steps")
    print(f"   Replays per episode: {replay_count}")
    print(f"   Time per replay: {slowest[1]/replay_count:.3f}s")
    print(f"\n   SOLUTIONS:")
    print(f"   1. Reduce batch_size: 256 → 64 (4x faster per replay)")
    print(f"   2. Increase update_every: {config.update_every} → 20 (2x fewer replays)")
    print(f"   3. Use GPU if available")
    print(f"   4. Simplify network: [256,256,128] → [128,128]")
    
elif slowest[0] == 'env.step()':
    print(f"\n DIAGNOSIS: Environment step is slow")
    print(f"   Pre-computation enabled: {env.precompute_observations}")
    print(f"   Pre-computed array exists: {env.precomputed_obs is not None}")
    if env.precompute_observations and env.precomputed_obs is not None:
        print(f"     Pre-computation enabled but step() still slow!")
        print(f"   Check: Are we still calling _calculate_observation()?")
    else:
        print(f"    Pre-computation NOT working!")
        print(f"   FIX: Enable precompute_observations=True")
    
elif slowest[0] == 'agent.act()':
    print(f"\n DIAGNOSIS: Action selection is slow")
    print(f"   Network forward pass taking too long")
    print(f"   Time per act: {slowest[1]/steps*1000:.2f}ms")
    print(f"\n   SOLUTIONS:")
    print(f"   1. Use GPU")
    print(f"   2. Simplify network architecture")
    print(f"   3. Batch inference (advanced)")

# Estimate full training time
print(f"\n" + "="*80)
print("PROJECTED TRAINING TIME")
print("="*80)

print(f"\nBased on this episode:")
print(f"  Time per episode: {episode_time:.1f}s")
print(f"  For 100 episodes: {episode_time*100/60:.1f} minutes")
print(f"  For 1000 episodes: {episode_time*1000/3600:.2f} hours")

if episode_time > 100:  # If > 100 seconds per episode
    print(f"\n  WARNING: Training will be VERY slow!")
    print(f"  This matches your reported 270s per episode")
    print(f"  Focus on fixing the '{slowest[0]}' bottleneck above")
elif episode_time > 10:
    print(f"\n  Training is moderate speed")
    print(f"  Could be optimized further")
else:
    print(f"\n Training speed is GOOD!")
    print(f"  If your actual training is slower, there may be other issues")

print()