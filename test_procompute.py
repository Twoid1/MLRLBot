"""
Test pre-computation speedup
"""

import pandas as pd
import numpy as np
import time
from src.environment.trading_env import TradingEnvironment

# Create sample data
print("Creating test data...")
dates = pd.date_range(start='2024-01-01', periods=10000, freq='1h')
test_data = pd.DataFrame({
    'open': np.random.uniform(40000, 45000, len(dates)),
    'high': np.random.uniform(45000, 46000, len(dates)),
    'low': np.random.uniform(39000, 40000, len(dates)),
    'close': np.random.uniform(40000, 45000, len(dates)),
    'volume': np.random.uniform(100, 1000, len(dates))
}, index=dates)

# Create fake features
features = pd.DataFrame(
    np.random.randn(len(dates), 50),
    index=dates
)

print("\n" + "="*60)
print("SPEED TEST: Pre-computation vs On-the-fly")
print("="*60)

# Test WITHOUT pre-computation
print("\n1. Testing WITHOUT pre-computation...")
start = time.time()
env_slow = TradingEnvironment(
    df=test_data,
    features_df=features,
    precompute_observations=False  # Disabled
)
init_time_slow = time.time() - start

print(f"   Initialization time: {init_time_slow:.3f}s")

# Run 100 steps
print("   Running 100 steps...")
start = time.time()
state = env_slow.reset()
for i in range(100):
    action = np.random.randint(0, 3)
    state, reward, done, truncated, info = env_slow.step(action)
    if done:
        break
step_time_slow = time.time() - start

print(f"   100 steps took: {step_time_slow:.3f}s ({step_time_slow/100*1000:.1f}ms per step)")

# Test WITH pre-computation
print("\n2. Testing WITH pre-computation...")
start = time.time()
env_fast = TradingEnvironment(
    df=test_data,
    features_df=features,
    precompute_observations=True  # Enabled!
)
init_time_fast = time.time() - start

print(f"   Initialization time: {init_time_fast:.3f}s")

# Run 100 steps
print("   Running 100 steps...")
start = time.time()
state = env_fast.reset()
for i in range(100):
    action = np.random.randint(0, 3)
    state, reward, done, truncated, info = env_fast.step(action)
    if done:
        break
step_time_fast = time.time() - start

print(f"   100 steps took: {step_time_fast:.3f}s ({step_time_fast/100*1000:.1f}ms per step)")

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Initialization:")
print(f"  Without pre-compute: {init_time_slow:.3f}s")
print(f"  With pre-compute:    {init_time_fast:.3f}s")
print(f"  Overhead:            +{init_time_fast - init_time_slow:.3f}s (ONE TIME)")

print(f"\nStep execution:")
print(f"  Without pre-compute: {step_time_slow/100*1000:.1f}ms per step")
print(f"  With pre-compute:    {step_time_fast/100*1000:.1f}ms per step")
print(f"  Speedup:             {step_time_slow/step_time_fast:.1f}x FASTER!")

print(f"\nFor 1000 episodes (200 steps each):")
print(f"  Without pre-compute: {step_time_slow*2000/60:.1f} minutes")
print(f"  With pre-compute:    {step_time_fast*2000/60:.1f} minutes")
print(f"  Time saved:          {(step_time_slow-step_time_fast)*2000/60:.1f} minutes")

print("\n Pre-computation test complete!")