"""
Diagnostic Script to Debug RL Training Issues
Identifies why episodes are ending immediately and why no trades are executed
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from src.environment.trading_env import TradingEnvironment
from src.models.dqn_agent import DQNAgent


def diagnose_environment_issue():
    """Diagnose why the environment is not working correctly"""
    
    print("="*80)
    print("TRADING ENVIRONMENT DIAGNOSTIC")
    print("="*80)
    
    # Create synthetic data
    print("\n1. Creating test data...")
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    
    # Create realistic price data
    price_start = 45000
    prices = []
    current_price = price_start
    
    for i in range(1000):
        change = np.random.randn() * 100  # Random walk
        current_price += change
        prices.append(current_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)
    
    print(f" Created {len(df)} rows of test data")
    print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Create environment
    print("\n2. Creating trading environment...")
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,
        window_size=50,
        precompute_observations=False  # Disable for debugging
    )
    print(f" Environment created")
    print(f"  Data length: {len(env.prices)}")
    print(f"  Window size: {env.window_size}")
    print(f"  Expected max steps: {len(env.prices) - env.window_size - 1}")
    
    # Test episode length
    print("\n3. Testing episode length...")
    obs = env.reset()
    print(f" Environment reset")
    print(f"  Initial step: {env.current_step}")
    print(f"  Initial balance: ${env.balance:,.2f}")
    print(f"  Initial portfolio: ${env._get_portfolio_value():,.2f}")
    print(f"  Done flag: {env.done}")
    print(f"  Observation shape: {obs.shape}")
    
    # Run a few steps manually
    print("\n4. Running manual steps...")
    steps_completed = 0
    actions_taken = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    
    for i in range(10):
        # Random action
        action = np.random.choice([0, 1, 2])
        action_name = ['HOLD', 'BUY', 'SELL'][action]
        actions_taken[action_name] += 1
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"\n  Step {i+1}:")
        print(f"    Action: {action_name}")
        print(f"    Current step: {env.current_step}")
        print(f"    Balance: ${env.balance:,.2f}")
        print(f"    Position: {env.position.name}")
        print(f"    Portfolio: ${info['portfolio_value']:,.2f}")
        print(f"    Reward: {reward:.6f}")
        print(f"    Done: {done}, Truncated: {truncated}")
        print(f"    Trades executed: {len(env.trades)}")
        
        steps_completed += 1
        
        if done or truncated:
            print(f"\n   Episode ended after {steps_completed} steps!")
            print(f"     Reason: {'Done' if done else 'Truncated'}")
            break
    
    print(f"\n  Action distribution: {actions_taken}")
    print(f"  Total trades recorded: {len(env.trades)}")
    
    # Check why episode might be ending early
    print("\n5. Checking episode termination conditions...")
    print(f"  Current step ({env.current_step}) >= Data length - 1 ({len(env.prices) - 1}): {env.current_step >= len(env.prices) - 1}")
    print(f"  Balance ({env.balance:.2f}) < 10% of initial ({env.initial_balance * 0.1:.2f}): {env.balance < env.initial_balance * 0.1}")
    print(f"  Max drawdown ({env.max_drawdown:.2%}) > 50%: {env.max_drawdown > 0.5}")
    
    # Test agent interaction
    print("\n6. Testing with DQN Agent...")
    
    state_dim = env.observation_space_shape[0]
    action_dim = 3
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    print(f" Agent created")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Initial epsilon: {agent.epsilon}")
    
    # Run one episode with agent
    print("\n7. Running full episode with agent...")
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    agent_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    
    while not env.done and episode_steps < 100:  # Limit to 100 steps for diagnostic
        action = agent.act(obs, training=True)
        agent_actions[['HOLD', 'BUY', 'SELL'][action]] += 1
        
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        
        if episode_steps % 20 == 0:
            print(f"  Step {episode_steps}: Portfolio=${info['portfolio_value']:,.2f}, Reward={reward:.6f}")
    
    print(f"\n Episode completed")
    print(f"  Steps: {episode_steps}")
    print(f"  Total reward: {episode_reward:.6f}")
    print(f"  Final portfolio: ${info['portfolio_value']:,.2f}")
    print(f"  Trades executed: {info['num_trades']}")
    print(f"  Agent actions: {agent_actions}")
    
    # Analyze why no trades
    print("\n8. DIAGNOSIS:")
    print("="*80)
    
    if episode_steps < 10:
        print(" PROBLEM: Episode ending too quickly!")
        print("   Possible causes:")
        print("   - Data too short")
        print("   - _check_done() being called incorrectly")
        print("   - current_step advancing too fast")
    
    if info['num_trades'] == 0:
        print(" PROBLEM: Agent not executing any trades!")
        print("   Possible causes:")
        print("   - Agent always choosing HOLD action")
        print("   - Position already open (can't buy)")
        print("   - No position to sell")
        print("   - Reward function not incentivizing trades")
        print("\n   Agent action distribution:", agent_actions)
        
        # Check if it's action selection issue
        if agent_actions['HOLD'] == episode_steps:
            print("\n    Agent is ONLY choosing HOLD!")
            print("      This is likely because:")
            print("      1. Q-values for all actions are similar")
            print("      2. Epsilon decay too fast")
            print("      3. Reward function doesn't encourage exploration")
        
        if agent_actions['BUY'] > 0 and info['num_trades'] == 0:
            print("\n    Agent is trying to BUY but trades not executing!")
            print("      This means:")
            print("      1. Position might already be open")
            print("      2. _execute_buy() has a logic error")
            print("      3. Balance is too low")
    
    # Print environment state for inspection
    print("\n9. Final Environment State:")
    print(f"  Current step: {env.current_step} / {len(env.prices)}")
    print(f"  Position: {env.position.name}")
    print(f"  Balance: ${env.balance:,.2f}")
    print(f"  Trades list length: {len(env.trades)}")
    print(f"  Done: {env.done}")
    print(f"  Truncated: {env.truncated}")
    
    if len(env.trades) > 0:
        print("\n  Recent trades:")
        for trade in env.trades[-5:]:
            print(f"    {trade}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    diagnose_environment_issue()