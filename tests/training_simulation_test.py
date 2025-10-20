"""
Training Loop Simulation Test
Tests environment exactly as it's used during training

This will reveal if the bug is in:
- Training loop logic
- Episode reset
- Multi-environment handling
- Pre-computation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.environment.trading_env import TradingEnvironment, Actions


def create_sol_data():
    """Create data that matches your report (SOL around $3.53)"""
    dates = pd.date_range(start='2020-09-01', periods=1000, freq='1h')
    
    # Price around $3.53 like in your report
    base_price = 3.53
    prices = base_price + np.random.uniform(-0.2, 0.2, 1000)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002,
        'low': prices * 0.998,
        'close': prices,
        'volume': np.ones(1000) * 1000
    }, index=dates)
    
    return df


def simulate_single_episode(env, agent_actions=None, max_steps=900):
    """
    Simulate exactly how training runs one episode
    
    Returns episode stats and any warnings
    """
    obs = env.reset()
    
    episode_reward = 0
    steps = 0
    done = False
    
    max_position_value = 0
    position_values = []
    warnings = []
    
    while not done and steps < max_steps:
        # Agent selects action (we'll use random for testing)
        if agent_actions is not None and steps < len(agent_actions):
            action = agent_actions[steps]
        else:
            action = np.random.choice([0, 1, 2])  # Random action
        
        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        # Track position values
        if env.position != 0:  # If in a position
            position_value = env.position_size * env._get_current_price()
            position_values.append(position_value)
            max_position_value = max(max_position_value, position_value)
            
            # Check for unrealistic position
            if position_value > 20000:
                warnings.append({
                    'step': steps,
                    'position_size': env.position_size,
                    'position_value': position_value,
                    'balance': env.balance,
                    'message': f'Unrealistic position: {env.position_size:.2f} coins = ${position_value:,.2f}'
                })
    
    # Get final stats
    final_portfolio = env._get_portfolio_value()
    
    # Analyze trades
    trades_with_pnl = [t for t in env.trades if t.get('pnl') is not None]
    
    if trades_with_pnl:
        max_trade_pnl = max(t['pnl'] for t in trades_with_pnl)
        min_trade_pnl = min(t['pnl'] for t in trades_with_pnl)
        
        # Check for impossible trades
        for trade in trades_with_pnl:
            if abs(trade['pnl']) > 10000:  # More than initial balance
                warnings.append({
                    'step': trade['step'],
                    'trade': trade,
                    'message': f"Impossible trade: ${trade['pnl']:,.2f} PnL with $10k balance"
                })
    else:
        max_trade_pnl = 0
        min_trade_pnl = 0
    
    return {
        'episode_reward': episode_reward,
        'final_portfolio': final_portfolio,
        'steps': steps,
        'num_trades': len(trades_with_pnl),
        'max_position_value': max_position_value,
        'max_trade_pnl': max_trade_pnl,
        'min_trade_pnl': min_trade_pnl,
        'warnings': warnings,
        'position_values': position_values
    }


def test_multiple_episodes():
    """
    Test multiple episodes like in training
    This is where bugs might appear!
    """
    print("="*80)
    print(" TESTING MULTIPLE EPISODES (LIKE TRAINING)")
    print("="*80)
    print("\nThis simulates exactly how your training works:")
    print("  - Multiple episodes in sequence")
    print("  - Random actions (like RL agent)")
    print("  - Reset between episodes")
    print("  - 900 steps per episode\n")
    
    df = create_sol_data()
    
    # Create environment ONCE (like in training)
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,
        precompute_observations=False  # Test without pre-computation first
    )
    
    print("Running 10 episodes...\n")
    
    all_warnings = []
    episode_stats = []
    
    for episode in range(10):
        print(f"Episode {episode+1}:", end=" ")
        
        stats = simulate_single_episode(env, max_steps=900)
        episode_stats.append(stats)
        
        # Report episode
        print(f"Reward={stats['episode_reward']:+.2f}, "
              f"Portfolio=${stats['final_portfolio']:,.2f}, "
              f"MaxPos=${stats['max_position_value']:,.2f}, "
              f"Trades={stats['num_trades']}")
        
        # Check for warnings
        if stats['warnings']:
            print(f"    {len(stats['warnings'])} warnings!")
            all_warnings.extend(stats['warnings'])
            for w in stats['warnings']:
                print(f"     {w['message']}")
    
    print("\n" + "="*80)
    print("MULTI-EPISODE ANALYSIS")
    print("="*80)
    
    # Analyze position sizes across episodes
    all_position_values = []
    for stats in episode_stats:
        all_position_values.extend(stats['position_values'])
    
    if all_position_values:
        max_pos = max(all_position_values)
        avg_pos = np.mean(all_position_values)
        
        print(f"\nPosition Value Statistics:")
        print(f"  Average: ${avg_pos:,.2f}")
        print(f"  Maximum: ${max_pos:,.2f}")
        print(f"  95th percentile: ${np.percentile(all_position_values, 95):,.2f}")
        
        if max_pos > 20000:
            print(f"\n   FAIL: Max position ${max_pos:,.2f} exceeds $20,000!")
            return False
        else:
            print(f"   PASS: All positions within limits")
    
    # Analyze trades
    all_max_pnls = [s['max_trade_pnl'] for s in episode_stats]
    max_pnl_overall = max(all_max_pnls)
    
    print(f"\nTrade PnL Statistics:")
    print(f"  Max profit in any trade: ${max_pnl_overall:,.2f}")
    
    if max_pnl_overall > 5000:
        print(f"   FAIL: Max profit ${max_pnl_overall:,.2f} is unrealistic!")
        return False
    else:
        print(f"   PASS: All trade profits realistic")
    
    # Check for warnings
    if all_warnings:
        print(f"\n  {len(all_warnings)} WARNINGS DETECTED:")
        for w in all_warnings[:5]:  # Show first 5
            print(f"  - {w['message']}")
        return False
    else:
        print(f"\n NO WARNINGS: All episodes ran cleanly")
    
    return True


def test_with_precomputation():
    """Test with pre-computation enabled (like in training)"""
    print("\n" + "="*80)
    print(" TESTING WITH PRE-COMPUTATION ENABLED")
    print("="*80)
    print("\nYour training uses precompute_observations=True")
    print("Let's test if that causes issues...\n")
    
    df = create_sol_data()
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,
        precompute_observations=True  # ← ENABLED (like in training)
    )
    
    print("Running 5 episodes with pre-computation...\n")
    
    for episode in range(5):
        print(f"Episode {episode+1}:", end=" ")
        
        stats = simulate_single_episode(env, max_steps=900)
        
        print(f"Reward={stats['episode_reward']:+.2f}, "
              f"Portfolio=${stats['final_portfolio']:,.2f}, "
              f"MaxPos=${stats['max_position_value']:,.2f}")
        
        if stats['warnings']:
            print(f"   WARNINGS with pre-computation!")
            for w in stats['warnings']:
                print(f"     {w['message']}")
            return False
        
        if stats['max_position_value'] > 20000:
            print(f"   FAIL: Position grew to ${stats['max_position_value']:,.2f}")
            return False
    
    print("\n PASS: Pre-computation doesn't cause issues")
    return True


def test_environment_reuse():
    """Test if reusing environment causes state leakage"""
    print("\n" + "="*80)
    print(" TESTING ENVIRONMENT REUSE (STATE LEAKAGE)")
    print("="*80)
    print("\nTesting if state leaks between episodes...\n")
    
    df = create_sol_data()
    env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
    
    # Run 3 episodes and check if initial state is consistent
    initial_balances = []
    initial_positions = []
    
    for episode in range(3):
        obs = env.reset()
        
        initial_balances.append(env.balance)
        initial_positions.append(env.position_size)
        
        print(f"Episode {episode+1} after reset:")
        print(f"  Balance: ${env.balance:,.2f}")
        print(f"  Position: {env.position.name}")
        print(f"  Position size: {env.position_size:.2f}")
        
        # Run episode
        for _ in range(100):
            action = np.random.choice([0, 1, 2])
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        print(f"  After episode: Balance=${env.balance:,.2f}, "
              f"Position={env.position_size:.2f}")
    
    print()
    
    # Check consistency
    if all(b == 10000 for b in initial_balances):
        print(" PASS: Balance resets to $10,000 every episode")
    else:
        print(f" FAIL: Balance not consistent: {initial_balances}")
        return False
    
    if all(p == 0 for p in initial_positions):
        print(" PASS: Position resets to 0 every episode")
    else:
        print(f" FAIL: Position not consistent: {initial_positions}")
        return False
    
    return True


def test_with_real_agent_behavior():
    """Simulate more realistic agent behavior (less random)"""
    print("\n" + "="*80)
    print(" TESTING WITH REALISTIC AGENT BEHAVIOR")
    print("="*80)
    print("\nSimulating how a trained agent actually trades...\n")
    
    df = create_sol_data()
    env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=True)
    
    # Simulate agent that trades frequently (like your report: 144 trades/episode)
    def agent_policy(step):
        """Simulate agent that buys/sells frequently"""
        if step % 10 < 3:
            return 1  # BUY
        elif step % 10 < 6:
            return 0  # HOLD
        else:
            return 2  # SELL
    
    print("Running episode with frequent trading...\n")
    
    obs = env.reset()
    episode_reward = 0
    steps = 0
    trades_executed = 0
    
    max_position_seen = 0
    position_history = []
    
    while steps < 900:
        action = agent_policy(steps)
        
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        if env.position != 0:
            pos_value = env.position_size * env._get_current_price()
            max_position_seen = max(max_position_seen, pos_value)
            position_history.append({
                'step': steps,
                'position_size': env.position_size,
                'position_value': pos_value,
                'balance': env.balance
            })
        
        if len(env.trades) > trades_executed:
            trades_executed = len(env.trades)
        
        if done:
            break
    
    print(f"Episode completed:")
    print(f"  Steps: {steps}")
    print(f"  Trades: {trades_executed}")
    print(f"  Final portfolio: ${env._get_portfolio_value():,.2f}")
    print(f"  Max position: ${max_position_seen:,.2f}")
    
    # Look for position growth pattern
    if len(position_history) > 10:
        first_10_avg = np.mean([p['position_value'] for p in position_history[:10]])
        last_10_avg = np.mean([p['position_value'] for p in position_history[-10:]])
        
        print(f"\n  First 10 positions avg: ${first_10_avg:,.2f}")
        print(f"  Last 10 positions avg: ${last_10_avg:,.2f}")
        
        if last_10_avg > first_10_avg * 2:
            print(f"   FAIL: Positions grew {last_10_avg/first_10_avg:.1f}x during episode!")
            print(f"  This suggests accumulation!")
            
            # Show details
            print(f"\n  Position growth over time:")
            for i in [0, len(position_history)//4, len(position_history)//2, 
                     3*len(position_history)//4, len(position_history)-1]:
                p = position_history[i]
                print(f"    Step {p['step']:3d}: {p['position_size']:8.2f} coins = ${p['position_value']:10,.2f}")
            
            return False
        else:
            print(f"   PASS: Position sizes remained stable")
    
    if max_position_seen > 20000:
        print(f"\n   FAIL: Max position ${max_position_seen:,.2f} too high!")
        return False
    
    return True


def main():
    """Run all training simulation tests"""
    print("\n" + "="*80)
    print(" TRAINING LOOP SIMULATION TEST SUITE")
    print("="*80)
    print("\nThese tests simulate exactly how your training works")
    print("to find bugs that only appear during training.\n")
    print("="*80)
    
    results = []
    
    # Test 1: Multiple episodes
    try:
        result1 = test_multiple_episodes()
        results.append(("Multiple Episodes", result1))
    except Exception as e:
        print(f"\n Multiple episodes test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Multiple Episodes", False))
    
    # Test 2: Pre-computation
    try:
        result2 = test_with_precomputation()
        results.append(("Pre-computation", result2))
    except Exception as e:
        print(f"\n Pre-computation test crashed: {e}")
        results.append(("Pre-computation", False))
    
    # Test 3: Environment reuse
    try:
        result3 = test_environment_reuse()
        results.append(("Environment Reuse", result3))
    except Exception as e:
        print(f"\n Environment reuse test crashed: {e}")
        results.append(("Environment Reuse", result3))
    
    # Test 4: Realistic agent
    try:
        result4 = test_with_real_agent_behavior()
        results.append(("Realistic Agent", result4))
    except Exception as e:
        print(f"\n Realistic agent test crashed: {e}")
        results.append(("Realistic Agent", False))
    
    # Summary
    print("\n" + "="*80)
    print(" TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status} | {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("="*80)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n ALL TRAINING SIMULATION TESTS PASSED!")
        print("\nThis means:")
        print("   Environment works correctly in training scenarios")
        print("   No state leakage between episodes")
        print("   Pre-computation doesn't cause issues")
        print("   Frequent trading doesn't cause accumulation")
        print("\nThe bug in your report must be elsewhere:")
        print("  - Check your actual training script")
        print("  - Check if you're using multiple environments")
        print("  - Check if features are calculated correctly")
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        print("\nBug found in training simulation!")
        print("Review the failed test output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)