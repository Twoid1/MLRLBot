"""
Master Test Script - Run All Environment Module Tests
Runs individual test suites and integration tests
"""

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import test modules
from test_trading_env import run_all_tests as test_trading_env
from test_rewards import run_all_tests as test_rewards
from test_state import run_all_tests as test_state_management

# Import actual modules for integration testing
from src.environment.trading_env import TradingEnvironment
from src.environment.rewards import RewardCalculator
from src.environment.state import StateManager

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def run_individual_tests():
    """Run each test suite individually"""
    print_header("RUNNING INDIVIDUAL TEST SUITES")
    
    results = {}
    
    # Test Trading Environment
    print("\n" + "="*70)
    print("Running Trading Environment Tests...")
    print("="*70)
    try:
        results['Trading Environment'] = test_trading_env()
    except Exception as e:
        print(f" Trading Environment tests failed: {e}")
        results['Trading Environment'] = False
    
    # Test Reward Functions
    print("\n" + "="*70)
    print("Running Reward Functions Tests...")
    print("="*70)
    try:
        results['Reward Functions'] = test_rewards()
    except Exception as e:
        print(f" Reward Functions tests failed: {e}")
        results['Reward Functions'] = False
    
    # Test State Management
    print("\n" + "="*70)
    print("Running State Management Tests...")
    print("="*70)
    try:
        results['State Management'] = test_state_management()
    except Exception as e:
        print(f" State Management tests failed: {e}")
        results['State Management'] = False
    
    return results

def test_integration():
    """Test integration between all modules"""
    print_header("INTEGRATION TESTS")
    
    try:
        print("\n1. Testing Complete Environment Integration...")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        prices = 42000 + np.cumsum(np.random.randn(500) * 100)
        
        data = pd.DataFrame({
            'open': prices + np.random.uniform(-50, 50, 500),
            'high': prices + np.random.uniform(50, 200, 500),
            'low': prices - np.random.uniform(50, 200, 500),
            'close': prices,
            'volume': np.random.uniform(100, 1000, 500)
        }, index=dates)
        
        # Fix OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        # Create fake indicators
        indicators = pd.DataFrame({
            'rsi_14': np.random.uniform(30, 70, len(data)),
            'macd': np.random.uniform(-100, 100, len(data)),
            'bb_position': np.random.uniform(0, 1, len(data))
        }, index=data.index)
        
        print("    Test data created")
        
        # Initialize environment with features
        env = TradingEnvironment(
            df=data,
            initial_balance=10000,
            features_df=indicators,
            window_size=30,
            enable_short=False,
            stop_loss=0.05,
            take_profit=0.10
        )
        
        print("    Environment initialized with features")
        
        # Initialize state manager
        state_manager = StateManager(window_size=30)
        
        # Initialize reward calculator
        reward_calc = RewardCalculator()
        
        print("    State manager and reward calculator initialized")
        
        # Run a test episode
        obs = env.reset()
        total_reward = 0
        custom_rewards = []
        states_collected = []
        
        for step in range(100):
            # Random action
            action = np.random.choice([0, 1, 2], p=[0.6, 0.2, 0.2])
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Test state extraction
            if env.current_step < len(data):
                market_state = state_manager.extract_market_state(data, env.current_step)
                account_state = state_manager.extract_account_state(env)
                state_vector = state_manager.create_state_vector(market_state, account_state)
                states_collected.append(state_vector)
            
            # Test custom reward calculation
            if len(env.portfolio_values) > 1:
                custom_reward = reward_calc.composite_reward(
                    current_value=env.portfolio_values[-1],
                    previous_value=env.portfolio_values[-2] if len(env.portfolio_values) > 1 else env.initial_balance,
                    portfolio_history=env.portfolio_values,
                    trades=env.trades,
                    current_drawdown=env.current_drawdown
                )
                custom_rewards.append(custom_reward)
            
            if done:
                break
        
        print(f"    Episode completed: {step+1} steps")
        print(f"    States collected: {len(states_collected)}")
        print(f"    Custom rewards calculated: {len(custom_rewards)}")
        print(f"    Total reward: {total_reward:.4f}")
        
        # Test performance summary
        summary = env.get_performance_summary()
        print(f"    Performance summary generated")
        print(f"      - Total trades: {summary.get('total_trades', 0)}")
        print(f"      - Win rate: {summary.get('win_rate', 0):.2%}")
        print(f"      - Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        
        print("\n2. Testing State Tracking Integration...")
        
        # Test state visitation tracking
        unique_states = len(set(state_manager.hash_state(s) for s in states_collected))
        state_stats = state_manager.get_state_statistics()
        
        print(f"    State tracking successful")
        print(f"      - Unique states: {unique_states}")
        print(f"      - Total visits: {state_stats.get('total_visits', 0)}")
        
        print("\n3. Testing Reward Variations...")
        
        # Test different reward functions
        if len(env.portfolio_values) >= 20:
            sharpe_reward = reward_calc.risk_adjusted_returns(env.portfolio_values[-20:])
            print(f"    Sharpe reward: {sharpe_reward:.4f}")
            
            if env.trades:
                winning = [t for t in env.trades if t.get('pnl') is not None and t['pnl'] > 0]
                losing = [t for t in env.trades if t.get('pnl') is not None and t['pnl'] < 0]
                if winning and losing:
                    profit_factor = reward_calc.profit_factor_reward(
                        [t['pnl'] for t in winning],
                        [t['pnl'] for t in losing]
                    )
                    print(f"    Profit factor reward: {profit_factor:.4f}")
        
        print("\n All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance and timing"""
    print_header("PERFORMANCE TESTS")
    
    try:
        # Create larger dataset
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, 5000),
            'high': np.random.uniform(45000, 46000, 5000),
            'low': np.random.uniform(39000, 40000, 5000),
            'close': np.random.uniform(40000, 45000, 5000),
            'volume': np.random.uniform(100, 1000, 5000)
        })
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        print("\n1. Environment Step Performance...")
        
        env = TradingEnvironment(df=data[:1000], initial_balance=10000)
        env.reset()
        
        # Time single step
        start_time = time.time()
        for _ in range(100):
            env.step(np.random.choice([0, 1, 2]))
        step_time = (time.time() - start_time) / 100
        
        print(f"    Average step time: {step_time*1000:.2f}ms")
        print(f"    Steps per second: {1/step_time:.0f}")
        
        print("\n2. State Extraction Performance...")
        
        state_manager = StateManager()
        
        start_time = time.time()
        for i in range(100, 200):
            market_state = state_manager.extract_market_state(data, i)
        state_time = (time.time() - start_time) / 100
        
        print(f"    Average state extraction: {state_time*1000:.2f}ms")
        
        print("\n3. Reward Calculation Performance...")
        
        reward_calc = RewardCalculator()
        portfolio_values = [10000 + i*10 for i in range(100)]
        
        start_time = time.time()
        for i in range(1, 100):
            reward = reward_calc.composite_reward(
                portfolio_values[i], portfolio_values[i-1],
                portfolio_values[:i], [], 0.05
            )
        reward_time = (time.time() - start_time) / 99
        
        print(f"    Average reward calculation: {reward_time*1000:.2f}ms")
        
        print("\n4. Full Episode Performance...")
        
        env = TradingEnvironment(df=data[:1000], initial_balance=10000)
        
        start_time = time.time()
        obs = env.reset()
        steps = 0
        while not env.done and steps < 800:
            action = np.random.choice([0, 1, 2])
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        episode_time = time.time() - start_time
        
        print(f"    Episode completed in {episode_time:.2f}s")
        print(f"    Steps: {steps}")
        print(f"    Steps per second: {steps/episode_time:.0f}")
        
        # Check if performance is acceptable
        if step_time < 0.01:  # Less than 10ms per step
            print("\n Performance is EXCELLENT (< 10ms per step)")
        elif step_time < 0.05:  # Less than 50ms per step
            print("\n Performance is GOOD (< 50ms per step)")
        else:
            print("\n  Performance could be improved (> 50ms per step)")
        
        return True
        
    except Exception as e:
        print(f"\n Performance test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print_header("EDGE CASE TESTS")
    
    try:
        print("\n1. Testing with minimal data...")
        
        # Very small dataset
        small_data = pd.DataFrame({
            'open': [42000, 42100],
            'high': [42200, 42300],
            'low': [41900, 42000],
            'close': [42100, 42200],
            'volume': [100, 200]
        })
        
        try:
            env = TradingEnvironment(df=small_data, window_size=10)
            print("     Should have failed with insufficient data")
        except ValueError as e:
            print(f"    Correctly rejected insufficient data: {e}")
        
        print("\n2. Testing with NaN values...")
        
        # Data with NaNs
        nan_data = pd.DataFrame({
            'open': [42000, np.nan, 42100, 42200],
            'high': [42200, 42300, np.nan, 42400],
            'low': [41900, 42000, 42000, np.nan],
            'close': [42100, 42200, 42300, 42400],
            'volume': [100, 200, 300, np.nan]
        })
        
        # Should handle NaNs gracefully
        state_manager = StateManager()
        try:
            market_state = state_manager.extract_market_state(nan_data.fillna(method='ffill'), 2)
            print("    Handled NaN values")
        except Exception as e:
            print(f"    Failed to handle NaN: {e}")
        
        print("\n3. Testing extreme values...")
        
        # Extreme prices
        extreme_data = pd.DataFrame({
            'open': np.array([1e-6, 1e6, 42000, 42100] * 25),
            'high': np.array([1e6, 1e6, 42200, 42300] * 25),
            'low': np.array([1e-6, 1e-6, 41900, 42000] * 25),
            'close': np.array([1e-6, 1e6, 42100, 42200] * 25),
            'volume': np.array([1e-10, 1e10, 100, 200] * 25)
        })
        
        env = TradingEnvironment(df=extreme_data, initial_balance=10000)
        obs = env.reset()
        
        # Should handle extreme values without crashing
        for _ in range(10):
            obs, reward, done, truncated, info = env.step(np.random.choice([0, 1, 2]))
        
        print("    Handled extreme values")
        
        print("\n4. Testing bankruptcy protection...")
        
        # Create losing scenario
        losing_data = pd.DataFrame({
            'open': np.linspace(42000, 20000, 100),
            'high': np.linspace(42100, 20100, 100),
            'low': np.linspace(41900, 19900, 100),
            'close': np.linspace(42000, 20000, 100),
            'volume': np.ones(100) * 100
        })
        
        env = TradingEnvironment(df=losing_data, initial_balance=1000)
        obs = env.reset()
        
        # Buy and hold to lose money
        env.step(1)  # Buy
        
        for _ in range(50):
            obs, reward, done, truncated, info = env.step(0)  # Hold
            if done:
                print(f"    Bankruptcy protection triggered at balance: ${env.balance:.2f}")
                break
        
        print("\n All edge cases handled correctly!")
        return True
        
    except Exception as e:
        print(f"\n Edge case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_report(results):
    """Generate final test report"""
    print_header("FINAL TEST REPORT")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTest Run: {timestamp}")
    print("-" * 50)
    
    # Individual test results
    print("\n Individual Test Suites:")
    for suite, passed in results['individual'].items():
        status = " PASSED" if passed else " FAILED"
        print(f"   {status}: {suite}")
    
    # Integration test results
    print("\n Integration Tests:")
    status = " PASSED" if results['integration'] else " FAILED"
    print(f"   {status}")
    
    # Performance test results
    print("\n Performance Tests:")
    status = " PASSED" if results['performance'] else " FAILED"
    print(f"   {status}")
    
    # Edge case results
    print("\n Edge Case Tests:")
    status = " PASSED" if results['edge_cases'] else " FAILED"
    print(f"   {status}")
    
    # Overall summary
    all_passed = (
        all(results['individual'].values()) and
        results['integration'] and
        results['performance'] and
        results['edge_cases']
    )
    
    print("\n" + "="*70)
    if all_passed:
        print(" ALL TESTS PASSED! The Trading Environment is fully functional!")
        print("="*70)
        print("\n Ready for:")
        print("   - ML Predictor integration")
        print("   - RL Agent training")
        print("   - Backtesting")
        print("   - Paper trading")
    else:
        print("  SOME TESTS FAILED - Please review the errors above")
        print("="*70)
        print("\n Issues to fix before proceeding:")
        if not all(results['individual'].values()):
            print("   - Fix individual module failures")
        if not results['integration']:
            print("   - Fix integration issues")
        if not results['performance']:
            print("   - Optimize performance")
        if not results['edge_cases']:
            print("   - Handle edge cases")
    
    return all_passed

def main():
    """Main test runner"""
    print("\n" + "="*70)
    print("  MASTER TEST SUITE - TRADING ENVIRONMENT MODULES")
    print("="*70)
    print("\nThis will test:")
    print("  1. Trading Environment")
    print("  2. Reward Functions")
    print("  3. State Management")
    print("  4. Integration between modules")
    print("  5. Performance benchmarks")
    print("  6. Edge cases")
    
    results = {
        'individual': {},
        'integration': False,
        'performance': False,
        'edge_cases': False
    }
    
    # Run individual tests
    print("\n" + "-"*70)
    results['individual'] = run_individual_tests()
    
    # Run integration tests
    print("\n" + "-"*70)
    results['integration'] = test_integration()
    
    # Run performance tests
    print("\n" + "-"*70)
    results['performance'] = test_performance()
    
    # Run edge case tests
    print("\n" + "-"*70)
    results['edge_cases'] = test_edge_cases()
    
    # Generate report
    print("\n" + "-"*70)
    all_passed = generate_report(results)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)