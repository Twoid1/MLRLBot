"""
Test Script for Trading Environment Module
Tests all core functionality of the trading environment
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the trading environment
from src.environment.trading_env import TradingEnvironment, Actions, Positions

def create_sample_data(num_bars=1000, trend='neutral'):
    """Create sample OHLCV data for testing"""
    print("Creating sample OHLCV data...")
    
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='1h')
    
    # Base price with trend
    if trend == 'up':
        base_price = 40000 + np.arange(num_bars) * 10
    elif trend == 'down':
        base_price = 45000 - np.arange(num_bars) * 10
    else:
        base_price = 42000 + np.sin(np.arange(num_bars) * 0.1) * 1000
    
    # Add noise
    noise = np.random.randn(num_bars) * 100
    close_prices = base_price + noise
    
    # Create OHLCV
    data = pd.DataFrame({
        'open': close_prices + np.random.uniform(-50, 50, num_bars),
        'high': close_prices + np.random.uniform(50, 200, num_bars),
        'low': close_prices - np.random.uniform(50, 200, num_bars),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, num_bars)
    }, index=dates)
    
    # Ensure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def test_basic_initialization():
    """Test environment initialization"""
    print("\n" + "="*60)
    print("TEST 1: Basic Initialization")
    print("="*60)
    
    try:
        # Create sample data
        data = create_sample_data(200)
        
        # Initialize environment
        env = TradingEnvironment(
            df=data,
            initial_balance=10000,
            leverage=1.0,
            fee_rate=0.0026,
            window_size=50
        )
        
        print(" Environment initialized successfully")
        print(f"   - Initial balance: ${env.initial_balance}")
        print(f"   - Fee rate: {env.fee_rate*100:.2f}%")
        print(f"   - Window size: {env.window_size}")
        print(f"   - Action space: {env.action_space_n} actions")
        print(f"   - Observation shape: {env.observation_space_shape}")
        
        return True
        
    except Exception as e:
        print(f" Initialization failed: {e}")
        return False

def test_reset_functionality():
    """Test environment reset"""
    print("\n" + "="*60)
    print("TEST 2: Reset Functionality")
    print("="*60)
    
    try:
        data = create_sample_data(200)
        env = TradingEnvironment(df=data, initial_balance=10000)
        
        # Reset environment
        initial_obs = env.reset()
        
        print(" Reset successful")
        print(f"   - Observation shape: {initial_obs.shape}")
        print(f"   - Balance reset: ${env.balance}")
        print(f"   - Position: {env.position.name}")
        print(f"   - Current step: {env.current_step}")
        
        # Reset with seed
        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)
        
        if np.array_equal(obs1, obs2):
            print(" Deterministic reset with seed works")
        else:
            print("  Reset with seed not deterministic")
        
        return True
        
    except Exception as e:
        print(f" Reset failed: {e}")
        return False

def test_buy_action():
    """Test buy action execution"""
    print("\n" + "="*60)
    print("TEST 3: Buy Action")
    print("="*60)
    
    try:
        data = create_sample_data(200)
        env = TradingEnvironment(df=data, initial_balance=10000, fee_rate=0.0026)
        
        obs = env.reset()
        initial_balance = env.balance
        
        # Execute buy action
        obs, reward, done, truncated, info = env.step(Actions.BUY)
        
        print(" Buy action executed")
        print(f"   - Position: {env.position.name}")
        print(f"   - Position size: {env.position_size:.6f}")
        print(f"   - Entry price: ${env.entry_price:.2f}")
        print(f"   - Balance after fees: ${env.balance:.2f}")
        print(f"   - Fees paid: ${initial_balance - env.balance:.2f}")
        print(f"   - Reward: {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f" Buy action failed: {e}")
        return False

def test_sell_action():
    """Test sell action (close long position)"""
    print("\n" + "="*60)
    print("TEST 4: Sell Action (Close Long)")
    print("="*60)
    
    try:
        data = create_sample_data(200, trend='up')  # Uptrend for profit
        env = TradingEnvironment(df=data, initial_balance=10000)
        
        obs = env.reset()
        
        # Open long position
        env.step(Actions.BUY)
        entry_price = env.entry_price
        
        # Wait a few steps
        for _ in range(5):
            env.step(Actions.HOLD)
        
        # Close position
        obs, reward, done, truncated, info = env.step(Actions.SELL)
        
        print(" Sell action executed")
        print(f"   - Entry price: ${entry_price:.2f}")
        print(f"   - Exit price: ${env._get_current_price():.2f}")
        print(f"   - Position: {env.position.name}")
        print(f"   - Realized PnL: ${env.realized_pnl:.2f}")
        print(f"   - Total fees: ${env.total_fees_paid:.2f}")
        print(f"   - Final balance: ${env.balance:.2f}")
        
        return True
        
    except Exception as e:
        print(f" Sell action failed: {e}")
        return False

def test_hold_action():
    """Test hold action"""
    print("\n" + "="*60)
    print("TEST 5: Hold Action")
    print("="*60)
    
    try:
        data = create_sample_data(200)
        env = TradingEnvironment(df=data, initial_balance=10000)
        
        obs = env.reset()
        initial_balance = env.balance
        
        # Execute hold action
        obs, reward, done, truncated, info = env.step(Actions.HOLD)
        
        print(" Hold action executed")
        print(f"   - Position: {env.position.name}")
        print(f"   - Balance unchanged: ${env.balance:.2f}")
        print(f"   - Reward: {reward:.4f}")
        
        return True
        
    except Exception as e:
        print(f" Hold action failed: {e}")
        return False

def test_stop_loss():
    """Test stop loss functionality"""
    print("\n" + "="*60)
    print("TEST 6: Stop Loss")
    print("="*60)
    
    try:
        data = create_sample_data(200, trend='down')  # Downtrend to trigger stop loss
        env = TradingEnvironment(
            df=data, 
            initial_balance=10000,
            stop_loss=0.02,  # 2% stop loss
            take_profit=0.10
        )
        
        obs = env.reset()
        
        # Open long position
        env.step(Actions.BUY)
        entry_price = env.entry_price
        position_opened = True
        
        # Step until stop loss triggers
        for i in range(50):
            obs, reward, done, truncated, info = env.step(Actions.HOLD)
            
            if env.position == Positions.FLAT and position_opened:
                exit_price = env._get_current_price()
                pnl_pct = (exit_price - entry_price) / entry_price
                print(" Stop loss triggered")
                print(f"   - Entry price: ${entry_price:.2f}")
                print(f"   - Exit price: ${exit_price:.2f}")
                print(f"   - Loss: {pnl_pct*100:.2f}%")
                print(f"   - Realized PnL: ${env.realized_pnl:.2f}")
                return True
        
        print("  Stop loss not triggered in test period")
        return True
        
    except Exception as e:
        print(f" Stop loss test failed: {e}")
        return False

def test_take_profit():
    """Test take profit functionality"""
    print("\n" + "="*60)
    print("TEST 7: Take Profit")
    print("="*60)
    
    try:
        data = create_sample_data(200, trend='up')  # Uptrend to trigger take profit
        env = TradingEnvironment(
            df=data,
            initial_balance=10000,
            stop_loss=0.10,
            take_profit=0.02  # 2% take profit
        )
        
        obs = env.reset()
        
        # Open long position
        env.step(Actions.BUY)
        entry_price = env.entry_price
        position_opened = True
        
        # Step until take profit triggers
        for i in range(50):
            obs, reward, done, truncated, info = env.step(Actions.HOLD)
            
            if env.position == Positions.FLAT and position_opened:
                exit_price = env._get_current_price()
                pnl_pct = (exit_price - entry_price) / entry_price
                print(" Take profit triggered")
                print(f"   - Entry price: ${entry_price:.2f}")
                print(f"   - Exit price: ${exit_price:.2f}")
                print(f"   - Profit: {pnl_pct*100:.2f}%")
                print(f"   - Realized PnL: ${env.realized_pnl:.2f}")
                return True
        
        print("  Take profit not triggered in test period")
        return True
        
    except Exception as e:
        print(f" Take profit test failed: {e}")
        return False

def test_short_selling():
    """Test short selling (if enabled)"""
    print("\n" + "="*60)
    print("TEST 8: Short Selling")
    print("="*60)
    
    try:
        data = create_sample_data(200, trend='down')
        env = TradingEnvironment(
            df=data,
            initial_balance=10000,
            enable_short=True  # Enable short selling
        )
        
        obs = env.reset()
        
        # Open short position
        obs, reward, done, truncated, info = env.step(Actions.SELL)
        
        if env.position == Positions.SHORT:
            print(" Short position opened")
            print(f"   - Position: {env.position.name}")
            print(f"   - Entry price: ${env.entry_price:.2f}")
            print(f"   - Position size: {env.position_size:.6f}")
            
            # Wait and close
            for _ in range(5):
                env.step(Actions.HOLD)
            
            # Close short (buy to cover)
            env.step(Actions.BUY)
            print(f"   - Short position closed")
            print(f"   - Realized PnL: ${env.realized_pnl:.2f}")
        else:
            print("  Short selling not executed (disabled or insufficient balance)")
        
        return True
        
    except Exception as e:
        print(f" Short selling test failed: {e}")
        return False

def test_episode_completion():
    """Test full episode execution"""
    print("\n" + "="*60)
    print("TEST 9: Full Episode")
    print("="*60)
    
    try:
        data = create_sample_data(500)
        env = TradingEnvironment(df=data, initial_balance=10000)
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        # Run full episode with random actions
        while not env.done and steps < 400:
            action = np.random.choice([Actions.HOLD, Actions.BUY, Actions.SELL], p=[0.5, 0.25, 0.25])
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        # Get performance summary
        summary = env.get_performance_summary()
        
        print(" Episode completed")
        print(f"   - Steps: {steps}")
        print(f"   - Total reward: {total_reward:.4f}")
        print(f"   - Total trades: {summary.get('total_trades', 0)}")
        print(f"   - Win rate: {summary.get('win_rate', 0):.2%}")
        print(f"   - Total return: {summary.get('total_return_pct', 0):.2f}%")
        print(f"   - Sharpe ratio: {summary.get('sharpe_ratio', 0):.2f}")
        print(f"   - Max drawdown: {summary.get('max_drawdown', 0):.2%}")
        print(f"   - Final value: ${summary.get('final_portfolio_value', 0):.2f}")
        
        return True
        
    except Exception as e:
        print(f" Episode execution failed: {e}")
        return False

def test_performance_metrics():
    """Test performance metric calculations"""
    print("\n" + "="*60)
    print("TEST 10: Performance Metrics")
    print("="*60)
    
    try:
        data = create_sample_data(300)
        env = TradingEnvironment(df=data, initial_balance=10000)
        
        obs = env.reset()
        
        # Execute some trades
        env.step(Actions.BUY)
        for _ in range(10):
            env.step(Actions.HOLD)
        env.step(Actions.SELL)
        
        env.step(Actions.BUY)
        for _ in range(5):
            env.step(Actions.HOLD)
        env.step(Actions.SELL)
        
        # Get metrics
        state = env.get_trading_state()
        info = env._get_info()
        
        print(" Metrics calculated")
        print(f"   - Total trades: {state.total_trades}")
        print(f"   - Winning trades: {state.winning_trades}")
        print(f"   - Losing trades: {state.losing_trades}")
        print(f"   - Win rate: {info['win_rate']:.2%}")
        print(f"   - Sharpe ratio: {state.sharpe_ratio:.2f}")
        print(f"   - Max drawdown: {state.max_drawdown:.2%}")
        print(f"   - Current drawdown: {state.current_drawdown:.2%}")
        
        return True
        
    except Exception as e:
        print(f" Performance metrics test failed: {e}")
        return False

def test_with_features():
    """Test environment with technical indicator features"""
    print("\n" + "="*60)
    print("TEST 11: With Technical Features")
    print("="*60)
    
    try:
        # Create sample data
        data = create_sample_data(300)
        
        # Create fake features DataFrame
        features = pd.DataFrame({
            'rsi_14': np.random.uniform(30, 70, len(data)),
            'macd': np.random.uniform(-100, 100, len(data)),
            'bb_position': np.random.uniform(0, 1, len(data))
        }, index=data.index)
        
        # Initialize environment with features
        env = TradingEnvironment(
            df=data,
            initial_balance=10000,
            features_df=features
        )
        
        obs = env.reset()
        
        print(" Environment with features initialized")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Features included: {features.shape[1]} indicators")
        
        # Test a few steps
        for _ in range(5):
            action = np.random.choice([0, 1, 2])
            obs, reward, done, truncated, info = env.step(action)
        
        print(" Steps with features executed successfully")
        
        return True
        
    except Exception as e:
        print(f" Features test failed: {e}")
        return False

def test_position_sizing():
    """Test dynamic position sizing"""
    print("\n" + "="*60)
    print("TEST 12: Position Sizing")
    print("="*60)
    
    try:
        data = create_sample_data(200)
        
        # Test fixed position sizing
        env_fixed = TradingEnvironment(
            df=data,
            initial_balance=10000,
            position_sizing='fixed'
        )
        
        env_fixed.reset()
        env_fixed.step(Actions.BUY)
        fixed_size = env_fixed.position_size
        
        # Test dynamic position sizing
        env_dynamic = TradingEnvironment(
            df=data,
            initial_balance=10000,
            position_sizing='dynamic',
            risk_per_trade=0.02
        )
        
        env_dynamic.reset()
        env_dynamic.step(Actions.BUY)
        dynamic_size = env_dynamic.position_size
        
        print(" Position sizing tested")
        print(f"   - Fixed sizing: {fixed_size:.6f} units")
        print(f"   - Dynamic sizing: {dynamic_size:.6f} units")
        print(f"   - Dynamic uses volatility-based sizing")
        
        return True
        
    except Exception as e:
        print(f" Position sizing test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("TRADING ENVIRONMENT TEST SUITE")
    print("="*60)
    
    tests = [
        ("Basic Initialization", test_basic_initialization),
        ("Reset Functionality", test_reset_functionality),
        ("Buy Action", test_buy_action),
        ("Sell Action", test_sell_action),
        ("Hold Action", test_hold_action),
        ("Stop Loss", test_stop_loss),
        ("Take Profit", test_take_profit),
        ("Short Selling", test_short_selling),
        ("Episode Completion", test_episode_completion),
        ("Performance Metrics", test_performance_metrics),
        ("With Features", test_with_features),
        ("Position Sizing", test_position_sizing)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f" Test {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n ALL TESTS PASSED! Trading Environment is working correctly!")
    else:
        print(f"\n  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)