"""
Test Script for Reward Functions Module
Tests all reward calculation methods
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import reward functions
from src.environment.rewards import RewardCalculator, RewardShaper

def test_simple_returns():
    """Test simple returns calculation"""
    print("\n" + "="*60)
    print("TEST 1: Simple Returns")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Test positive return
        reward = calc.simple_returns(10500, 10000)
        expected = 0.05
        print(f" Positive return: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test negative return
        reward = calc.simple_returns(9500, 10000)
        expected = -0.05
        print(f" Negative return: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test with scaling
        reward = calc.simple_returns(10500, 10000, scaling=2.0)
        expected = 0.10
        print(f" Scaled return: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test zero return
        reward = calc.simple_returns(10000, 10000)
        print(f" Zero return: {reward:.4f}")
        assert reward == 0
        
        return True
        
    except Exception as e:
        print(f" Simple returns test failed: {e}")
        return False

def test_log_returns():
    """Test logarithmic returns calculation"""
    print("\n" + "="*60)
    print("TEST 2: Log Returns")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Test positive log return
        reward = calc.log_returns(11000, 10000)
        expected = np.log(1.1)
        print(f" Positive log return: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test negative log return
        reward = calc.log_returns(9000, 10000)
        expected = np.log(0.9)
        print(f" Negative log return: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test edge case (zero/negative values)
        reward = calc.log_returns(0, 10000)
        print(f" Edge case handled: {reward:.4f}")
        assert reward == 0
        
        return True
        
    except Exception as e:
        print(f" Log returns test failed: {e}")
        return False

def test_risk_adjusted_returns():
    """Test Sharpe ratio calculation"""
    print("\n" + "="*60)
    print("TEST 3: Risk-Adjusted Returns (Sharpe Ratio)")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Create portfolio with positive Sharpe
        portfolio_values = [10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400]
        sharpe = calc.risk_adjusted_returns(portfolio_values)
        print(f" Positive Sharpe ratio: {sharpe:.4f}")
        assert sharpe > 0
        
        # Create portfolio with negative Sharpe
        portfolio_values = [10000, 9900, 9950, 9800, 9850, 9700, 9750, 9600]
        sharpe = calc.risk_adjusted_returns(portfolio_values)
        print(f" Negative Sharpe ratio: {sharpe:.4f}")
        assert sharpe < 0
        
        # Test with insufficient data
        portfolio_values = [10000]
        sharpe = calc.risk_adjusted_returns(portfolio_values)
        print(f" Insufficient data handled: {sharpe:.4f}")
        assert sharpe == 0
        
        return True
        
    except Exception as e:
        print(f" Risk-adjusted returns test failed: {e}")
        return False

def test_profit_factor():
    """Test profit factor reward calculation"""
    print("\n" + "="*60)
    print("TEST 4: Profit Factor Reward")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Test with more wins than losses
        winning_trades = [100, 200, 150]
        losing_trades = [-50, -30]
        pf = calc.profit_factor_reward(winning_trades, losing_trades)
        expected = (450 / 80) - 1
        print(f" Profit factor (wins > losses): {pf:.4f} (expected {expected:.4f})")
        assert abs(pf - expected) < 0.0001
        
        # Test with only winning trades
        winning_trades = [100, 200]
        losing_trades = []
        pf = calc.profit_factor_reward(winning_trades, losing_trades)
        print(f" Only wins: {pf:.4f}")
        assert pf == 1
        
        # Test with only losing trades
        winning_trades = []
        losing_trades = [-50, -100]
        pf = calc.profit_factor_reward(winning_trades, losing_trades)
        print(f" Only losses: {pf:.4f}")
        assert pf == -1
        
        return True
        
    except Exception as e:
        print(f" Profit factor test failed: {e}")
        return False

def test_drawdown_adjusted_returns():
    """Test drawdown-adjusted returns"""
    print("\n" + "="*60)
    print("TEST 5: Drawdown-Adjusted Returns")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Test with no drawdown
        reward = calc.drawdown_adjusted_returns(10500, 10000, 0.0, 0.0)
        expected = 0.05
        print(f" No drawdown: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test with moderate drawdown
        reward = calc.drawdown_adjusted_returns(10500, 10000, 0.1, 0.1, drawdown_penalty=2.0)
        expected = 0.05 - (0.1 * 2.0)
        print(f" With 10% drawdown: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Test with high drawdown (>30%)
        reward = calc.drawdown_adjusted_returns(10500, 10000, 0.4, 0.4, drawdown_penalty=2.0)
        print(f" High drawdown penalty: {reward:.4f}")
        assert reward < 0  # Should be negative despite positive returns
        
        return True
        
    except Exception as e:
        print(f" Drawdown-adjusted returns test failed: {e}")
        return False

def test_composite_reward():
    """Test composite reward calculation"""
    print("\n" + "="*60)
    print("TEST 6: Composite Reward")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Create test data
        portfolio_history = [10000, 10100, 10050, 10200, 10150, 10300]
        trades = [
            {'pnl': 100}, {'pnl': -50}, {'pnl': 200},
            {'pnl': -30}, {'pnl': 150}
        ]
        
        # Test with default weights
        reward = calc.composite_reward(
            current_value=10300,
            previous_value=10150,
            portfolio_history=portfolio_history,
            trades=trades,
            current_drawdown=0.05
        )
        print(f" Composite reward (default weights): {reward:.4f}")
        assert isinstance(reward, float)
        
        # Test with custom weights
        custom_weights = {
            'returns': 2.0,
            'sharpe': 0.5,
            'drawdown': 1.0,
            'win_rate': 0.3
        }
        reward = calc.composite_reward(
            current_value=10300,
            previous_value=10150,
            portfolio_history=portfolio_history,
            trades=trades,
            current_drawdown=0.05,
            weights=custom_weights
        )
        print(f" Composite reward (custom weights): {reward:.4f}")
        assert isinstance(reward, float)
        
        # Test with no trades
        reward = calc.composite_reward(
            current_value=10300,
            previous_value=10150,
            portfolio_history=portfolio_history,
            trades=[],
            current_drawdown=0.05
        )
        print(f" Composite reward (no trades): {reward:.4f}")
        assert isinstance(reward, float)
        
        return True
        
    except Exception as e:
        print(f" Composite reward test failed: {e}")
        return False

def test_sortino_ratio():
    """Test Sortino ratio calculation"""
    print("\n" + "="*60)
    print("TEST 7: Sortino Ratio")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Portfolio with minimal downside
        portfolio_values = [10000, 10100, 10090, 10200, 10190, 10300]
        sortino = calc.sortino_ratio_reward(portfolio_values)
        print(f" Low downside volatility: {sortino:.4f}")
        assert sortino > 0
        
        # Portfolio with high downside
        portfolio_values = [10000, 9800, 10100, 9700, 10200, 9600]
        sortino = calc.sortino_ratio_reward(portfolio_values)
        print(f" High downside volatility: {sortino:.4f}")
        
        # No downside returns
        portfolio_values = [10000, 10100, 10200, 10300, 10400]
        sortino = calc.sortino_ratio_reward(portfolio_values)
        print(f" No downside: {sortino:.4f}")
        assert sortino == 1  # Perfect score when no downside
        
        return True
        
    except Exception as e:
        print(f" Sortino ratio test failed: {e}")
        return False

def test_calmar_ratio():
    """Test Calmar ratio calculation"""
    print("\n" + "="*60)
    print("TEST 8: Calmar Ratio")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Good Calmar ratio (high return, low drawdown)
        calmar = calc.calmar_ratio_reward(total_return=0.50, max_drawdown=0.10)
        expected = 5.0
        print(f" Good Calmar: {calmar:.4f} (expected {expected:.4f})")
        assert abs(calmar - expected) < 0.0001
        
        # Poor Calmar ratio
        calmar = calc.calmar_ratio_reward(total_return=0.10, max_drawdown=0.30)
        expected = 0.10 / 0.30
        print(f" Poor Calmar: {calmar:.4f} (expected {expected:.4f})")
        assert abs(calmar - expected) < 0.0001
        
        # No drawdown edge case
        calmar = calc.calmar_ratio_reward(total_return=0.20, max_drawdown=0.0)
        print(f" No drawdown: {calmar:.4f}")
        assert calmar == 0.20
        
        return True
        
    except Exception as e:
        print(f" Calmar ratio test failed: {e}")
        return False

def test_trade_efficiency():
    """Test trade efficiency reward"""
    print("\n" + "="*60)
    print("TEST 9: Trade Efficiency")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Efficient trading (good return, moderate trades)
        efficiency = calc.trade_efficiency_reward(
            num_trades=100,
            total_return=0.50,
            target_trades=100
        )
        print(f" Optimal efficiency: {efficiency:.4f}")
        assert efficiency > 0
        
        # Too many trades
        efficiency = calc.trade_efficiency_reward(
            num_trades=200,
            total_return=0.50,
            target_trades=100
        )
        print(f" Too many trades: {efficiency:.4f}")
        
        # Too few trades
        efficiency = calc.trade_efficiency_reward(
            num_trades=20,
            total_return=0.50,
            target_trades=100
        )
        print(f" Too few trades: {efficiency:.4f}")
        
        # No trades edge case
        efficiency = calc.trade_efficiency_reward(
            num_trades=0,
            total_return=0.0,
            target_trades=100
        )
        print(f" No trades: {efficiency:.4f}")
        assert efficiency == 0
        
        return True
        
    except Exception as e:
        print(f" Trade efficiency test failed: {e}")
        return False

def test_sparse_reward():
    """Test sparse reward function"""
    print("\n" + "="*60)
    print("TEST 10: Sparse Reward")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        # Successful episode
        reward = calc.sparse_reward(episode_return=0.10, episode_length=100)
        print(f" Success (10% return): {reward:.4f}")
        assert reward > 1.0
        
        # Small positive return
        reward = calc.sparse_reward(episode_return=0.02, episode_length=100)
        print(f" Small positive: {reward:.4f}")
        assert reward == 0.02
        
        # Loss
        reward = calc.sparse_reward(episode_return=-0.05, episode_length=100)
        print(f" Loss: {reward:.4f}")
        assert reward < -0.5
        
        return True
        
    except Exception as e:
        print(f" Sparse reward test failed: {e}")
        return False

def test_curiosity_reward():
    """Test curiosity/exploration reward"""
    print("\n" + "="*60)
    print("TEST 11: Curiosity Reward")
    print("="*60)
    
    try:
        calc = RewardCalculator()
        
        state_visits = {}
        
        # First visit to state
        reward = calc.curiosity_reward(state_visits, "state_001", exploration_bonus=0.1)
        print(f" First visit: {reward:.4f}")
        assert reward == 0.1
        
        # Update visits
        state_visits["state_001"] = 1
        
        # Second visit
        reward = calc.curiosity_reward(state_visits, "state_001", exploration_bonus=0.1)
        expected = 0.1 / 2
        print(f" Second visit: {reward:.4f} (expected {expected:.4f})")
        assert abs(reward - expected) < 0.0001
        
        # Many visits
        state_visits["state_001"] = 10
        reward = calc.curiosity_reward(state_visits, "state_001", exploration_bonus=0.1)
        print(f" Many visits (10): {reward:.4f}")
        assert reward < 0.01
        
        return True
        
    except Exception as e:
        print(f" Curiosity reward test failed: {e}")
        return False

def test_reward_shaping():
    """Test reward shaping functions"""
    print("\n" + "="*60)
    print("TEST 12: Reward Shaping")
    print("="*60)
    
    try:
        shaper = RewardShaper()
        
        # Test normalization
        normalized = shaper.normalize_reward(2.5, min_reward=-1, max_reward=1)
        print(f" Normalization (2.5 -> {normalized:.4f})")
        assert normalized == 1.0
        
        normalized = shaper.normalize_reward(-3.0, min_reward=-1, max_reward=1)
        print(f" Normalization (-3.0 -> {normalized:.4f})")
        assert normalized == -1.0
        
        # Test exponential scaling
        scaled = shaper.exponential_scaling(0.5, temperature=1.0)
        print(f" Exponential scaling: {scaled:.4f}")
        assert 0 < scaled < 0.5
        
        # Test potential-based shaping
        shaped = shaper.potential_based_shaping(
            current_potential=0.8,
            previous_potential=0.6,
            gamma=0.99
        )
        expected = 0.99 * 0.8 - 0.6
        print(f" Potential shaping: {shaped:.4f} (expected {expected:.4f})")
        assert abs(shaped - expected) < 0.0001
        
        # Test adaptive scaling
        rewards_history = [0.1, -0.05, 0.2, -0.1, 0.15, 0.0, 0.1, -0.2]
        current_reward = 0.5
        scaled = shaper.adaptive_scaling(rewards_history, current_reward, window=5)
        print(f" Adaptive scaling: {scaled:.4f}")
        assert isinstance(scaled, float)
        
        return True
        
    except Exception as e:
        print(f" Reward shaping test failed: {e}")
        return False

def run_all_tests():
    """Run all reward function tests"""
    print("\n" + "="*60)
    print("REWARD FUNCTIONS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Simple Returns", test_simple_returns),
        ("Log Returns", test_log_returns),
        ("Risk-Adjusted Returns", test_risk_adjusted_returns),
        ("Profit Factor", test_profit_factor),
        ("Drawdown-Adjusted Returns", test_drawdown_adjusted_returns),
        ("Composite Reward", test_composite_reward),
        ("Sortino Ratio", test_sortino_ratio),
        ("Calmar Ratio", test_calmar_ratio),
        ("Trade Efficiency", test_trade_efficiency),
        ("Sparse Reward", test_sparse_reward),
        ("Curiosity Reward", test_curiosity_reward),
        ("Reward Shaping", test_reward_shaping)
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
        print("\n ALL TESTS PASSED! Reward functions are working correctly!")
    else:
        print(f"\n  {total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)