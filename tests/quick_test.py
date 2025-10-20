"""
Quick Position Sizing Bug Detector
Specifically tests for the issues found in your report

This will immediately show you if positions are accumulating incorrectly.

Usage:
    python quick_test.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.trading_env import TradingEnvironment, Actions, Positions


def create_simple_data(n_steps=200):
    """Create simple test data"""
    dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='1h')
    
    # Simple prices around $3.53 (like SOL in your report)
    base_price = 3.53
    prices = base_price + np.random.uniform(-0.1, 0.1, n_steps)
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.ones(n_steps) * 1000
    }, index=dates)
    
    return df


def test_position_bug():
    """
    Test for the specific bug seen in your report:
    - Position sizes growing to impossible levels (18,928 SOL with $10k balance)
    """
    
    print("="*80)
    print(" TESTING FOR POSITION SIZING BUG")
    print("="*80)
    print("\nScenario: Trade SOL @ $3.53 with $10,000 balance")
    print("Expected max position: ~2,692 SOL ($9,500 ÷ $3.53)")
    print("Your report showed: 18,928 SOL (7x more than possible!)")
    print("\nLet's test if this happens...\n")
    
    df = create_simple_data(n_steps=200)
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,
        precompute_observations=False
    )
    
    env.reset()
    
    print("Step 1: Initial state")
    print(f"  Balance: ${env.balance:,.2f}")
    print(f"  Position: {env.position.name}")
    print(f"  Position size: {env.position_size:.2f} coins")
    
    # BUY
    print("\nStep 2: Execute BUY")
    env.step(Actions.BUY)
    
    buy_price = env.entry_price
    position_size_1 = env.position_size
    position_value_1 = position_size_1 * buy_price
    balance_after_buy = env.balance
    
    print(f"  Bought at: ${buy_price:.4f}")
    print(f"  Position size: {position_size_1:.2f} coins")
    print(f"  Position value: ${position_value_1:,.2f}")
    print(f"  Remaining balance: ${balance_after_buy:,.2f}")
    
    # Check if position is reasonable
    if position_value_1 > 10000 * 1.1:
        print(f"\n   ERROR: Position value ${position_value_1:,.2f} exceeds initial balance!")
        print(f"  This should be impossible!")
        return False
    else:
        print(f"   Position value is reasonable")
    
    # HOLD for a few steps
    print("\nStep 3: Hold for 10 steps")
    for i in range(10):
        env.step(Actions.HOLD)
    
    # SELL
    print("\nStep 4: Execute SELL")
    env.step(Actions.SELL)
    
    print(f"  Position closed: {env.position.name}")
    print(f"  Final balance: ${env.balance:,.2f}")
    print(f"  P&L: ${env.balance - 10000:,.2f}")
    
    # BUY AGAIN (this is where bugs often appear)
    print("\nStep 5: BUY AGAIN (critical test!)")
    env.step(Actions.BUY)
    
    buy_price_2 = env.entry_price
    position_size_2 = env.position_size
    position_value_2 = position_size_2 * buy_price_2
    
    print(f"  Bought at: ${buy_price_2:.4f}")
    print(f"  Position size: {position_size_2:.2f} coins")
    print(f"  Position value: ${position_value_2:,.2f}")
    
    # Compare to first buy
    print(f"\n  Comparing to first buy:")
    print(f"    First position:  {position_size_1:.2f} coins (${position_value_1:,.2f})")
    print(f"    Second position: {position_size_2:.2f} coins (${position_value_2:,.2f})")
    
    # Check if position grew unreasonably
    if position_size_2 > position_size_1 * 2:
        print(f"\n   ERROR: Position size more than doubled!")
        print(f"  Second position is {position_size_2/position_size_1:.1f}x the first")
        print(f"  This suggests positions are accumulating!")
        return False
    
    if position_value_2 > 20000:
        print(f"\n   ERROR: Position value ${position_value_2:,.2f} exceeds 2x initial balance!")
        return False
    
    print(f"   Position sizes are consistent")
    
    # Multiple cycles to check for accumulation
    print("\nStep 6: Multiple buy-sell cycles (checking for accumulation)")
    
    position_sizes = []
    
    for cycle in range(5):
        # Sell current position
        if env.position != Positions.FLAT:
            env.step(Actions.SELL)
        
        # Hold a bit
        for _ in range(3):
            env.step(Actions.HOLD)
        
        # Buy
        env.step(Actions.BUY)
        
        position_value = env.position_size * env._get_current_price()
        position_sizes.append(position_value)
        
        print(f"  Cycle {cycle+1}: Position value = ${position_value:,.2f}")
        
        # Check for unrealistic growth
        if position_value > 20000:
            print(f"\n   ERROR: Position value ${position_value:,.2f} grew too large!")
            print(f"  This indicates accumulating error!")
            return False
    
    # Check consistency
    avg_position = np.mean(position_sizes)
    std_position = np.std(position_sizes)
    
    print(f"\n  Position statistics:")
    print(f"    Average: ${avg_position:,.2f}")
    print(f"    Std Dev: ${std_position:,.2f}")
    print(f"    CV: {std_position/avg_position*100:.1f}%")
    
    if std_position / avg_position > 0.5:
        print(f"    WARNING: High variance in position sizes!")
        print(f"  Positions may be unstable")
    else:
        print(f"   Position sizes are stable")
    
    return True


def test_impossible_profit():
    """
    Test if the environment can generate impossible profits
    like the $16,327 profit on $10k balance you saw
    """
    
    print("\n" + "="*80)
    print(" TESTING FOR IMPOSSIBLE PROFIT BUG")
    print("="*80)
    print("\nScenario: Can we generate $16,000 profit with $10,000 balance?")
    print("This should be impossible without extreme leverage or bugs!\n")
    
    df = create_simple_data(n_steps=200)
    
    # Create price movement that would give big profit
    # Starting at $3.53, move to $4.00 (13% gain)
    df['close'] = np.linspace(3.53, 4.00, len(df))
    df['open'] = df['close'] * 0.999
    df['high'] = df['close'] * 1.001
    df['low'] = df['close'] * 0.998
    
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,
        precompute_observations=False
    )
    
    env.reset()
    
    print("Step 1: Buy at $3.53")
    env.step(Actions.BUY)
    
    position_size = env.position_size
    entry_price = env.entry_price
    
    print(f"  Position: {position_size:.2f} coins @ ${entry_price:.4f}")
    print(f"  Position value: ${position_size * entry_price:,.2f}")
    
    # Hold until price reaches $4.00
    print("\nStep 2: Hold while price rises to $4.00")
    for _ in range(len(df) - 20):
        env.step(Actions.HOLD)
    
    current_price = env._get_current_price()
    unrealized_pnl = (current_price - entry_price) * position_size
    
    print(f"  Current price: ${current_price:.4f}")
    print(f"  Unrealized P&L: ${unrealized_pnl:,.2f}")
    
    print("\nStep 3: Sell at peak")
    balance_before_sell = env.balance
    env.step(Actions.SELL)
    balance_after_sell = env.balance
    
    actual_profit = balance_after_sell - 10000
    
    print(f"  Balance before sell: ${balance_before_sell:,.2f}")
    print(f"  Balance after sell: ${balance_after_sell:,.2f}")
    print(f"  Actual profit: ${actual_profit:,.2f}")
    
    # Check if profit is realistic
    # With $9,500 invested and 13% price gain, profit should be ~$1,235
    expected_max_profit = 9500 * 0.13  # ~$1,235
    
    print(f"\n  Expected profit (13% on $9,500): ${expected_max_profit:,.2f}")
    print(f"  Actual profit: ${actual_profit:,.2f}")
    
    if actual_profit > expected_max_profit * 2:
        print(f"\n   ERROR: Profit is {actual_profit/expected_max_profit:.1f}x higher than possible!")
        print(f"  This indicates a bug in P&L calculation")
        return False
    
    if actual_profit > 10000:
        print(f"\n   ERROR: Profit ${actual_profit:,.2f} exceeds initial balance!")
        print(f"  100%+ profit from 13% price move is impossible!")
        return False
    
    print(f"   Profit is realistic (within 2x of expected)")
    
    return True


def test_fee_accumulation():
    """Test if fees accumulate correctly without double-counting"""
    
    print("\n" + "="*80)
    print(" TESTING FEE CALCULATION")
    print("="*80)
    print("\nScenario: Execute 10 trades and check total fees\n")
    
    df = create_simple_data(n_steps=200)
    env = TradingEnvironment(
        df=df,
        initial_balance=10000,
        fee_rate=0.0026,  # 0.26%
        precompute_observations=False
    )
    
    env.reset()
    
    # Execute 10 buy-sell cycles
    for cycle in range(10):
        env.step(Actions.BUY)
        for _ in range(3):
            env.step(Actions.HOLD)
        env.step(Actions.SELL)
        for _ in range(2):
            env.step(Actions.HOLD)
    
    total_fees = env.total_fees_paid
    total_trades = len([t for t in env.trades if t.get('pnl') is not None])
    
    # Expected: ~$50 per round trip (0.26% * 2 * $9,500) * 10 cycles = ~$500
    expected_fees = 10 * 2 * 9500 * 0.0026  # ~$494
    
    print(f"Executed: {total_trades} completed trades")
    print(f"Total fees paid: ${total_fees:,.2f}")
    print(f"Expected fees: ${expected_fees:,.2f}")
    print(f"Difference: ${abs(total_fees - expected_fees):,.2f}")
    
    fee_ratio = total_fees / expected_fees
    
    if 0.8 <= fee_ratio <= 1.5:
        print(f"\n Fees are reasonable ({fee_ratio:.2f}x expected)")
        return True
    elif fee_ratio > 1.5:
        print(f"\n ERROR: Fees are {fee_ratio:.2f}x higher than expected!")
        print(f"Possible double-counting of fees")
        return False
    else:
        print(f"\n  WARNING: Fees are {fee_ratio:.2f}x lower than expected")
        print(f"Fees may not be calculated correctly")
        return False


def main():
    """Run all quick tests"""
    
    print("\n" + "="*80)
    print(" QUICK BUG DETECTION TEST SUITE")
    print("="*80)
    print("\nThis will test for the specific issues found in your report:")
    print("  1. Position sizes growing to impossible levels")
    print("  2. Profits exceeding what's mathematically possible")
    print("  3. Fee calculation errors")
    print("\n" + "="*80 + "\n")
    
    results = []
    
    # Test 1: Position sizing
    try:
        result1 = test_position_bug()
        results.append(("Position Sizing", result1))
    except Exception as e:
        print(f"\n Position sizing test crashed: {e}")
        results.append(("Position Sizing", False))
    
    # Test 2: Impossible profits
    try:
        result2 = test_impossible_profit()
        results.append(("Profit Limits", result2))
    except Exception as e:
        print(f"\n Profit test crashed: {e}")
        results.append(("Profit Limits", False))
    
    # Test 3: Fee calculation
    try:
        result3 = test_fee_accumulation()
        results.append(("Fee Calculation", result3))
    except Exception as e:
        print(f"\n Fee test crashed: {e}")
        results.append(("Fee Calculation", False))
    
    # Summary
    print("\n" + "="*80)
    print(" TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = " PASS" if result else " FAIL"
        print(f"{status} | {test_name}")
    
    print("="*80)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n ALL TESTS PASSED!")
        print("Your environment is NOT showing the bugs from the report.")
        print("The issues may be in training logic or data handling.")
    else:
        print(f"\n  {total - passed} TEST(S) FAILED")
        print("\nYour environment has bugs that need fixing!")
        print("\nNext steps:")
        print("  1. Review the failed test output above")
        print("  2. Check your position sizing logic in _execute_buy()")
        print("  3. Check your fee calculation in _close_position()")
        print("  4. Apply Fix #3 (tighter validation) to catch bugs earlier")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)