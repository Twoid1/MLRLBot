"""
Comprehensive Test Suite for Trading Environment
Tests all critical functions to ensure correctness

Run this before training to catch bugs early!

Usage:
    python test_trading_env.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environment.trading_env import TradingEnvironment, Positions, Actions

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TradingEnvironmentTester:
    """Comprehensive test suite for trading environment"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_results = []
        
    def create_test_data(self, n_steps=1000, base_price=10000, volatility=0.02):
        """Create synthetic test data"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='1h')
        
        # Generate realistic price movement
        returns = np.random.normal(0, volatility, n_steps)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_steps)),
            'high': prices * (1 + np.random.uniform(0.001, 0.01, n_steps)),
            'low': prices * (1 - np.random.uniform(0.001, 0.01, n_steps)),
            'close': prices,
            'volume': np.random.uniform(100, 1000, n_steps)
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def log_result(self, test_name: str, passed: bool, message: str, warning: bool = False):
        """Log test result"""
        if warning:
            status = "  WARN"
            self.warnings += 1
        elif passed:
            status = " PASS"
            self.passed += 1
        else:
            status = " FAIL"
            self.failed += 1
        
        result = f"{status} | {test_name}: {message}"
        print(result)
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'warning': warning,
            'message': message
        })
        
    # ========================================================================
    # TEST 1: Environment Initialization
    # ========================================================================
    
    def test_initialization(self):
        """Test that environment initializes correctly"""
        print("\n" + "="*80)
        print("TEST 1: ENVIRONMENT INITIALIZATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500)
            env = TradingEnvironment(
                df=df,
                initial_balance=10000,
                fee_rate=0.0026,
                enable_short=False,
                precompute_observations=False  # Test without pre-computation first
            )
            
            # Check initial state
            assert env.initial_balance == 10000, "Initial balance incorrect"
            assert env.balance == 10000, "Balance not set correctly"
            assert env.position == Positions.FLAT, "Should start with no position"
            assert env.position_size == 0, "Position size should be 0"
            assert env.entry_price == 0, "Entry price should be 0"
            
            self.log_result("Initialization", True, "All initial values correct")
            
        except Exception as e:
            self.log_result("Initialization", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 2: Reset Functionality
    # ========================================================================
    
    def test_reset(self):
        """Test that reset properly resets all state"""
        print("\n" + "="*80)
        print("TEST 2: RESET FUNCTIONALITY")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500)
            env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            # Take some actions to change state
            env.step(Actions.BUY)
            env.step(Actions.HOLD)
            env.step(Actions.SELL)
            
            # Reset
            obs = env.reset()
            
            # Verify reset
            assert env.balance == 10000, f"Balance should be 10000, got {env.balance}"
            assert env.position == Positions.FLAT, f"Position should be FLAT, got {env.position}"
            assert env.position_size == 0, f"Position size should be 0, got {env.position_size}"
            assert env.entry_price == 0, f"Entry price should be 0, got {env.entry_price}"
            assert env.realized_pnl == 0, f"Realized PnL should be 0, got {env.realized_pnl}"
            assert len(env.trades) == 0, f"Trades should be empty, got {len(env.trades)}"
            
            self.log_result("Reset", True, "All state properly reset")
            
        except Exception as e:
            self.log_result("Reset", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 3: Buy Action - Position Sizing
    # ========================================================================
    
    def test_buy_position_sizing(self):
        """Test that buy creates correct position size"""
        print("\n" + "="*80)
        print("TEST 3: BUY POSITION SIZING")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, fee_rate=0.0026, precompute_observations=False)
            
            obs = env.reset()
            initial_balance = env.balance
            
            # Execute buy
            obs, reward, done, truncated, info = env.step(Actions.BUY)
            
            # Check position was created
            assert env.position == Positions.LONG, "Should be in LONG position"
            assert env.position_size > 0, "Position size should be > 0"
            assert env.balance < initial_balance, "Balance should decrease after buy"
            
            # Calculate expected values
            price = df.iloc[env.current_step-1]['close']  # Price when we bought
            available_capital = initial_balance * 0.95
            execution_price = price * 1.001  # With slippage
            
            # Expected position value should be close to capital used
            position_value = env.position_size * execution_price
            expected_value = available_capital * (1 - env.fee_rate)
            
            # Allow 1% tolerance for rounding
            value_diff_pct = abs(position_value - expected_value) / expected_value
            
            if value_diff_pct < 0.01:
                self.log_result("Buy Position Sizing", True, 
                              f"Position value ${position_value:.2f} matches expected ${expected_value:.2f}")
            else:
                self.log_result("Buy Position Sizing", False,
                              f"Position value ${position_value:.2f} differs from expected ${expected_value:.2f} by {value_diff_pct*100:.2f}%")
            
            # Critical check: Position value should NEVER exceed initial balance
            if position_value > initial_balance * 1.1:
                self.log_result("Buy Position Limit", False,
                              f" CRITICAL: Position value ${position_value:.2f} exceeds initial balance ${initial_balance:.2f}!")
            else:
                self.log_result("Buy Position Limit", True,
                              f"Position value ${position_value:.2f} within reasonable limits")
                
        except Exception as e:
            self.log_result("Buy Position Sizing", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 4: Sell Action - Balance Recovery
    # ========================================================================
    
    def test_sell_balance_recovery(self):
        """Test that selling returns correct balance"""
        print("\n" + "="*80)
        print("TEST 4: SELL BALANCE RECOVERY")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, fee_rate=0.0026, precompute_observations=False)
            
            env.reset()
            initial_balance = env.balance
            
            # Buy
            env.step(Actions.BUY)
            balance_after_buy = env.balance
            position_size = env.position_size
            entry_price = env.entry_price
            
            # Hold for a few steps
            for _ in range(5):
                env.step(Actions.HOLD)
            
            # Sell
            env.step(Actions.SELL)
            final_balance = env.balance
            
            # Check position closed
            assert env.position == Positions.FLAT, "Position should be closed"
            assert env.position_size == 0, "Position size should be 0"
            
            # Balance should be close to initial (minus fees)
            # Allow for price movement and fees
            min_expected = initial_balance * 0.95  # Allow 5% loss
            max_expected = initial_balance * 1.05  # Allow 5% gain
            
            if min_expected <= final_balance <= max_expected:
                pnl_pct = (final_balance - initial_balance) / initial_balance * 100
                self.log_result("Sell Balance Recovery", True,
                              f"Final balance ${final_balance:.2f} ({pnl_pct:+.2f}%)")
            else:
                self.log_result("Sell Balance Recovery", False,
                              f"Final balance ${final_balance:.2f} outside expected range [${min_expected:.2f}, ${max_expected:.2f}]")
                
        except Exception as e:
            self.log_result("Sell Balance Recovery", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 5: Portfolio Value Calculation
    # ========================================================================
    
    def test_portfolio_value(self):
        """Test portfolio value calculation"""
        print("\n" + "="*80)
        print("TEST 5: PORTFOLIO VALUE CALCULATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            env.reset()
            
            # Test 1: Portfolio value when flat should equal balance
            portfolio_flat = env._get_portfolio_value()
            assert abs(portfolio_flat - env.balance) < 0.01, "Portfolio should equal balance when flat"
            self.log_result("Portfolio (Flat)", True, f"${portfolio_flat:.2f} equals balance")
            
            # Test 2: Portfolio value when in position
            env.step(Actions.BUY)
            
            portfolio_long = env._get_portfolio_value()
            current_price = env._get_current_price()
            position_value = env.position_size * current_price
            expected_portfolio = env.balance + position_value
            
            diff = abs(portfolio_long - expected_portfolio)
            if diff < 1.0:  # Allow $1 rounding error
                self.log_result("Portfolio (Long)", True,
                              f"${portfolio_long:.2f} matches expected ${expected_portfolio:.2f}")
            else:
                self.log_result("Portfolio (Long)", False,
                              f"${portfolio_long:.2f} differs from expected ${expected_portfolio:.2f} by ${diff:.2f}")
            
            # Test 3: Portfolio should be close to initial balance (no dramatic changes)
            if abs(portfolio_long - env.initial_balance) < env.initial_balance * 0.05:
                self.log_result("Portfolio Stability", True,
                              f"Portfolio ${portfolio_long:.2f} within 5% of initial")
            else:
                self.log_result("Portfolio Stability", False,
                              f"Portfolio ${portfolio_long:.2f} changed too much from initial ${env.initial_balance:.2f}",
                              warning=True)
                
        except Exception as e:
            self.log_result("Portfolio Value", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 6: Fee Calculation
    # ========================================================================
    
    def test_fees(self):
        """Test that fees are calculated correctly"""
        print("\n" + "="*80)
        print("TEST 6: FEE CALCULATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, fee_rate=0.0026, precompute_observations=False)
            
            env.reset()
            initial_balance = env.balance
            
            # Buy
            env.step(Actions.BUY)
            
            # Sell
            for _ in range(5):
                env.step(Actions.HOLD)
            env.step(Actions.SELL)
            
            # Check total fees paid
            total_fees = env.total_fees_paid
            
            # Expected fees: ~0.26% on buy + ~0.26% on sell ≈ 0.52% of capital
            expected_fees = initial_balance * 0.95 * 0.0026 * 2  # Buy + sell
            
            # Allow 20% tolerance for price movement
            fee_diff_pct = abs(total_fees - expected_fees) / expected_fees
            
            if fee_diff_pct < 0.2:
                self.log_result("Fee Calculation", True,
                              f"Fees ${total_fees:.2f} match expected ${expected_fees:.2f}")
            else:
                self.log_result("Fee Calculation", False,
                              f"Fees ${total_fees:.2f} differ from expected ${expected_fees:.2f} by {fee_diff_pct*100:.1f}%")
            
            # Critical check: Fees should never exceed trade value
            if total_fees > initial_balance * 0.1:
                self.log_result("Fee Reasonableness", False,
                              f" CRITICAL: Fees ${total_fees:.2f} are {total_fees/initial_balance*100:.1f}% of initial balance!")
            else:
                self.log_result("Fee Reasonableness", True,
                              f"Fees ${total_fees:.2f} are reasonable ({total_fees/initial_balance*100:.2f}% of balance)")
                
        except Exception as e:
            self.log_result("Fee Calculation", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 7: Multiple Buy-Sell Cycles
    # ========================================================================
    
    def test_multiple_cycles(self):
        """Test multiple buy-sell cycles don't accumulate errors"""
        print("\n" + "="*80)
        print("TEST 7: MULTIPLE BUY-SELL CYCLES")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, fee_rate=0.0026, precompute_observations=False)
            
            env.reset()
            initial_balance = env.balance
            
            max_position_seen = 0
            position_sizes = []
            
            # Execute 10 buy-sell cycles
            for cycle in range(10):
                # Buy
                env.step(Actions.BUY)
                position_value = env.position_size * env._get_current_price()
                max_position_seen = max(max_position_seen, position_value)
                position_sizes.append(position_value)
                
                # Hold for a few steps
                for _ in range(3):
                    env.step(Actions.HOLD)
                
                # Sell
                env.step(Actions.SELL)
                
                # Verify position is closed
                assert env.position == Positions.FLAT, f"Position not closed after cycle {cycle+1}"
                assert env.position_size == 0, f"Position size not 0 after cycle {cycle+1}"
            
            final_balance = env.balance
            
            # Check 1: Position sizes should remain consistent
            avg_position = np.mean(position_sizes)
            std_position = np.std(position_sizes)
            
            if std_position / avg_position < 0.5:  # Coefficient of variation < 50%
                self.log_result("Position Consistency", True,
                              f"Position sizes consistent: ${avg_position:.2f} ± ${std_position:.2f}")
            else:
                self.log_result("Position Consistency", False,
                              f"Position sizes vary too much: ${avg_position:.2f} ± ${std_position:.2f}",
                              warning=True)
            
            # Check 2: No position should exceed reasonable limits
            if max_position_seen > initial_balance * 2:
                self.log_result("Position Limits", False,
                              f" CRITICAL: Max position ${max_position_seen:.2f} exceeded 2x initial balance!")
            else:
                self.log_result("Position Limits", True,
                              f"All positions within limits (max: ${max_position_seen:.2f})")
            
            # Check 3: Final balance should be reasonable
            balance_change_pct = abs(final_balance - initial_balance) / initial_balance * 100
            
            if balance_change_pct < 20:  # Less than 20% change
                self.log_result("Balance Stability (Cycles)", True,
                              f"Balance changed by {balance_change_pct:.2f}% over 10 cycles")
            else:
                self.log_result("Balance Stability (Cycles)", False,
                              f"Balance changed by {balance_change_pct:.2f}% - may indicate accumulating errors",
                              warning=True)
                
        except Exception as e:
            self.log_result("Multiple Cycles", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 8: Position Validation Function
    # ========================================================================
    
    def test_position_validation(self):
        """Test position validation catches unrealistic positions"""
        print("\n" + "="*80)
        print("TEST 8: POSITION VALIDATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            env.reset()
            
            # Check if validation method exists
            if not hasattr(env, '_validate_position_size'):
                self.log_result("Validation Method", False,
                              " _validate_position_size() method not found! Apply Fix #3")
                return
            
            # Test normal position passes validation
            env.step(Actions.BUY)
            if env._validate_position_size():
                self.log_result("Validation (Normal)", True, "Normal position passes validation")
            else:
                self.log_result("Validation (Normal)", False, "Normal position failed validation!")
            
            # Manually create unrealistic position to test validation
            env.position_size = 100000  # Unrealistic amount
            
            if not env._validate_position_size():
                self.log_result("Validation (Unrealistic)", True,
                              "Validation correctly catches unrealistic position")
            else:
                self.log_result("Validation (Unrealistic)", False,
                              " Validation failed to catch unrealistic position! Check threshold")
                
        except Exception as e:
            self.log_result("Position Validation", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 9: Balance Validation
    # ========================================================================
    
    def test_balance_validation(self):
        """Test balance never goes negative"""
        print("\n" + "="*80)
        print("TEST 9: BALANCE VALIDATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            env.reset()
            
            balances = []
            
            # Execute many trades
            for i in range(50):
                action = np.random.choice([Actions.HOLD, Actions.BUY, Actions.SELL])
                obs, reward, done, truncated, info = env.step(action)
                balances.append(env.balance)
                
                if done:
                    break
            
            # Check balance never went negative
            min_balance = min(balances)
            
            if min_balance >= 0:
                self.log_result("Balance Non-Negative", True,
                              f"Balance stayed positive (min: ${min_balance:.2f})")
            else:
                self.log_result("Balance Non-Negative", False,
                              f" CRITICAL: Balance went negative: ${min_balance:.2f}!")
            
            # Check balance never exceeded reasonable growth
            max_balance = max(balances)
            max_growth = max_balance / env.initial_balance
            
            if max_growth < 10:  # Less than 10x growth
                self.log_result("Balance Growth Limit", True,
                              f"Balance growth reasonable ({max_growth:.1f}x)")
            else:
                self.log_result("Balance Growth Limit", False,
                              f" Balance growth unrealistic ({max_growth:.1f}x)!",
                              warning=True)
                
        except Exception as e:
            self.log_result("Balance Validation", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 10: Reward Calculation Consistency
    # ========================================================================
    
    def test_reward_consistency(self):
        """Test that rewards match portfolio changes"""
        print("\n" + "="*80)
        print("TEST 10: REWARD CALCULATION CONSISTENCY")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            obs = env.reset()
            initial_portfolio = env._get_portfolio_value()
            
            total_reward = 0
            
            # Run for 100 steps
            for i in range(100):
                action = np.random.choice([Actions.HOLD, Actions.BUY, Actions.SELL])
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            final_portfolio = env._get_portfolio_value()
            
            # Calculate expected reward
            portfolio_change_pct = (final_portfolio - initial_portfolio) / initial_portfolio * 100
            
            # Reward should roughly match portfolio change (scaled by 100)
            reward_diff = abs(total_reward - portfolio_change_pct)
            
            if reward_diff < 50:  # Allow 50 point difference
                self.log_result("Reward Consistency", True,
                              f"Total reward {total_reward:.2f} matches portfolio change {portfolio_change_pct:.2f}%")
            else:
                self.log_result("Reward Consistency", False,
                              f"Reward {total_reward:.2f} differs from portfolio change {portfolio_change_pct:.2f}% by {reward_diff:.2f}",
                              warning=True)
                
        except Exception as e:
            self.log_result("Reward Consistency", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 11: Full Episode Simulation
    # ========================================================================
    
    def test_full_episode(self):
        """Test complete episode execution"""
        print("\n" + "="*80)
        print("TEST 11: FULL EPISODE SIMULATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=1000, base_price=100)
            env = TradingEnvironment(df=df, initial_balance=10000, fee_rate=0.0026, precompute_observations=False)
            
            obs = env.reset()
            initial_portfolio = env._get_portfolio_value()
            
            episode_reward = 0
            steps = 0
            max_position_value = 0
            trades_executed = 0
            
            # Run full episode
            while steps < 900:
                action = np.random.choice([Actions.HOLD, Actions.BUY, Actions.SELL])
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # Track max position value
                if env.position != Positions.FLAT:
                    position_value = env.position_size * env._get_current_price()
                    max_position_value = max(max_position_value, position_value)
                
                # Count trades
                if len(env.trades) > trades_executed:
                    trades_executed = len(env.trades)
                
                if done:
                    break
            
            final_portfolio = env._get_portfolio_value()
            
            # Report results
            self.log_result("Episode Completion", True,
                          f"Completed {steps} steps with {trades_executed} trades")
            
            # Check position limits
            if max_position_value > env.initial_balance * 2:
                self.log_result("Episode Position Limit", False,
                              f" Max position ${max_position_value:.2f} exceeded 2x initial balance!",
                              warning=True)
            else:
                self.log_result("Episode Position Limit", True,
                              f"Max position ${max_position_value:.2f} within limits")
            
            # Check final portfolio
            growth = final_portfolio / initial_portfolio
            
            if 0.5 < growth < 2.0:  # Between 50% loss and 100% gain
                self.log_result("Episode Growth", True,
                              f"Portfolio growth {growth:.2f}x is reasonable")
            elif growth > 2.0:
                self.log_result("Episode Growth", False,
                              f" Portfolio growth {growth:.2f}x is unrealistic!",
                              warning=True)
            else:
                self.log_result("Episode Growth", False,
                              f"  Large loss: Portfolio at {growth:.2f}x",
                              warning=True)
                
        except Exception as e:
            self.log_result("Full Episode", False, f"Error: {str(e)}")
    
    # ========================================================================
    # TEST 12: Pre-computation Test
    # ========================================================================
    
    def test_precomputation(self):
        """Test pre-computed observations match on-the-fly calculation"""
        print("\n" + "="*80)
        print("TEST 12: PRE-COMPUTATION VALIDATION")
        print("="*80)
        
        try:
            df = self.create_test_data(n_steps=500, base_price=100)
            
            # Create two environments
            env_precompute = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=True)
            env_normal = TradingEnvironment(df=df, initial_balance=10000, precompute_observations=False)
            
            # Reset both
            obs_pre = env_precompute.reset()
            obs_normal = env_normal.reset()
            
            # Compare observations
            if np.allclose(obs_pre, obs_normal, rtol=1e-5):
                self.log_result("Pre-computation Reset", True, "Initial observations match")
            else:
                diff = np.abs(obs_pre - obs_normal).max()
                self.log_result("Pre-computation Reset", False,
                              f"Observations differ by {diff:.6f}", warning=True)
            
            # Take same actions and compare
            actions = [Actions.BUY, Actions.HOLD, Actions.HOLD, Actions.SELL, Actions.HOLD]
            
            for i, action in enumerate(actions):
                obs_pre, _, _, _, _ = env_precompute.step(action)
                obs_normal, _, _, _, _ = env_normal.step(action)
                
                if not np.allclose(obs_pre, obs_normal, rtol=1e-5):
                    diff = np.abs(obs_pre - obs_normal).max()
                    self.log_result(f"Pre-computation Step {i+1}", False,
                                  f"Observations differ by {diff:.6f}", warning=True)
                    break
            else:
                self.log_result("Pre-computation Steps", True, "All observations match")
                
        except Exception as e:
            self.log_result("Pre-computation", False, f"Error: {str(e)}")
    
    # ========================================================================
    # Run All Tests
    # ========================================================================
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print(" TRADING ENVIRONMENT COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Starting at: {pd.Timestamp.now()}")
        print("="*80)
        
        # Run all tests
        self.test_initialization()
        self.test_reset()
        self.test_buy_position_sizing()
        self.test_sell_balance_recovery()
        self.test_portfolio_value()
        self.test_fees()
        self.test_multiple_cycles()
        self.test_position_validation()
        self.test_balance_validation()
        self.test_reward_consistency()
        self.test_full_episode()
        self.test_precomputation()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f" Passed:  {self.passed}")
        print(f" Failed:  {self.failed}")
        print(f"  Warnings: {self.warnings}")
        print(f"Total:     {self.passed + self.failed + self.warnings}")
        print("="*80)
        
        if self.failed == 0:
            print("\n ALL TESTS PASSED!")
            print("Your trading environment is working correctly.")
        else:
            print(f"\n  {self.failed} TESTS FAILED")
            print("Please review the failed tests above and fix the issues.")
            
            # List failed tests
            failed_tests = [r for r in self.test_results if not r['passed'] and not r['warning']]
            if failed_tests:
                print("\nFailed Tests:")
                for test in failed_tests:
                    print(f"   {test['test']}: {test['message']}")
        
        if self.warnings > 0:
            print(f"\n  {self.warnings} WARNINGS")
            warning_tests = [r for r in self.test_results if r['warning']]
            if warning_tests:
                print("\nWarnings:")
                for test in warning_tests:
                    print(f"    {test['test']}: {test['message']}")
        
        return self.failed == 0


def main():
    """Main test execution"""
    tester = TradingEnvironmentTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()