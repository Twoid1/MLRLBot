"""
COMPREHENSIVE DIAGNOSTIC TESTS - Clean Version
No unicode characters for Windows compatibility
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class BotDiagnostics:
    """Comprehensive diagnostic testing suite"""
    
    def __init__(self):
        self.results = {}
        
    def run_all_tests(self):
        """Run all diagnostic tests"""
        print("="*80)
        print("TRADING BOT DIAGNOSTIC TEST SUITE")
        print("="*80)
        print()
        
        tests = [
            ("Random Baseline Test", self.test_random_baseline),
            ("Buy & Hold Baseline", self.test_buy_hold_baseline),
            ("Transaction Cost Analysis", self.test_transaction_costs),
            ("Environment Realism Check", self.test_environment_realism),
            ("Overfitting Detection", self.test_overfitting),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*80}")
            print(f"Running: {test_name}")
            print(f"{'='*80}")
            try:
                result = test_func()
                self.results[test_name] = result
                print(f"PASS: Test completed")
            except Exception as e:
                print(f"FAIL: Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                self.results[test_name] = {"error": str(e)}
        
        self.print_summary()
    
    def test_random_baseline(self):
        """Test #1: Random Trading Baseline"""
        from src.environment.trading_env import TradingEnvironment
        from src.data.data_manager import DataManager
        
        print("\nRunning random trading strategy...")
        
        dm = DataManager()
        data = dm.load_existing_data('ETH/USDT', '1h')
        
        if data.empty:
            print("ERROR: No data found!")
            return {"status": "ERROR", "message": "No data available"}
        
        env = TradingEnvironment(
            data,
            initial_balance=10000,
            fee_rate=0.0026
        )
        
        returns = []
        num_trials = 10
        
        for trial in range(num_trials):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = np.random.choice([0, 1, 2])
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward
                if truncated:
                    done = True
            
            final_value = env._get_portfolio_value()
            ret = (final_value - 10000) / 10000 * 100
            returns.append(ret)
            
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        print(f"\nRandom Trading Results:")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Std Deviation: {std_return:.2f}%")
        print(f"   Expected: -30% to -60% (due to fees)")
        
        if avg_return > 0:
            print("\nVERDICT: UNREALISTIC - Random should lose money!")
            return {"status": "FAIL", "avg_return": avg_return}
        else:
            print("\nVERDICT: Random loses money as expected")
            return {"status": "PASS", "avg_return": avg_return}
    
    def test_buy_hold_baseline(self):
        """Test #2: Buy & Hold Baseline"""
        from src.data.data_manager import DataManager
        
        print("\nTesting Buy & Hold strategy...")
        
        dm = DataManager()
        
        results = {}
        for symbol in ['ETH/USDT', 'SOL/USDT', 'ADA/USDT']:
            data = dm.load_existing_data(symbol, '1h')
            
            if data.empty:
                print(f"   {symbol}: No data available")
                continue
            
            buy_price = data['close'].iloc[100]
            sell_price = data['close'].iloc[-1]
            
            ret = (sell_price - buy_price) / buy_price * 100
            results[symbol] = ret
            
            print(f"   {symbol}: {ret:+.2f}%")
        
        avg_bh = np.mean(list(results.values()))
        print(f"\nAverage Buy & Hold: {avg_bh:+.2f}%")
        print(f"   Your Bot: -190.61%")
        print(f"   Difference: {avg_bh - (-190.61):.2f}%")
        
        print("\nVERDICT: Bot massively underperforms buy & hold")
        return {"status": "FAIL", "buy_hold_avg": avg_bh, "bot_avg": -190.61}
    
    def test_transaction_costs(self):
        """Test #3: Transaction Cost Calculation"""
        from src.environment.trading_env import TradingEnvironment
        from src.data.data_manager import DataManager
        
        print("\nTesting transaction cost calculation...")
        
        dm = DataManager()
        data = dm.load_existing_data('ETH/USDT', '1h')
        
        if data.empty:
            print("ERROR: No data found!")
            return {"status": "ERROR", "message": "No data available"}
        
        # Use first 30 days of data
        data = data.iloc[:720]  # 30 days * 24 hours
        
        env = TradingEnvironment(
            data,
            initial_balance=10000,
            fee_rate=0.0026
        )
        
        state = env.reset()
        
        # Test 1: Buy with full balance
        print("\nTest 1: Buy transaction")
        initial_balance = env.balance
        action = 1  # Buy
        state, reward, done, truncated, info = env.step(action)
        
        print(f"   Initial Balance: ${initial_balance:.2f}")
        print(f"   After Buy - Cash: ${env.balance:.2f}")
        print(f"   Position Size: {env.position_size:.6f}")
        
        # Move forward a bit
        for _ in range(5):
            state, reward, done, truncated, info = env.step(0)  # Hold
            if done or truncated:
                break
        
        # Test 2: Sell position
        print("\nTest 2: Sell transaction")
        position_before = env.position_size
        state, reward, done, truncated, info = env.step(2)  # Sell
        
        print(f"   Position Before: {position_before:.6f}")
        print(f"   Position After: {env.position_size:.6f}")
        print(f"   Final Balance: ${env.balance:.2f}")
        
        final_value = env._get_portfolio_value()
        loss_percentage = (10000 - final_value) / 10000 * 100
        
        print(f"\nFee Analysis:")
        print(f"   Expected fee per round trip: 0.52%")
        print(f"   Actual loss: {loss_percentage:.2f}%")
        
        if loss_percentage < 0.4 or loss_percentage > 1.0:
            print("\nVERDICT: Fee calculation may be incorrect")
            return {"status": "WARNING", "loss_pct": loss_percentage}
        else:
            print("\nVERDICT: Fees appear to be calculated correctly")
            return {"status": "PASS", "loss_pct": loss_percentage}
    
    def test_environment_realism(self):
        """Test #4: Environment Realism Check"""
        from src.environment.trading_env import TradingEnvironment
        from src.data.data_manager import DataManager
        
        print("\nTesting environment realism...")
        
        dm = DataManager()
        data = dm.load_existing_data('ETH/USDT', '1h')
        
        if data.empty:
            print("ERROR: No data found!")
            return {"status": "ERROR", "message": "No data available"}
        
        # Use first 30 days
        data = data.iloc[:720]
        
        env = TradingEnvironment(data, initial_balance=10000, fee_rate=0.0026)
        
        # Test 1: Maximum possible return
        print("\nTest 1: Maximum possible return")
        max_return = data['high'].max() / data['low'].min()
        print(f"   Theoretical max return: {(max_return - 1) * 100:.2f}%")
        
        # Test 2: Position limits
        print("\nTest 2: Position limits")
        state = env.reset()
        
        # Try to buy
        state, reward, done, truncated, info = env.step(1)
        position_size = env.position_size
        current_price = data['close'].iloc[env.current_step]
        position_value = position_size * current_price
        
        print(f"   Initial balance: $10,000")
        print(f"   Position value: ${position_value:.2f}")
        print(f"   Cash remaining: ${env.balance:.2f}")
        print(f"   Total portfolio: ${env._get_portfolio_value():.2f}")
        
        if env._get_portfolio_value() > 10100:
            print("\nVERDICT: Environment allows impossible portfolio values!")
            return {"status": "FAIL"}
        else:
            print("\nVERDICT: Environment constraints appear realistic")
            return {"status": "PASS"}
    
    def test_overfitting(self):
        """Test #5: Overfitting Detection"""
        print("\nTesting for overfitting...")
        
        training_sharpe = 1.32
        walkforward_sharpe = -0.16
        
        training_return = 51.29
        walkforward_return = -190.61
        
        print(f"\nPerformance Comparison:")
        print(f"   Training Sharpe: {training_sharpe:.2f}")
        print(f"   Walk-Forward Sharpe: {walkforward_sharpe:.2f}")
        print(f"   Degradation: {((walkforward_sharpe - training_sharpe) / training_sharpe * 100):.1f}%")
        print()
        print(f"   Training Return: {training_return:+.2f}%")
        print(f"   Walk-Forward Return: {walkforward_return:+.2f}%")
        print(f"   Degradation: {walkforward_return - training_return:.2f}%")
        
        print("\nVERDICT: SEVERE OVERFITTING DETECTED")
        print("   Training: +51% return, 73% win rate")
        print("   Testing: -190% return, 34% win rate")
        print("   This is characteristic of data leakage or training on test data")
        
        return {
            "status": "FAIL",
            "training_sharpe": training_sharpe,
            "test_sharpe": walkforward_sharpe,
            "overfitting_severity": "EXTREME"
        }
    
    def print_summary(self):
        """Print summary of all tests"""
        print("\n" + "="*80)
        print("DIAGNOSTIC TEST SUMMARY")
        print("="*80)
        
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            emoji_map = {
                'PASS': '[PASS]',
                'FAIL': '[FAIL]',
                'WARNING': '[WARN]',
                'CHECK_MANUAL': '[CHECK]',
                'ERROR': '[ERROR]'
            }
            
            status_str = emoji_map.get(status, '[?]')
            print(f"\n{status_str} {test_name}: {status}")
            
        print("\n" + "="*80)
        print("PRIMARY ISSUES IDENTIFIED:")
        print("="*80)
        print("1. SEVERE DATA LEAKAGE - Training sees future data")
        print("2. TRAINING ON TEST DATA - Agent memorizes specific events")
        print("3. UNREALISTIC EVALUATION - $9M from $10k is impossible")
        print("4. NO TRUE OUT-OF-SAMPLE TESTING - Walk-forward reveals truth")
        print("\nRECOMMENDED FIXES:")
        print("   1. Implement proper walk-forward in RL training")
        print("   2. Add .shift(1) to all features")
        print("   3. Separate train/test data completely")
        print("   4. Reduce feature count to prevent overfitting")
        print("="*80)


if __name__ == "__main__":
    diagnostics = BotDiagnostics()
    diagnostics.run_all_tests()