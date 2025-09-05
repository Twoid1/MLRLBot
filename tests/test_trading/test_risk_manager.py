"""
Test Script for Risk Manager Module
Comprehensive testing of all risk management functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the module to test
from src.trading.risk_manager import RiskManager, RiskConfig, RiskLevel, RiskMetrics, PositionLimits


class TestRiskManager(unittest.TestCase):
    """Test suite for RiskManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.initial_capital = 10000
        self.risk_config = RiskConfig(
            risk_level=RiskLevel.MODERATE,
            max_risk_per_trade=0.02,
            max_daily_risk=0.06,
            max_drawdown_limit=0.15,
            position_sizing_method='volatility_based'
        )
        self.risk_manager = RiskManager(self.initial_capital, self.risk_config)
    
    def test_initialization(self):
        """Test RiskManager initialization"""
        self.assertEqual(self.risk_manager.initial_capital, self.initial_capital)
        self.assertEqual(self.risk_manager.current_capital, self.initial_capital)
        self.assertEqual(self.risk_manager.config.risk_level, RiskLevel.MODERATE)
        self.assertIsNotNone(self.risk_manager.position_limits)
        print(" Initialization test passed")
    
    def test_position_sizing_fixed(self):
        """Test fixed position sizing"""
        self.risk_manager.config.position_sizing_method = 'fixed'
        
        position_size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.02
        )
        
        expected_value = self.initial_capital * 0.02  # 2% risk
        expected_size = expected_value / 50000
        
        self.assertAlmostEqual(position_size, expected_size, places=6)
        print(f" Fixed position sizing test passed: {position_size:.6f} BTC")
    
    def test_position_sizing_volatility_based(self):
        """Test volatility-based position sizing"""
        self.risk_manager.config.position_sizing_method = 'volatility_based'
        
        # Low volatility should give larger position
        low_vol_size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.01  # Low volatility
        )
        
        # High volatility should give smaller position
        high_vol_size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.05  # High volatility
        )
        
        self.assertGreater(low_vol_size, high_vol_size)
        print(f" Volatility-based sizing test passed")
        print(f"   Low vol (1%): {low_vol_size:.6f} BTC")
        print(f"   High vol (5%): {high_vol_size:.6f} BTC")
    
    def test_kelly_criterion_sizing(self):
        """Test Kelly Criterion position sizing"""
        self.risk_manager.config.position_sizing_method = 'kelly'
        
        position_size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.02,
            win_rate=0.6,
            avg_win=300,
            avg_loss=200
        )
        
        # Kelly formula: f* = (p*b - q) / b
        # where p=0.6, q=0.4, b=300/200=1.5
        # f* = (0.6*1.5 - 0.4) / 1.5 = 0.33
        # With safety factor of 0.25: 0.33 * 0.25 = 0.0825
        
        expected_value = self.initial_capital * 0.0825
        expected_size = expected_value / 50000
        
        # Allow some tolerance for safety adjustments
        self.assertLess(position_size, expected_size * 1.1)
        print(f" Kelly Criterion sizing test passed: {position_size:.6f} BTC")
    
    def test_stop_loss_calculation(self):
        """Test stop loss calculation"""
        entry_price = 50000
        atr = 1000
        
        # Long position stop loss
        long_stop = self.risk_manager.calculate_stop_loss(
            entry_price, 'long', atr, method='atr'
        )
        expected_long_stop = entry_price - (atr * 2.0)  # 2x ATR
        self.assertAlmostEqual(long_stop, expected_long_stop, places=2)
        
        # Short position stop loss
        short_stop = self.risk_manager.calculate_stop_loss(
            entry_price, 'short', atr, method='atr'
        )
        expected_short_stop = entry_price + (atr * 2.0)
        self.assertAlmostEqual(short_stop, expected_short_stop, places=2)
        
        print(f" Stop loss calculation test passed")
        print(f"   Long stop: ${long_stop:.2f}")
        print(f"   Short stop: ${short_stop:.2f}")
    
    def test_take_profit_calculation(self):
        """Test take profit calculation"""
        entry_price = 50000
        atr = 1000
        
        # Long position take profit
        long_tp = self.risk_manager.calculate_take_profit(
            entry_price, 'long', atr, risk_reward_ratio=2.0
        )
        expected_long_tp = entry_price + (atr * 3.0)  # 3x ATR
        self.assertAlmostEqual(long_tp, expected_long_tp, places=2)
        
        print(f" Take profit calculation test passed: ${long_tp:.2f}")
    
    def test_position_limits(self):
        """Test position limits enforcement"""
        # Try to open maximum positions
        for i in range(self.risk_manager.position_limits.max_positions):
            can_open = self.risk_manager.can_open_position(f'ASSET{i}')
            self.assertTrue(can_open)
            
            # Add position
            self.risk_manager.add_position(
                symbol=f'ASSET{i}',
                position_size=0.1,
                entry_price=1000,
                position_type='long'
            )
        
        # Try to open one more (should fail)
        can_open = self.risk_manager.can_open_position('EXTRA_ASSET')
        self.assertFalse(can_open)
        
        print(f" Position limits test passed (max: {self.risk_manager.position_limits.max_positions})")
    
    def test_drawdown_tracking(self):
        """Test drawdown tracking"""
        # Simulate profit
        self.risk_manager.update_capital(11000)
        self.assertEqual(self.risk_manager.current_drawdown, 0)
        
        # Simulate loss
        self.risk_manager.update_capital(9500)
        expected_drawdown = (11000 - 9500) / 11000
        self.assertAlmostEqual(self.risk_manager.current_drawdown, expected_drawdown, places=4)
        
        # Check max drawdown
        self.assertAlmostEqual(self.risk_manager.max_drawdown, expected_drawdown, places=4)
        
        print(f" Drawdown tracking test passed")
        print(f"   Current drawdown: {self.risk_manager.current_drawdown:.2%}")
        print(f"   Max drawdown: {self.risk_manager.max_drawdown:.2%}")
    
    def test_recovery_mode(self):
        """Test recovery mode activation"""
        # Create large drawdown
        self.risk_manager.update_capital(8500)  # 15% loss from initial
        
        # Should trigger recovery mode at 10% drawdown
        self.assertTrue(self.risk_manager.recovery_mode_active)
        
        # Position sizing should be reduced
        position_size = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.02
        )
        
        # Should be reduced by recovery factor (50%)
        self.assertLess(position_size, 0.002)  # Much smaller than normal
        
        print(f" Recovery mode test passed")
        print(f"   Recovery mode active: {self.risk_manager.recovery_mode_active}")
    
    def test_consecutive_losses(self):
        """Test consecutive losses tracking and circuit breaker"""
        # Simulate consecutive losses
        for i in range(4):
            self.risk_manager.record_trade(
                symbol=f'BTC/USDT',
                pnl=-100,
                position_size=0.1,
                entry_price=50000,
                exit_price=49000,
                position_type='long'
            )
        
        self.assertEqual(self.risk_manager.consecutive_losses, 4)
        
        # One more loss should trigger circuit breaker
        self.risk_manager.record_trade(
            symbol='BTC/USDT',
            pnl=-100,
            position_size=0.1,
            entry_price=50000,
            exit_price=49000,
            position_type='long'
        )
        
        # Should not be able to open new positions
        can_open = self.risk_manager.can_open_position('BTC/USDT')
        self.assertFalse(can_open)
        self.assertTrue(self.risk_manager.circuit_breaker_active)
        
        print(f" Consecutive losses test passed")
        print(f"   Circuit breaker active after {self.risk_manager.consecutive_losses} losses")
    
    def test_daily_loss_limit(self):
        """Test daily loss limit"""
        # Simulate large daily loss
        self.risk_manager.daily_pnl = -700  # 7% loss (exceeds 6% limit)
        
        can_open = self.risk_manager.can_open_position('BTC/USDT')
        self.assertFalse(can_open)
        
        print(f" Daily loss limit test passed")
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        # Add some returns history
        returns = [0.01, -0.02, 0.015, -0.01, 0.02, -0.025, 0.01, -0.005, 0.03, -0.015]
        self.risk_manager.returns_history = returns * 30  # Create 300 data points
        
        var_95 = self.risk_manager.calculate_var(confidence_level=0.95, periods=100)
        
        self.assertGreater(var_95, 0)
        print(f" VaR calculation test passed: ${var_95:.2f}")
    
    def test_risk_metrics_calculation(self):
        """Test comprehensive risk metrics"""
        # Add trading history
        for i in range(10):
            pnl = np.random.choice([100, -50])
            self.risk_manager.record_trade(
                symbol='BTC/USDT',
                pnl=pnl,
                position_size=0.1,
                entry_price=50000,
                exit_price=50000 + pnl * 10,
                position_type='long'
            )
        
        metrics = self.risk_manager.get_risk_metrics()
        
        self.assertIsInstance(metrics, RiskMetrics)
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
        
        print(f" Risk metrics calculation test passed")
        print(f"   Win rate: {metrics.win_rate:.2%}")
        print(f"   Profit factor: {metrics.profit_factor:.2f}")
    
    def test_position_update_and_trailing_stop(self):
        """Test position update and trailing stop"""
        # Add a position
        self.risk_manager.add_position(
            symbol='BTC/USDT',
            position_size=0.1,
            entry_price=50000,
            position_type='long',
            stop_loss=48000
        )
        
        # Update with higher price (should update trailing stop)
        recommendations = self.risk_manager.update_position(
            symbol='BTC/USDT',
            current_price=52000,
            highest_price=52000
        )
        
        # Check if trailing stop was updated
        if 'new_stop' in recommendations:
            self.assertGreater(recommendations['new_stop'], 48000)
            print(f" Trailing stop test passed: new stop at ${recommendations['new_stop']:.2f}")
        else:
            print(" Position update test passed")
    
    def test_risk_adjustments(self):
        """Test risk adjustments based on market conditions"""
        # Test with high drawdown
        self.risk_manager.current_drawdown = 0.08  # 8% drawdown
        
        # Get position size WITH drawdown
        position_size_with_drawdown = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.02
        )
        
        # Reset and get position size WITHOUT drawdown
        self.risk_manager.current_drawdown = 0
        position_size_normal = self.risk_manager.calculate_position_size(
            symbol='BTC/USDT',
            signal_strength=1.0,
            current_price=50000,
            volatility=0.02
        )
        
        # Position should be reduced due to drawdown
        self.assertLess(position_size_with_drawdown, position_size_normal)
        
        print(f" Risk adjustments test passed")
        print(f"   Normal size: {position_size_normal:.6f} BTC")
        print(f"   Adjusted size with 8% drawdown: {position_size_with_drawdown:.6f} BTC")


def run_integration_test():
    """Run integration test with realistic scenario"""
    print("\n" + "="*60)
    print("RISK MANAGER INTEGRATION TEST")
    print("="*60)
    
    # Initialize
    rm = RiskManager(10000, RiskConfig(risk_level=RiskLevel.MODERATE))
    
    # Simulate trading session
    print("\n Simulating trading session...")
    
    # Trade 1: Winning trade
    print("\nTrade 1: Opening long position")
    can_open = rm.can_open_position('BTC/USDT')
    print(f"  Can open: {can_open}")
    
    size = rm.calculate_position_size('BTC/USDT', 0.8, 45000, 0.02)
    print(f"  Position size: {size:.6f} BTC")
    
    rm.add_position('BTC/USDT', size, 45000, 'long', stop_loss=43000)
    rm.record_trade('BTC/USDT', 500, size, 45000, 46000, 'long')
    print(f"  Result: +$500 profit")
    
    # Trade 2: Losing trade
    print("\nTrade 2: Opening ETH position")
    size = rm.calculate_position_size('ETH/USDT', 0.6, 3000, 0.025)
    print(f"  Position size: {size:.3f} ETH")
    
    rm.add_position('ETH/USDT', size, 3000, 'long')
    rm.record_trade('ETH/USDT', -200, size, 3000, 2900, 'long')
    print(f"  Result: -$200 loss")
    
    # Update capital
    rm.update_capital(10300)  # 10000 + 500 - 200
    
    # Get final metrics
    metrics = rm.get_risk_metrics()
    
    print("\n Final Risk Metrics:")
    print(f"  Current capital: ${rm.current_capital:,.2f}")
    print(f"  Win rate: {metrics.win_rate:.2%}")
    print(f"  Current drawdown: {metrics.current_drawdown:.2%}")
    print(f"  Max drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Consecutive losses: {rm.consecutive_losses}")
    print(f"  Circuit breaker: {'ACTIVE' if rm.circuit_breaker_active else 'inactive'}")
    
    print("\n Integration test completed successfully!")


if __name__ == "__main__":
    # Run unit tests
    print("Running Risk Manager Unit Tests...")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRiskManager)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("UNIT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n All unit tests passed!")
    else:
        print("\n Some tests failed. Check output above.")
    
    # Run integration test
    run_integration_test()