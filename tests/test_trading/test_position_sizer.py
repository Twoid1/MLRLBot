
"""
Test Script for Position Sizer Module - FIXED VERSION
Updated to account for max_risk_per_trade capping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the module to test
from src.trading.position_sizer import (
    PositionSizer, 
    MarketConditions, 
    PositionSizeResult
)


class TestPositionSizer(unittest.TestCase):
    """Test suite for PositionSizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.capital = 10000
        self.sizer = PositionSizer(
            capital=self.capital,
            max_risk_per_trade=0.02,
            max_portfolio_risk=0.06
        )
    
    def test_initialization(self):
        """Test PositionSizer initialization"""
        self.assertEqual(self.sizer.capital, self.capital)
        self.assertEqual(self.sizer.max_risk_per_trade, 0.02)
        self.assertEqual(self.sizer.max_portfolio_risk, 0.06)
        print("  Initialization test passed")
    
    def test_kelly_criterion_basic(self):
        """Test basic Kelly Criterion calculation"""
        result = self.sizer.kelly_criterion_basic(
            win_probability=0.6,
            win_amount=300,
            loss_amount=200,
            kelly_fraction=0.25
        )
        
        # Kelly formula: f* = (p*b - q) / b
        # p=0.6, q=0.4, b=300/200=1.5
        # f* = (0.6*1.5 - 0.4) / 1.5 = 0.333
        # With 25% Kelly: 0.333 * 0.25 = 0.0833
        # BUT capped at max_risk_per_trade (0.02)
        # So expected = 10000 * 0.02 = 200
        
        expected_position_value = self.capital * 0.02  # Capped at max_risk
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertAlmostEqual(result.position_value, expected_position_value, delta=1)
        self.assertEqual(result.method_used, "kelly_basic")
        self.assertGreater(result.confidence, 0)
        
        # Verify the Kelly fraction was correctly calculated before capping
        self.assertAlmostEqual(result.metadata['kelly_f'], 0.02, delta=0.001)
        
        print(f"  Kelly Criterion basic test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Kelly fraction (capped): {result.metadata['kelly_f']:.4f}")
        print(f"   Confidence: {result.confidence:.2%}")
    
    def test_kelly_criterion_continuous(self):
        """Test continuous Kelly Criterion (Thorp's formula)"""
        mean_return = 0.10  # 10% expected return
        variance = 0.04  # 20% volatility squared
        
        result = self.sizer.kelly_criterion_continuous(
            mean_return=mean_return,
            variance=variance,
            kelly_fraction=0.25
        )
        
        # f* = μ / σ² = 0.10 / 0.04 = 2.5
        # With 25% Kelly: 2.5 * 0.25 = 0.625
        # Capped at max_risk_per_trade (0.02)
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertLessEqual(result.position_value, self.capital * 0.02)  # Should be capped
        self.assertEqual(result.method_used, "kelly_continuous")
        
        print(f"  Kelly Criterion continuous test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Sharpe ratio: {result.metadata['sharpe']:.2f}")
    
    def test_kelly_with_multiple_outcomes(self):
        """Test Kelly with multiple possible outcomes"""
        outcomes = [
            (0.2, -0.05),  # 20% chance of -5% loss
            (0.5, 0.02),   # 50% chance of 2% gain
            (0.3, 0.10)    # 30% chance of 10% gain
        ]
        
        result = self.sizer.kelly_with_multiple_outcomes(
            outcomes=outcomes,
            kelly_fraction=0.25
        )
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertEqual(result.method_used, "kelly_multiple")
        self.assertGreater(result.position_value, 0)
        
        print(f"  Kelly with multiple outcomes test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Expected return: ${result.expected_return:.2f}")
        print(f"   Number of outcomes: {result.metadata['num_outcomes']}")
    
    def test_optimal_f(self):
        """Test Optimal F calculation"""
        # Historical trade returns
        trade_returns = [150, -100, 200, -80, 120, -60, 180, -50, 250, -120]
        
        result = self.sizer.optimal_f(trade_returns)
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertEqual(result.method_used, "optimal_f")
        self.assertGreater(result.position_value, 0)
        self.assertLessEqual(result.position_value, self.capital * 0.25)  # Should be reasonable
        
        print(f"  Optimal F test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Optimal f: {result.metadata['optimal_f']:.3f}")
        print(f"   TWR: {result.metadata['twr']:.2f}")
    
    def test_risk_parity_sizing(self):
        """Test Risk Parity position sizing"""
        volatilities = {
            'BTC': 0.02,
            'ETH': 0.025,
            'SOL': 0.03
        }
        
        results = self.sizer.risk_parity_sizing(volatilities)
        
        self.assertEqual(len(results), 3)
        
        # Check that lower volatility assets get larger positions
        btc_value = results['BTC'].position_value
        eth_value = results['ETH'].position_value
        sol_value = results['SOL'].position_value
        
        self.assertGreater(btc_value, sol_value)  # BTC has lower vol than SOL
        
        print(f"  Risk Parity sizing test passed")
        for asset, result in results.items():
            print(f"   {asset}: ${result.position_value:.2f} (vol: {volatilities[asset]:.1%})")
    
    def test_maximum_sharpe_sizing(self):
        """Test Maximum Sharpe Ratio position sizing"""
        expected_returns = {
            'BTC': 0.15,
            'ETH': 0.20,
            'SOL': 0.25
        }
        
        volatilities = {
            'BTC': 0.20,
            'ETH': 0.25,
            'SOL': 0.30
        }
        
        # Create correlation matrix
        correlations = pd.DataFrame([
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0]
        ], index=['BTC', 'ETH', 'SOL'], columns=['BTC', 'ETH', 'SOL'])
        
        results = self.sizer.maximum_sharpe_sizing(
            expected_returns, volatilities, correlations
        )
        
        self.assertEqual(len(results), 3)
        
        # Check that total allocation is reasonable
        total_allocation = sum(r.position_value for r in results.values())
        self.assertLessEqual(total_allocation, self.capital)
        
        print(f"  Maximum Sharpe sizing test passed")
        for asset, result in results.items():
            print(f"   {asset}: ${result.position_value:.2f} (weight: {result.metadata['weight']:.2%})")
    
    def test_monte_carlo_sizing(self):
        """Test Monte Carlo simulation sizing"""
        result = self.sizer.monte_carlo_sizing(
            expected_return=0.15,  # 15% annual
            volatility=0.20,  # 20% annual vol
            n_simulations=1000,
            time_horizon=252,
            max_drawdown_limit=0.15
        )
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertEqual(result.method_used, "monte_carlo")
        self.assertGreater(result.position_value, 0)
        
        print(f"  Monte Carlo sizing test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Optimal size: {result.metadata['optimal_size_fraction']:.3f}")
        print(f"   Avg drawdown: {result.metadata.get('avg_drawdown', 0):.2%}")
        print(f"   95% drawdown: {result.metadata.get('percentile_95_dd', 0):.2%}")
    
    def test_ml_based_sizing(self):
        """Test ML-based position sizing"""
        features = {
            'volatility': 0.015,
            'trend_strength': 0.8,
            'volume_ratio': 1.2
        }
        
        result = self.sizer.ml_based_sizing(
            features=features,
            model_confidence=0.75,
            predicted_return=0.08,
            prediction_std=0.12
        )
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertEqual(result.method_used, "ml_based")
        self.assertEqual(result.confidence, 0.75)
        
        print(f"  ML-based sizing test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Model confidence: {result.confidence:.2%}")
        print(f"   Adjustments: {result.adjustments_applied}")
    
    def test_dynamic_sizing(self):
        """Test dynamic position sizing"""
        current_performance = {
            'win_rate': 0.65,
            'avg_win': 200,
            'avg_loss': 150,
            'drawdown': 0.05
        }
        
        market_conditions = MarketConditions(
            volatility=0.02,
            trend_strength=0.7,
            correlation_matrix=pd.DataFrame(),
            market_regime='trending',
            liquidity=0.9,
            spread=0.001,
            volume_profile={}
        )
        
        result = self.sizer.dynamic_sizing(
            current_performance=current_performance,
            market_conditions=market_conditions,
            base_method='kelly'
        )
        
        self.assertIsInstance(result, PositionSizeResult)
        self.assertTrue(result.method_used.startswith("dynamic_"))
        
        print(f"  Dynamic sizing test passed")
        print(f"   Position value: ${result.position_value:.2f}")
        print(f"   Adjustments: {result.adjustments_applied}")
        print(f"   Adjustment factor: {result.metadata['adjustment_factor']:.2f}")
    
    def test_convert_to_shares(self):
        """Test conversion to shares/units"""
        position_value = 1000
        
        # Test with different prices and lot sizes
        shares_btc = self.sizer.convert_to_shares(position_value, 50000, lot_size=0.001)
        shares_eth = self.sizer.convert_to_shares(position_value, 3000, lot_size=0.01)
        shares_penny = self.sizer.convert_to_shares(position_value, 0.5, lot_size=1.0)
        
        self.assertAlmostEqual(shares_btc, 0.020, places=3)  # 1000/50000 = 0.02
        self.assertAlmostEqual(shares_eth, 0.33, places=2)   # 1000/3000 = 0.333, rounded to 0.33
        self.assertEqual(shares_penny, 2000)  # 1000/0.5 = 2000
        
        print(f"  Convert to shares test passed")
        print(f"   BTC: {shares_btc:.6f} (lot size: 0.001)")
        print(f"   ETH: {shares_eth:.3f} (lot size: 0.01)")
        print(f"   Penny: {shares_penny:.0f} (lot size: 1)")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with zero volatility
        result = self.sizer.kelly_criterion_continuous(0.1, 0, 0.25)
        self.assertEqual(result.method_used, "kelly_continuous_default")
        
        # Test with no trade history
        result = self.sizer.optimal_f([])
        self.assertEqual(result.method_used, "optimal_f_default")
        
        # Test with zero loss amount
        result = self.sizer.kelly_criterion_basic(0.6, 100, 0, 0.25)
        self.assertEqual(result.method_used, "kelly_basic_default")
        
        print(f"  Edge cases test passed")
    
    def test_capital_update(self):
        """Test capital update functionality"""
        original_capital = self.sizer.capital
        new_capital = 15000
        
        self.sizer.update_capital(new_capital)
        self.assertEqual(self.sizer.capital, new_capital)
        
        # Position sizes should scale with new capital but still be capped
        result = self.sizer.kelly_criterion_basic(0.6, 300, 200, 0.25)
        
        # With new capital of 15000 and max_risk of 0.02
        # Expected position = 15000 * 0.02 = 300
        expected_position = new_capital * self.sizer.max_risk_per_trade
        self.assertAlmostEqual(result.position_value, expected_position, delta=1)
        
        print(f"  Capital update test passed")
        print(f"   Original capital: ${original_capital:,.2f}")
        print(f"   New capital: ${new_capital:,.2f}")
        print(f"   New position value: ${result.position_value:.2f}")
    
    def test_kelly_without_cap(self):
        """Test Kelly Criterion without risk cap to verify calculation"""
        # Create a sizer with higher max_risk to avoid capping
        high_risk_sizer = PositionSizer(
            capital=10000,
            max_risk_per_trade=0.10,  # 10% instead of 2%
            max_portfolio_risk=0.30
        )
        
        result = high_risk_sizer.kelly_criterion_basic(
            win_probability=0.6,
            win_amount=300,
            loss_amount=200,
            kelly_fraction=0.25
        )
        
        # Now the Kelly fraction won't be capped
        # f* = (0.6*1.5 - 0.4) / 1.5 = 0.333
        # With 25% Kelly: 0.333 * 0.25 = 0.0833
        expected_position = 10000 * 0.0833
        
        self.assertAlmostEqual(result.position_value, expected_position, delta=10)
        
        print(f"  Kelly without cap test passed")
        print(f"   Position value (uncapped): ${result.position_value:.2f}")
        print(f"   Kelly fraction: {result.metadata['kelly_f']:.4f}")


def run_comprehensive_comparison():
    """Run comprehensive comparison of all sizing methods"""
    print("\n" + "="*60)
    print("POSITION SIZING METHODS COMPARISON")
    print("="*60)
    
    sizer = PositionSizer(capital=10000)
    
    # Common parameters
    win_prob = 0.55
    win_amount = 300
    loss_amount = 200
    volatility = 0.02
    
    results = {}
    
    # 1. Kelly Basic
    results['Kelly Basic'] = sizer.kelly_criterion_basic(
        win_prob, win_amount, loss_amount, 0.25
    )
    
    # 2. Kelly Continuous
    results['Kelly Continuous'] = sizer.kelly_criterion_continuous(
        mean_return=0.10, variance=0.04, kelly_fraction=0.25
    )
    
    # 3. Optimal F
    trade_returns = [150, -100, 200, -80, 120, -60, 180, -50, 250, -120]
    results['Optimal F'] = sizer.optimal_f(trade_returns)
    
    # 4. Risk Parity
    rp_results = sizer.risk_parity_sizing({'Asset': volatility})
    results['Risk Parity'] = rp_results['Asset']
    
    # 5. Monte Carlo
    results['Monte Carlo'] = sizer.monte_carlo_sizing(
        expected_return=0.15,
        volatility=0.20,
        n_simulations=500,
        max_drawdown_limit=0.15
    )
    
    # Compare results
    print("\n Sizing Method Comparison:")
    print("-" * 60)
    print(f"{'Method':<20} {'Position Value':<15} {'Risk Amount':<15} {'Confidence':<10}")
    print("-" * 60)
    
    for method, result in results.items():
        print(f"{method:<20} ${result.position_value:<14.2f} "
              f"${result.risk_amount:<14.2f} {result.confidence:<9.2%}")
    
    # Statistics
    values = [r.position_value for r in results.values()]
    print("\n Statistics:")
    print(f"   Average position: ${np.mean(values):.2f}")
    print(f"   Std deviation: ${np.std(values):.2f}")
    print(f"   Min position: ${np.min(values):.2f}")
    print(f"   Max position: ${np.max(values):.2f}")
    print(f"   Range: ${np.max(values) - np.min(values):.2f}")


def run_scenario_test():
    """Test position sizing under different market scenarios"""
    print("\n" + "="*60)
    print("MARKET SCENARIO TESTING")
    print("="*60)
    
    sizer = PositionSizer(capital=10000)
    
    scenarios = [
        {
            'name': 'Bull Market',
            'performance': {'win_rate': 0.70, 'avg_win': 400, 'avg_loss': 150, 'drawdown': 0.02},
            'conditions': MarketConditions(0.015, 0.9, pd.DataFrame(), 'trending', 1.0, 0.0005, {})
        },
        {
            'name': 'Bear Market',
            'performance': {'win_rate': 0.40, 'avg_win': 200, 'avg_loss': 300, 'drawdown': 0.12},
            'conditions': MarketConditions(0.035, -0.8, pd.DataFrame(), 'trending', 0.7, 0.002, {})
        },
        {
            'name': 'Choppy Market',
            'performance': {'win_rate': 0.50, 'avg_win': 180, 'avg_loss': 180, 'drawdown': 0.06},
            'conditions': MarketConditions(0.025, 0.1, pd.DataFrame(), 'ranging', 0.8, 0.001, {})
        },
        {
            'name': 'Volatile Market',
            'performance': {'win_rate': 0.45, 'avg_win': 500, 'avg_loss': 400, 'drawdown': 0.08},
            'conditions': MarketConditions(0.05, 0.3, pd.DataFrame(), 'volatile', 0.6, 0.003, {})
        }
    ]
    
    print("\n Position Sizing by Market Scenario:")
    print("-" * 70)
    print(f"{'Scenario':<15} {'Kelly':<12} {'Dynamic':<12} {'Risk Adj':<12} {'Recommended':<12}")
    print("-" * 70)
    
    for scenario in scenarios:
        # Kelly sizing
        kelly = sizer.kelly_criterion_basic(
            scenario['performance']['win_rate'],
            scenario['performance']['avg_win'],
            scenario['performance']['avg_loss'],
            0.25
        )
        
        # Dynamic sizing
        dynamic = sizer.dynamic_sizing(
            scenario['performance'],
            scenario['conditions'],
            'kelly'
        )
        
        # Risk-adjusted (considering drawdown)
        risk_adj = kelly.position_value * (1 - scenario['performance']['drawdown'])
        
        # Recommended (most conservative)
        recommended = min(kelly.position_value, dynamic.position_value, risk_adj)
        
        print(f"{scenario['name']:<15} ${kelly.position_value:<11.2f} "
              f"${dynamic.position_value:<11.2f} ${risk_adj:<11.2f} "
              f"${recommended:<11.2f}")
    
    print("\n Key Insights:")
    print("   - Bull market allows larger positions")
    print("   - Bear market requires conservative sizing")
    print("   - Volatile markets need reduced exposure")
    print("   - Dynamic sizing adapts to conditions")


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" POSITION SIZER UNIT TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPositionSizer)
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print(" UNIT TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n All unit tests passed!")
    else:
        print("\n Some tests failed. Check output above.")
    
    # Run additional tests
    run_comprehensive_comparison()
    run_scenario_test()
    
    print("\n" + "="*60)
    print(" Position Sizer testing complete!")
    print("="*60)