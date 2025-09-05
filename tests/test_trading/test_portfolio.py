"""
Test Script for Portfolio Module
Comprehensive testing of portfolio management functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Import the module to test
from src.trading.portfolio import Portfolio, Position, PortfolioSnapshot, PerformanceMetrics


class TestPortfolio(unittest.TestCase):
    """Test suite for Portfolio class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.initial_capital = 10000
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            max_positions=5,
            enable_short=False
        )
    
    def test_initialization(self):
        """Test Portfolio initialization"""
        self.assertEqual(self.portfolio.initial_capital, self.initial_capital)
        self.assertEqual(self.portfolio.cash_balance, self.initial_capital)
        self.assertEqual(self.portfolio.total_value, self.initial_capital)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(self.portfolio.realized_pnl, 0)
        print(" Initialization test passed")
    
    def test_open_position(self):
        """Test opening a position"""
        success = self.portfolio.open_position(
            symbol='BTC/USDT',
            quantity=0.1,
            price=50000,
            position_type='long',
            stop_loss=48000,
            take_profit=52000,
            fees=10
        )
        
        self.assertTrue(success)
        self.assertIn('BTC/USDT', self.portfolio.positions)
        
        position = self.portfolio.positions['BTC/USDT']
        self.assertEqual(position.quantity, 0.1)
        self.assertEqual(position.entry_price, 50000)
        self.assertEqual(position.position_type, 'long')
        
        # Check cash balance
        expected_cash = self.initial_capital - (0.1 * 50000) - 10
        self.assertEqual(self.portfolio.cash_balance, expected_cash)
        
        print(" Open position test passed")
        print(f"   Position: 0.1 BTC @ $50,000")
        print(f"   Cash remaining: ${self.portfolio.cash_balance:.2f}")
    
    def test_close_position(self):
        """Test closing a position with profit"""
        # Open position
        self.portfolio.open_position('BTC/USDT', 0.1, 50000, 'long', fees=10)
        
        # Close with profit
        result = self.portfolio.close_position('BTC/USDT', 52000, fees=10)
        
        self.assertIsNotNone(result)
        self.assertNotIn('BTC/USDT', self.portfolio.positions)
        
        # Check P&L calculation
        expected_gross_pnl = (52000 - 50000) * 0.1  # $200
        expected_net_pnl = expected_gross_pnl - 20  # Minus total fees
        
        self.assertAlmostEqual(result['net_pnl'], expected_net_pnl, places=2)
        self.assertEqual(self.portfolio.realized_pnl, expected_net_pnl)
        
        print(" Close position test passed")
        print(f"   Sold 0.1 BTC @ $52,000")
        print(f"   Net P&L: ${result['net_pnl']:.2f}")
        print(f"   Return: {result['return_pct']:.2%}")
    
    def test_position_limits(self):
        """Test maximum positions limit"""
        # Open max positions
        for i in range(self.portfolio.max_positions):
            success = self.portfolio.open_position(
                symbol=f'ASSET{i}',
                quantity=0.01,
                price=1000
            )
            self.assertTrue(success)
        
        # Try to open one more
        success = self.portfolio.open_position(
            symbol='EXTRA',
            quantity=0.01,
            price=1000
        )
        
        self.assertFalse(success)
        self.assertEqual(len(self.portfolio.positions), self.portfolio.max_positions)
        
        print(f" Position limits test passed (max: {self.portfolio.max_positions})")
    
    def test_insufficient_capital(self):
        """Test handling of insufficient capital"""
        success = self.portfolio.open_position(
            symbol='BTC/USDT',
            quantity=1,  # Trying to buy 1 BTC with $10k
            price=50000,  # Would cost $50k
            fees=100
        )
        
        self.assertFalse(success)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(self.portfolio.cash_balance, self.initial_capital)
        
        print(" Insufficient capital test passed")
    
    def test_update_position_price(self):
        """Test updating position prices"""
        # Open position
        self.portfolio.open_position('BTC/USDT', 0.1, 50000, 'long')
        
        # Update price
        self.portfolio.update_position_price('BTC/USDT', 51000)
        
        position = self.portfolio.positions['BTC/USDT']
        self.assertEqual(position.current_price, 51000)
        self.assertEqual(position.unrealized_pnl, 100)  # (51000-50000)*0.1
        
        print(" Update position price test passed")
        print(f"   Unrealized P&L: ${position.unrealized_pnl:.2f}")
    
    def test_stop_loss_trigger(self):
        """Test stop loss automatic trigger"""
        # Open position with stop loss
        self.portfolio.open_position(
            'BTC/USDT', 0.1, 50000, 'long',
            stop_loss=49000
        )
        
        # Update price below stop loss
        self.portfolio.update_position_price('BTC/USDT', 48900)
        
        # Position should be closed
        self.assertNotIn('BTC/USDT', self.portfolio.positions)
        
        print(" Stop loss trigger test passed")
    
    def test_take_profit_trigger(self):
        """Test take profit automatic trigger"""
        # Open position with take profit
        self.portfolio.open_position(
            'BTC/USDT', 0.1, 50000, 'long',
            take_profit=51000
        )
        
        # Update price above take profit
        self.portfolio.update_position_price('BTC/USDT', 51100)
        
        # Position should be closed
        self.assertNotIn('BTC/USDT', self.portfolio.positions)
        
        print(" Take profit trigger test passed")
    
    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation"""
        # Create some trading history
        self.portfolio.open_position('BTC/USDT', 0.1, 50000, 'long')
        self.portfolio.update_position_price('BTC/USDT', 51000)
        self.portfolio.close_position('BTC/USDT', 51000, fees=10)
        
        self.portfolio.open_position('ETH/USDT', 1, 3000, 'long')
        self.portfolio.update_position_price('ETH/USDT', 2900)
        self.portfolio.close_position('ETH/USDT', 2900, fees=5)
        
        # Calculate metrics
        metrics = self.portfolio.calculate_performance_metrics()
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.win_rate)
        
        print(" Portfolio metrics test passed")
        print(f"   Total return: {metrics.total_return:.2%}")
        print(f"   Win rate: {metrics.win_rate:.2%}")
        print(f"   Max drawdown: {metrics.max_drawdown:.2%}")
    
    def test_position_allocation(self):
        """Test position allocation calculation"""
        # Open multiple positions
        self.portfolio.open_position('BTC/USDT', 0.05, 50000)
        self.portfolio.open_position('ETH/USDT', 1, 3000)
        self.portfolio.open_position('SOL/USDT', 50, 100)
        
        allocation = self.portfolio.get_position_allocation()
        
        # Check allocations sum to 1
        total_allocation = sum(allocation.values())
        self.assertAlmostEqual(total_allocation, 1.0, places=5)
        
        # Check cash allocation
        self.assertIn('cash', allocation)
        
        print(" Position allocation test passed")
        for symbol, weight in allocation.items():
            print(f"   {symbol}: {weight:.2%}")
    
    def test_exposure_summary(self):
        """Test exposure summary calculation"""
        # Open positions
        self.portfolio.open_position('BTC/USDT', 0.05, 50000, 'long')
        self.portfolio.open_position('ETH/USDT', 1, 3000, 'long')
        
        exposure = self.portfolio.get_exposure_summary()
        
        self.assertEqual(exposure['n_positions'], 2)
        self.assertEqual(exposure['n_long'], 2)
        self.assertEqual(exposure['n_short'], 0)
        self.assertGreater(exposure['total_exposure'], 0)
        
        print(" Exposure summary test passed")
        print(f"   Total exposure: ${exposure['total_exposure']:.2f}")
        print(f"   Gross leverage: {exposure['gross_leverage']:.2f}x")
    
    def test_rebalancing(self):
        """Test portfolio rebalancing"""
        # Open initial positions
        self.portfolio.open_position('BTC/USDT', 0.05, 50000)
        self.portfolio.open_position('ETH/USDT', 1, 3000)
        
        # Define target weights
        target_weights = {
            'BTC/USDT': 0.4,
            'ETH/USDT': 0.3,
            'SOL/USDT': 0.2
        }
        
        # Current prices
        prices = {
            'BTC/USDT': 51000,
            'ETH/USDT': 3100,
            'SOL/USDT': 110
        }
        
        # Calculate rebalancing trades
        trades = self.portfolio.rebalance_to_weights(
            target_weights, prices, threshold=0.05
        )
        
        self.assertIsInstance(trades, list)
        
        print(" Rebalancing test passed")
        print(f"   {len(trades)} rebalancing trades needed")
        for trade in trades:
            print(f"   {trade['action']} {trade['quantity']:.4f} {trade['symbol']}")
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        # Create returns data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        returns_data = pd.DataFrame({
            'BTC/USDT': np.random.normal(0.001, 0.02, 100),
            'ETH/USDT': np.random.normal(0.0005, 0.025, 100)
        }, index=dates)
        
        # Open positions
        self.portfolio.open_position('BTC/USDT', 0.1, 50000)
        self.portfolio.open_position('ETH/USDT', 1, 3000)
        
        # Calculate VaR
        var_95 = self.portfolio.get_portfolio_var(returns_data, confidence=0.95)
        
        self.assertGreater(var_95, 0)
        
        print(f" VaR calculation test passed")
        print(f"   95% VaR: ${var_95:.2f}")
    
    def test_beta_calculation(self):
        """Test portfolio beta calculation"""
        # Create market returns
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        market_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 100),
            index=dates
        )
        
        # Calculate beta
        beta = self.portfolio.get_beta_to_market(market_returns)
        
        self.assertIsNotNone(beta)
        
        print(f" Beta calculation test passed")
        print(f"   Portfolio beta: {beta:.2f}")
    
    def test_export_history(self):
        """Test exporting portfolio history"""
        # Create some history
        self.portfolio.open_position('BTC/USDT', 0.1, 50000)
        self.portfolio.close_position('BTC/USDT', 51000)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            self.portfolio.export_history(f.name)
            
            # Read back and verify
            with open(f.name, 'r') as rf:
                data = json.load(rf)
                
                self.assertIn('transactions', data)
                self.assertIn('closed_positions', data)
                self.assertIn('summary', data)
        
        print(" Export history test passed")
    
    def test_position_duration(self):
        """Test position duration tracking"""
        # Open position
        self.portfolio.open_position('BTC/USDT', 0.1, 50000)
        
        # Wait a moment (in real scenario this would be longer)
        import time
        time.sleep(0.1)
        
        position = self.portfolio.positions['BTC/USDT']
        duration = position.duration
        
        self.assertIsInstance(duration, timedelta)
        self.assertGreater(duration.total_seconds(), 0)
        
        print(" Position duration test passed")
        print(f"   Duration: {duration.total_seconds():.2f} seconds")


def run_portfolio_simulation():
    """Run a realistic portfolio simulation"""
    print("\n" + "="*60)
    print("PORTFOLIO SIMULATION")
    print("="*60)
    
    portfolio = Portfolio(initial_capital=10000, max_positions=5)
    
    # Simulate trading day
    trades = [
        ('BTC/USDT', 0.05, 45000, 46000, 'win'),
        ('ETH/USDT', 1.0, 3000, 2850, 'loss'),
        ('SOL/USDT', 20, 100, 105, 'win'),
        ('ADA/USDT', 1000, 1.5, 1.45, 'loss'),
        ('DOT/USDT', 50, 30, 32, 'win')
    ]
    
    print("\n Executing trades...")
    
    for symbol, qty, entry, exit, result in trades:
        # Open position
        portfolio.open_position(symbol, qty, entry, 'long', fees=5)
        print(f"\nOpened {qty} {symbol} @ ${entry}")
        
        # Update and close
        portfolio.update_position_price(symbol, exit)
        closed = portfolio.close_position(symbol, exit, fees=5)
        
        if closed:
            print(f"Closed @ ${exit}")
            print(f"   P&L: ${closed['net_pnl']:.2f} ({result})")
    
    # Final summary
    summary = portfolio.get_summary()
    metrics = portfolio.calculate_performance_metrics()
    
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    print(f"\n Capital:")
    print(f"   Starting: ${portfolio.initial_capital:,.2f}")
    print(f"   Ending: ${summary['value']['total']:,.2f}")
    print(f"   Change: ${summary['pnl']['total']:.2f} ({summary['pnl']['total_return']:.2%})")
    
    print(f"\n Performance:")
    print(f"   Trades: {len(portfolio.closed_positions)}")
    print(f"   Win rate: {metrics.win_rate:.2%}")
    print(f"   Avg win: ${metrics.avg_win:.2f}")
    print(f"   Avg loss: ${metrics.avg_loss:.2f}")
    print(f"   Profit factor: {metrics.profit_factor:.2f}")
    
    print(f"\n Risk:")
    print(f"   Max drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Current drawdown: {metrics.current_drawdown:.2%}")
    print(f"   Sharpe ratio: {metrics.sharpe_ratio:.2f}")


def run_multi_asset_test():
    """Test multi-asset portfolio management"""
    print("\n" + "="*60)
    print("MULTI-ASSET PORTFOLIO TEST")
    print("="*60)
    
    portfolio = Portfolio(initial_capital=50000, max_positions=10)
    
    # Define assets with different characteristics
    assets = [
        {'symbol': 'BTC/USDT', 'qty': 0.2, 'price': 45000, 'vol': 0.02},
        {'symbol': 'ETH/USDT', 'qty': 3, 'price': 3000, 'vol': 0.025},
        {'symbol': 'BNB/USDT', 'qty': 10, 'price': 400, 'vol': 0.022},
        {'symbol': 'SOL/USDT', 'qty': 50, 'price': 100, 'vol': 0.03},
        {'symbol': 'ADA/USDT', 'qty': 5000, 'price': 1.2, 'vol': 0.028}
    ]
    
    print("\n Building multi-asset portfolio...")
    
    # Open all positions
    for asset in assets:
        success = portfolio.open_position(
            asset['symbol'],
            asset['qty'],
            asset['price'],
            'long'
        )
        if success:
            value = asset['qty'] * asset['price']
            print(f"   Added {asset['symbol']}: ${value:,.2f}")
    
    # Get allocation
    allocation = portfolio.get_position_allocation()
    
    print("\n Portfolio Allocation:")
    sorted_allocation = sorted(allocation.items(), key=lambda x: x[1], reverse=True)
    for symbol, weight in sorted_allocation:
        print(f"   {symbol:<10} {weight:>6.2%}")
    
    # Simulate price movements
    print("\n Simulating price movements...")
    
    np.random.seed(42)
    for asset in assets:
        # Generate random return based on volatility
        price_change = np.random.normal(0.001, asset['vol'])
        new_price = asset['price'] * (1 + price_change)
        portfolio.update_position_price(asset['symbol'], new_price)
        
        pct_change = ((new_price / asset['price']) - 1) * 100
        print(f"   {asset['symbol']}: {pct_change:+.2f}%")
    
    # Final metrics
    summary = portfolio.get_summary()
    
    print(f"\n Portfolio Value:")
    print(f"   Initial: ${portfolio.initial_capital:,.2f}")
    print(f"   Current: ${summary['value']['total']:,.2f}")
    print(f"   Unrealized P&L: ${summary['pnl']['unrealized']:,.2f}")
    
    # Get exposure summary
    exposure = portfolio.get_exposure_summary()
    print(f"\n Exposure:")
    print(f"   Total exposure: ${exposure['total_exposure']:,.2f}")
    print(f"   Gross leverage: {exposure['gross_leverage']:.2f}x")
    print(f"   Concentration: {exposure['concentration']:.2%}")


if __name__ == "__main__":
    # Run unit tests
    print("Running Portfolio Unit Tests...")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPortfolio)
    
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
    
    # Run additional tests
    run_portfolio_simulation()
    run_multi_asset_test()
    
    print("\n Portfolio testing complete!")