"""
Test suite for Backtester module
Tests backtesting engine functionality and integration with all components
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.backtester import (
    Backtester, BacktestConfig, BacktestResults, run_backtest, optimize_strategy
)
from src.data.data_manager import DataManager
from src.features.feature_engineer import FeatureEngineer


class TestBacktestConfig(unittest.TestCase):
    """Test BacktestConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BacktestConfig()
        
        self.assertEqual(config.initial_capital, 10000.0)
        self.assertEqual(config.position_size, 0.1)
        self.assertEqual(config.max_positions, 3)
        self.assertEqual(config.commission, 0.0026)
        self.assertTrue(config.use_risk_manager)
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = BacktestConfig(
            symbols=['BTC_USDT', 'ETH_USDT'],
            initial_capital=50000,
            strategy_type='ml_only',
            walk_forward=True,
            n_splits=10
        )
        
        self.assertEqual(len(config.symbols), 2)
        self.assertEqual(config.initial_capital, 50000)
        self.assertEqual(config.strategy_type, 'ml_only')
        self.assertTrue(config.walk_forward)
        self.assertEqual(config.n_splits, 10)


class TestBacktester(unittest.TestCase):
    """Test main Backtester class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Create temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Generate sample data
        cls.sample_data = cls._generate_sample_data()
        cls.sample_features = cls._generate_sample_features()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        # Remove temporary directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _generate_sample_data(cls):
        """Generate sample OHLCV data"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
        
        data = {}
        for symbol in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']:
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)
            
            close = 1000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
            
            df = pd.DataFrame({
                'open': close * (1 + np.random.randn(len(dates)) * 0.001),
                'high': close * (1 + np.abs(np.random.randn(len(dates)) * 0.002)),
                'low': close * (1 - np.abs(np.random.randn(len(dates)) * 0.002)),
                'close': close,
                'volume': np.random.randint(100000, 1000000, len(dates))
            }, index=dates)
            
            # Ensure OHLC relationships
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            data[symbol] = {'1h': df}
        
        return data
    
    @classmethod
    def _generate_sample_features(cls):
        """Generate sample features"""
        features = {}
        
        for symbol, timeframe_data in cls.sample_data.items():
            df = timeframe_data['1h']
            
            # Simple features for testing
            feature_df = pd.DataFrame(index=df.index)
            
            # Price features
            feature_df['returns'] = df['close'].pct_change()
            feature_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            feature_df['high_low_ratio'] = df['high'] / df['low']
            feature_df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            feature_df['sma_10'] = df['close'].rolling(10).mean()
            feature_df['sma_20'] = df['close'].rolling(20).mean()
            feature_df['sma_ratio'] = feature_df['sma_10'] / feature_df['sma_20']
            
            # Volume features
            feature_df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # Technical indicators
            feature_df['rsi'] = 50 + np.random.randn(len(df)) * 10  # Simplified RSI
            feature_df['macd'] = np.random.randn(len(df)) * 0.001
            
            # Fill NaN values
            feature_df = feature_df.fillna(method='ffill').fillna(0)
            
            features[symbol] = feature_df
        
        return features
    
    def setUp(self):
        """Set up each test"""
        self.config = BacktestConfig(
            symbols=['BTC_USDT', 'ETH_USDT'],
            timeframes=['1h'],
            start_date='2023-01-01',
            end_date='2023-06-30',
            initial_capital=10000,
            strategy_type='ml_only',
            use_ml_predictions=False,  # Disable ML for basic tests
            use_rl_agent=False,  # Disable RL for basic tests
            walk_forward=False,
            save_results=False,
            verbose=False
        )
        
        self.backtester = Backtester(self.config)
    
    def test_initialization(self):
        """Test backtester initialization"""
        self.assertIsNotNone(self.backtester.data_manager)
        self.assertIsNotNone(self.backtester.portfolio)
        self.assertIsNotNone(self.backtester.position_sizer)
        self.assertIsNotNone(self.backtester.executor)
        
        if self.config.use_risk_manager:
            self.assertIsNotNone(self.backtester.risk_manager)
    
    def test_prepare_data(self):
        """Test data preparation"""
        # Mock data loading
        self.backtester.data_manager.load_existing_data = lambda s, t: self.sample_data.get(s, {}).get(t, pd.DataFrame())
        
        prepared_data = self.backtester.prepare_data()
        
        self.assertIn('BTC_USDT', prepared_data)
        self.assertIn('ETH_USDT', prepared_data)
        self.assertIn('1h', prepared_data['BTC_USDT'])
        
        # Check date filtering
        for symbol in prepared_data:
            if '1h' in prepared_data[symbol]:
                df = prepared_data[symbol]['1h']
                self.assertTrue(all(df.index >= pd.to_datetime(self.config.start_date)))
                self.assertTrue(all(df.index <= pd.to_datetime(self.config.end_date)))
    
    def test_calculate_features(self):
        """Test feature calculation"""
        # Use sample data
        features = self.backtester.calculate_features(self.sample_data)
        
        self.assertIn('BTC_USDT', features)
        self.assertIn('ETH_USDT', features)
        
        # Check features exist and have correct shape
        for symbol in features:
            feature_df = features[symbol]
            data_df = self.sample_data[symbol]['1h']
            
            # Should have same number of rows
            self.assertEqual(len(feature_df), len(data_df))
            
            # Should have multiple features
            self.assertGreater(feature_df.shape[1], 0)
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        # Test data
        symbol = 'BTC_USDT'
        data = self.sample_data[symbol]['1h'][:100]
        features = self.sample_features[symbol][:100]
        
        # Generate signal
        signal = self.backtester._generate_signal(
            symbol=symbol,
            data=data,
            features=features,
            current_idx=99
        )
        
        # Should return -1, 0, or 1
        self.assertIn(signal, [-1, 0, 1])
    
    def test_position_opening(self):
        """Test position opening logic"""
        position = self.backtester._open_position(
            symbol='BTC_USDT',
            price=1000.0,
            time=datetime.now(),
            signal_strength=1.0
        )
        
        if position:
            self.assertEqual(position['symbol'], 'BTC_USDT')
            self.assertEqual(position['entry_price'], 1000.0)
            self.assertGreater(position['quantity'], 0)
            self.assertIsNotNone(position['stop_loss'])
            self.assertIsNotNone(position['take_profit'])
            
            # Check stop loss and take profit levels
            self.assertLess(position['stop_loss'], position['entry_price'])
            self.assertGreater(position['take_profit'], position['entry_price'])
    
    def test_position_closing(self):
        """Test position closing logic"""
        # Create a position
        position = {
            'symbol': 'BTC_USDT',
            'entry_time': datetime.now() - timedelta(hours=1),
            'entry_price': 1000.0,
            'quantity': 0.1,
            'stop_loss': 950.0,
            'take_profit': 1100.0
        }
        
        # Close position
        trade = self.backtester._close_position(
            position=position,
            price=1050.0,
            time=datetime.now(),
            reason='signal'
        )
        
        if trade:
            self.assertEqual(trade['symbol'], 'BTC_USDT')
            self.assertEqual(trade['entry_price'], 1000.0)
            self.assertAlmostEqual(trade['exit_price'], 1050.0 * (1 + self.config.slippage), places=2)
            self.assertIsNotNone(trade['pnl'])
            self.assertIsNotNone(trade['return_pct'])
    
    def test_risk_limits(self):
        """Test risk limit checking"""
        # Check with normal conditions first
        can_trade = self.backtester._check_risk_limits('BTC_USDT', 1000.0)
        self.assertTrue(can_trade)
        
        # Instead of directly setting current_drawdown property, create a drawdown condition:
        
        # 1. Store original balance
        original_balance = self.backtester.portfolio.balance
        
        # 2. Add a peak balance to history (this will be the high water mark)
        high_balance = original_balance * 1.3  # 30% higher than current
        peak_time = datetime.now() - timedelta(days=1)
        self.backtester.portfolio.balance_history.append({
            'timestamp': peak_time,
            'balance': high_balance,
            'unrealized_pnl': 0
        })
        
        # 3. Update current balance to create desired drawdown (25%)
        # If we want 25% drawdown from peak, current should be 75% of peak
        new_balance = high_balance * 0.75
        self.backtester.portfolio.update_balance(new_balance)
        
        # 4. Now test the risk limits with the drawdown in place
        can_trade = self.backtester._check_risk_limits('BTC_USDT', 1000.0)
        
        # 5. Assert based on your max_drawdown setting
        if self.config.max_drawdown < 0.25:
            self.assertFalse(can_trade)
        else:
            self.assertTrue(can_trade)
        
        # 6. Clean up (restore original balance for other tests)
        self.backtester.portfolio.update_balance(original_balance)
        # Remove the added history entry
        self.backtester.portfolio.balance_history.pop()
    
    def test_exit_conditions(self):
        """Test stop-loss and take-profit checking"""
        position = {
            'symbol': 'BTC_USDT',
            'entry_price': 1000.0,
            'stop_loss': 950.0,
            'take_profit': 1100.0
        }
        
        # Test stop loss
        should_exit = self.backtester._check_exit_conditions(position, 940.0)
        self.assertTrue(should_exit)
        
        # Test take profit
        should_exit = self.backtester._check_exit_conditions(position, 1110.0)
        self.assertTrue(should_exit)
        
        # Test normal price
        should_exit = self.backtester._check_exit_conditions(position, 1000.0)
        self.assertFalse(should_exit)
    
    def test_simulate_trading(self):
        """Test trading simulation for single symbol"""
        symbol = 'BTC_USDT'
        data = self.sample_data[symbol]['1h'][:500]
        features = self.sample_features[symbol][:500]
        
        results = self.backtester._simulate_trading(
            symbol=symbol,
            data=data,
            features=features
        )
        
        self.assertIn('trades', results)
        self.assertIn('positions', results)
        self.assertIn('signals', results)
        
        # Should have generated some signals
        self.assertGreater(len(results['signals']), 0)
        
        # Check equity curve was updated
        self.assertGreater(len(self.backtester.equity_curve), 0)
    
    def test_standard_backtest(self):
        """Test standard backtest execution"""
        # Mock data loading
        self.backtester.data_manager.load_existing_data = lambda s, t: self.sample_data.get(s, {}).get(t, pd.DataFrame())
        
        # Run backtest
        results = self.backtester._run_standard_backtest(
            self.sample_data,
            self.sample_features
        )
        
        self.assertIn('trades', results)
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
    
    def test_metrics_calculation(self):
        """Test final metrics calculation"""
        # Generate sample results
        trades = [
            {'symbol': 'BTC_USDT', 'pnl': 100, 'return_pct': 0.01, 'duration': 5},
            {'symbol': 'BTC_USDT', 'pnl': -50, 'return_pct': -0.005, 'duration': 3},
            {'symbol': 'ETH_USDT', 'pnl': 75, 'return_pct': 0.0075, 'duration': 4},
        ]
        
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        equity = 10000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
        equity_curve = pd.Series(equity, index=dates)
        
        results = {
            'trades': trades,
            'equity_curve': self.backtester.equity_curve if self.backtester.equity_curve else [{'time': dates[i], 'value': equity[i]} for i in range(len(dates))],
            'positions': []
        }
        
        # Calculate metrics
        final_metrics = self.backtester._calculate_final_metrics(results)
        
        self.assertIsInstance(final_metrics, BacktestResults)
        self.assertIsNotNone(final_metrics.total_return)
        self.assertIsNotNone(final_metrics.sharpe_ratio)
        self.assertIsNotNone(final_metrics.max_drawdown)
        self.assertIsNotNone(final_metrics.win_rate)
    
    def test_full_backtest_run(self):
        """Test complete backtest execution"""
        # Use smaller date range for speed
        self.config.start_date = '2023-01-01'
        self.config.end_date = '2023-01-31'
        self.config.save_results = False
        
        # Mock data loading
        self.backtester.data_manager.load_existing_data = lambda s, t: self.sample_data.get(s, {}).get(t, pd.DataFrame())
        
        # Run backtest
        results = self.backtester.run()
        
        self.assertIsInstance(results, BacktestResults)
        self.assertGreaterEqual(results.total_trades, 0)
        self.assertIsNotNone(results.sharpe_ratio)
        self.assertIsNotNone(results.max_drawdown)
        
        # Check runtime was recorded
        self.assertGreater(results.runtime, 0)
    
    def test_parameter_optimization(self):
        """Test strategy parameter optimization"""
        base_config = BacktestConfig(
            symbols=['BTC_USDT'],
            timeframes=['1h'],
            start_date='2023-01-01',
            end_date='2023-01-31',
            initial_capital=10000,
            save_results=False,
            verbose=False
        )
        
        param_grid = {
            'stop_loss': [0.03, 0.05],
            'take_profit': [0.05, 0.10]
        }
        
        # Mock data loading for optimize_strategy
        def mock_run():
            bt = Backtester(base_config)
            bt.data_manager.load_existing_data = lambda s, t: self.sample_data.get(s, {}).get(t, pd.DataFrame())
            return bt.run()
        
        # This would run optimization in real scenario
        # For testing, just verify the structure
        from src.backtesting.backtester import _generate_param_combinations
        
        combinations = _generate_param_combinations(param_grid)
        self.assertEqual(len(combinations), 4)  # 2x2 grid
        
        # Check combinations contain all parameters
        for combo in combinations:
            self.assertIn('stop_loss', combo)
            self.assertIn('take_profit', combo)


class TestBacktestResults(unittest.TestCase):
    """Test BacktestResults functionality"""
    
    def test_results_save_and_load(self):
        """Test saving and loading results"""
        # Create sample results
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        equity_curve = pd.Series(10000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01), index=dates)
        
        results = BacktestResults(
            total_return=0.15,
            annual_return=0.18,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=0.10,
            win_rate=0.55,
            profit_factor=1.8,
            total_trades=50,
            winning_trades=28,
            losing_trades=22,
            avg_win=100,
            avg_loss=-60,
            largest_win=500,
            largest_loss=-200,
            avg_holding_period=24,
            longest_winning_streak=5,
            longest_losing_streak=3,
            value_at_risk=-0.02,
            expected_shortfall=-0.03,
            beta=0.8,
            alpha=0.05,
            equity_curve=equity_curve,
            drawdown_curve=pd.Series(np.random.randn(len(dates)) * 0.01, index=dates),
            positions_history=[],
            trades_history=[]
        )
        
        # Test all metrics are set correctly
        self.assertEqual(results.total_return, 0.15)
        self.assertEqual(results.sharpe_ratio, 1.5)
        self.assertEqual(results.win_rate, 0.55)
        self.assertEqual(results.total_trades, 50)


class TestIntegration(unittest.TestCase):
    """Integration tests with other components"""
    
    def test_data_manager_integration(self):
        """Test integration with DataManager"""
        config = BacktestConfig(
            symbols=['BTC_USDT'],
            save_results=False
        )
        
        backtester = Backtester(config)
        
        # Verify DataManager is properly initialized
        self.assertIsNotNone(backtester.data_manager)
        self.assertTrue(hasattr(backtester.data_manager, 'load_existing_data'))
    
    def test_feature_engineer_integration(self):
        """Test integration with FeatureEngineer"""
        config = BacktestConfig(
            symbols=['BTC_USDT'],
            save_results=False
        )
        
        backtester = Backtester(config)
        
        # Verify FeatureEngineer is available
        self.assertIsNotNone(backtester.feature_engineer)
        self.assertTrue(hasattr(backtester.feature_engineer, 'calculate_all_features'))
    
    def test_risk_manager_integration(self):
        """Test integration with RiskManager"""
        config = BacktestConfig(
            use_risk_manager=True,
            save_results=False
        )
        
        backtester = Backtester(config)
        
        # Verify RiskManager is initialized when requested
        self.assertIsNotNone(backtester.risk_manager)
        self.assertTrue(hasattr(backtester.risk_manager, 'calculate_position_size'))
        
        # Test without risk manager
        config.use_risk_manager = False
        backtester = Backtester(config)
        self.assertIsNone(backtester.risk_manager)


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktester))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestResults))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)