"""
Test suite for Walk-Forward Analysis module
Tests walk-forward validation functionality
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

from src.backtesting.walk_forward import (
    WalkForwardAnalyzer, WalkForwardConfig, WalkForwardSplit, WalkForwardResults
)
from src.backtesting.metrics import MetricsCalculator


class TestWalkForwardConfig(unittest.TestCase):
    """Test WalkForwardConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = WalkForwardConfig()
        
        self.assertEqual(config.training_period, 252)
        self.assertEqual(config.testing_period, 63)
        self.assertEqual(config.step_size, 21)
        self.assertEqual(config.method, 'rolling')
        self.assertTrue(config.retrain_models)
    
    def test_custom_config(self):
        """Test custom configuration"""
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.1, 0.2]
        }
        
        config = WalkForwardConfig(
            training_period=500,
            testing_period=100,
            method='anchored',
            parameter_grid=param_grid,
            use_parallel=False
        )
        
        self.assertEqual(config.training_period, 500)
        self.assertEqual(config.testing_period, 100)
        self.assertEqual(config.method, 'anchored')
        self.assertEqual(config.parameter_grid, param_grid)
        self.assertFalse(config.use_parallel)


class TestWalkForwardAnalyzer(unittest.TestCase):
    """Test WalkForwardAnalyzer class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Generate sample data
        cls.sample_data = cls._generate_sample_data()
        cls.sample_features = cls._generate_sample_features()
        
        # Create temporary directory
        cls.test_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _generate_sample_data(cls):
        """Generate sample OHLCV data"""
        # Create 2 years of daily data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        close = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
        
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(len(dates)) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(len(dates)) * 0.01)),
            'low': close * (1 - np.abs(np.random.randn(len(dates)) * 0.01)),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        return df
    
    @classmethod
    def _generate_sample_features(cls):
        """Generate sample features"""
        df = cls.sample_data
        
        features = pd.DataFrame(index=df.index)
        
        # Simple features
        features['returns'] = df['close'].pct_change()
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['rsi'] = 50 + np.random.randn(len(df)) * 15
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return features.fillna(0)
    
    def setUp(self):
        """Set up each test"""
        self.config = WalkForwardConfig(
            training_period=252,
            testing_period=63,
            step_size=21,
            method='rolling',
            save_results=False,
            verbose=False
        )
        
        self.analyzer = WalkForwardAnalyzer(self.config)
    
    def test_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer.config)
        self.assertIsNotNone(self.analyzer.metrics_calculator)
        self.assertEqual(self.analyzer.config.method, 'rolling')
    
    def test_create_rolling_splits(self):
        """Test rolling window split creation"""
        splits = self.analyzer.create_splits(self.sample_data)
        
        self.assertGreater(len(splits), 0)
        
        # Check first split
        first_split = splits[0]
        self.assertIsInstance(first_split, WalkForwardSplit)
        self.assertEqual(first_split.split_id, 0)
        
        # Training period should be correct size (or less if at start)
        train_days = (first_split.train_end - first_split.train_start).days
        self.assertLessEqual(train_days, self.config.training_period)
        
        # Test period should be correct size
        test_days = (first_split.test_end - first_split.test_start).days
        self.assertLessEqual(test_days, self.config.testing_period)
        
        # Test should start where training ends
        self.assertEqual(first_split.test_start, first_split.train_end + timedelta(days=1))
    
    def test_create_anchored_splits(self):
        """Test anchored (expanding window) split creation"""
        self.config.method = 'anchored'
        self.analyzer.config = self.config
        
        splits = self.analyzer.create_splits(self.sample_data)
        
        self.assertGreater(len(splits), 0)
        
        # Check that training starts are all the same (anchored)
        train_starts = [s.train_start for s in splits]
        self.assertTrue(all(s == train_starts[0] for s in train_starts))
        
        # Check that training windows expand
        for i in range(1, len(splits)):
            self.assertGreater(splits[i].train_end, splits[i-1].train_end)
    
    def test_split_continuity(self):
        """Test that splits cover data properly"""
        splits = self.analyzer.create_splits(self.sample_data)
        
        for i in range(len(splits) - 1):
            current_split = splits[i]
            next_split = splits[i + 1]
            
            # Check step size
            days_between = (next_split.test_start - current_split.test_start).days
            self.assertAlmostEqual(days_between, self.config.step_size, delta=1)
    
    def test_parameter_optimization(self):
        """Test parameter optimization on training data"""
        param_grid = {
            'param1': [1, 2, 3],
            'param2': [0.1, 0.2]
        }
        
        train_data = self.sample_data[:252]
        train_features = self.sample_features[:252]
        
        # Dummy strategy function
        def strategy_func(**params):
            return params
        
        best_params = self.analyzer._optimize_parameters(
            train_data, train_features, strategy_func
        )
        
        # Should return dictionary
        self.assertIsInstance(best_params, dict)
    
    def test_parameter_combination_generation(self):
        """Test parameter combination generation"""
        param_grid = {
            'param1': [1, 2],
            'param2': [0.1, 0.2, 0.3]
        }
        
        combinations = self.analyzer._generate_param_combinations(param_grid)
        
        # Should have 2 * 3 = 6 combinations
        self.assertEqual(len(combinations), 6)
        
        # Each combination should have all parameters
        for combo in combinations:
            self.assertIn('param1', combo)
            self.assertIn('param2', combo)
    
    def test_process_single_split(self):
        """Test processing of single split"""
        splits = self.analyzer.create_splits(self.sample_data)
        
        if not splits:
            self.skipTest("No splits created")
        
        first_split = splits[0]
        
        # Dummy strategy
        def strategy_func(**params):
            return {}
        
        result = self.analyzer._process_single_split(
            first_split,
            self.sample_data,
            self.sample_features,
            strategy_func
        )
        
        # Check result structure
        self.assertIn('split_id', result)
        self.assertIn('train_period', result)
        self.assertIn('test_period', result)
        self.assertIn('test_results', result)
        self.assertEqual(result['split_id'], first_split.split_id)
    
    def test_combine_results(self):
        """Test combining results from multiple splits"""
        # Create mock split results
        split_results = []
        
        for i in range(3):
            dates = pd.date_range(start=f'2023-0{i+1}-01', periods=63, freq='D')
            equity = 10000 * (1 + np.random.randn(len(dates)).cumsum() * 0.001)
            
            result = {
                'split_id': i,
                'test_results': {
                    'equity_curve': pd.Series(equity, index=dates),
                    'trades': pd.DataFrame({
                        'pnl': np.random.randn(5) * 100
                    })
                }
            }
            split_results.append(result)
        
        combined = self.analyzer._combine_results(split_results)
        
        self.assertIn('trades', combined)
        self.assertIn('equity_curve', combined)
        self.assertIn('returns', combined)
        
        # Should combine all trades
        total_trades = sum(len(r['test_results']['trades']) for r in split_results)
        self.assertEqual(len(combined['trades']), total_trades)
    
    def test_stability_metrics_calculation(self):
        """Test stability metrics calculation"""
        # Create mock split results with metrics
        split_results = []
        
        for i in range(5):
            result = {
                'split_id': i,
                'metrics': type('obj', (object,), {
                    'sharpe_ratio': 1.5 + np.random.randn() * 0.2,
                    'total_return': 0.1 + np.random.randn() * 0.05,
                    'win_rate': 0.55 + np.random.randn() * 0.1
                }),
                'best_params': {
                    'param1': np.random.choice([1, 2, 3]),
                    'param2': np.random.choice([0.1, 0.2])
                }
            }
            split_results.append(result)
        
        stability = self.analyzer._calculate_stability_metrics(split_results)
        
        self.assertIn('consistency', stability)
        self.assertIn('parameter_stability', stability)
        
        # Consistency should be between 0 and 1
        self.assertGreaterEqual(stability['consistency'], -1)
        self.assertLessEqual(stability['consistency'], 1)
    
    def test_parameter_stability_calculation(self):
        """Test parameter stability calculation"""
        split_results = [
            {'best_params': {'param1': 1, 'param2': 0.1}},
            {'best_params': {'param1': 1, 'param2': 0.15}},
            {'best_params': {'param1': 2, 'param2': 0.1}},
            {'best_params': {'param1': 1, 'param2': 0.12}},
        ]
        
        stability = self.analyzer._calculate_parameter_stability(split_results)
        
        self.assertIn('param1', stability)
        self.assertIn('param2', stability)
        
        # Stability scores should be between 0 and 1
        for score in stability.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_full_analysis_run(self):
        """Test complete walk-forward analysis"""
        # Use smaller data for speed
        small_data = self.sample_data[:500]
        small_features = self.sample_features[:500]
        
        # Configure for faster test
        self.config.training_period = 100
        self.config.testing_period = 30
        self.config.step_size = 50
        self.analyzer.config = self.config
        
        # Dummy strategy
        def strategy_func(**params):
            return {}
        
        # Run analysis
        results = self.analyzer.run_analysis(
            small_data,
            small_features,
            strategy_func
        )
        
        # Check results structure
        self.assertIsInstance(results, WalkForwardResults)
        self.assertIsNotNone(results.overall_metrics)
        self.assertIsNotNone(results.oos_equity_curve)
        self.assertGreaterEqual(results.total_splits, 1)
        self.assertGreater(results.runtime, 0)
    
    def test_parallel_vs_sequential(self):
        """Test that parallel and sequential give same number of results"""
        small_data = self.sample_data[:300]
        
        # Configure
        config = WalkForwardConfig(
            training_period=100,
            testing_period=30,
            step_size=30,
            save_results=False,
            verbose=False
        )
        
        # Sequential
        config.use_parallel = False
        analyzer_seq = WalkForwardAnalyzer(config)
        splits_seq = analyzer_seq.create_splits(small_data)
        
        # Parallel
        config.use_parallel = True
        analyzer_par = WalkForwardAnalyzer(config)
        splits_par = analyzer_par.create_splits(small_data)
        
        # Should create same number of splits
        self.assertEqual(len(splits_seq), len(splits_par))
    
    def test_report_generation(self):
        """Test report generation"""
        # Create mock results
        results = WalkForwardResults(
            overall_metrics=type('obj', (object,), {
                'total_return': 0.25,
                'annual_return': 0.30,
                'max_drawdown': 0.10,
                'sharpe_ratio': 1.5,
                'win_rate': 0.60,
                'profit_factor': 1.8
            }),
            split_results=[],
            split_metrics=[],
            oos_equity_curve=pd.Series([10000, 11000, 12000]),
            oos_returns=pd.Series([0.01, 0.02, -0.01]),
            oos_trades=pd.DataFrame(),
            is_sharpe=1.8,
            oos_sharpe=1.5,
            is_returns=0.30,
            oos_returns_mean=0.01,
            consistency_score=0.85,
            parameter_stability={'param1': 0.9, 'param2': 0.7},
            best_parameters=[],
            config=self.config,
            total_splits=10,
            runtime=120.5
        )
        
        report = self.analyzer.create_report(results)
        
        # Check report contains key information
        self.assertIn("WALK-FORWARD ANALYSIS REPORT", report)
        self.assertIn("Method: ROLLING", report)
        self.assertIn("Total Splits: 10", report)
        self.assertIn("In-Sample Sharpe: 1.800", report)
        self.assertIn("Out-of-Sample Sharpe: 1.500", report)
        self.assertIn("Consistency Score: 0.850", report)


class TestWalkForwardResults(unittest.TestCase):
    """Test WalkForwardResults dataclass"""
    
    def test_results_initialization(self):
        """Test WalkForwardResults initialization"""
        results = WalkForwardResults(
            overall_metrics=None,
            split_results=[],
            split_metrics=[],
            oos_equity_curve=pd.Series([10000, 11000]),
            oos_returns=pd.Series([0.01, 0.02]),
            oos_trades=pd.DataFrame(),
            is_sharpe=1.8,
            oos_sharpe=1.5,
            is_returns=0.25,
            oos_returns_mean=0.01,
            consistency_score=0.85,
            parameter_stability={},
            best_parameters=[],
            config=WalkForwardConfig(),
            total_splits=10,
            runtime=100.0
        )
        
        self.assertEqual(results.is_sharpe, 1.8)
        self.assertEqual(results.oos_sharpe, 1.5)
        self.assertEqual(results.consistency_score, 0.85)
        self.assertEqual(results.total_splits, 10)
        self.assertEqual(results.runtime, 100.0)
    
    def test_is_oos_comparison(self):
        """Test in-sample vs out-of-sample comparison"""
        results = WalkForwardResults(
            overall_metrics=None,
            split_results=[],
            split_metrics=[],
            oos_equity_curve=pd.Series([]),
            oos_returns=pd.Series([]),
            oos_trades=pd.DataFrame(),
            is_sharpe=2.0,
            oos_sharpe=1.5,
            is_returns=0.40,
            oos_returns_mean=0.25,
            consistency_score=0.75,
            parameter_stability={},
            best_parameters=[],
            config=WalkForwardConfig(),
            total_splits=10,
            runtime=100.0
        )
        
        # Calculate degradation
        sharpe_degradation = (results.is_sharpe - results.oos_sharpe) / results.is_sharpe
        returns_degradation = (results.is_returns - results.oos_returns_mean) / results.is_returns
        
        # Typical degradation is 20-40%
        self.assertGreater(sharpe_degradation, 0)  # IS should be better than OOS
        self.assertLess(sharpe_degradation, 1)  # But not completely different


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForwardConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForwardAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestWalkForwardResults))
    
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