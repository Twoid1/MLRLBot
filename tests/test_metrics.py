"""
Test suite for Metrics module
Tests performance metrics calculations
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtesting.metrics import MetricsCalculator, PerformanceMetrics


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.calculator = MetricsCalculator()
        
        # Generate sample equity curve
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Upward trending equity with some volatility
        np.random.seed(42)
        returns = np.random.randn(len(dates)) * 0.01 + 0.0005  # Slight positive bias
        cumulative_returns = (1 + returns).cumprod()
        cls.equity_curve = pd.Series(10000 * cumulative_returns, index=dates)
        
        # Generate sample trades
        cls.trades = pd.DataFrame({
            'pnl': [100, -50, 150, -30, 200, -60, 80, 120, -40, 90],
            'duration': [5, 3, 7, 2, 10, 4, 6, 8, 3, 5],
            'entry_time': dates[:10],
            'exit_time': dates[5:15]
        })
        
        # Generate benchmark returns
        benchmark_returns = np.random.randn(len(dates)) * 0.008  # Market returns
        cls.benchmark = pd.Series(benchmark_returns, index=dates)
    
    def test_initialization(self):
        """Test calculator initialization"""
        calculator = MetricsCalculator(
            risk_free_rate=0.03,
            trading_days_per_year=252,
            confidence_level=0.95
        )
        
        self.assertEqual(calculator.risk_free_rate, 0.03)
        self.assertEqual(calculator.trading_days, 252)
        self.assertEqual(calculator.confidence_level, 0.95)
    
    def test_total_return_calculation(self):
        """Test total return calculation"""
        total_return = self.calculator._calculate_total_return(self.equity_curve)
        
        expected_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
        self.assertAlmostEqual(total_return, expected_return, places=6)
    
    def test_annual_return_calculation(self):
        """Test annualized return calculation"""
        total_return = self.calculator._calculate_total_return(self.equity_curve)
        annual_return = self.calculator._calculate_annual_return(total_return, self.equity_curve)
        
        # Should be positive for upward trending equity
        self.assertGreater(annual_return, 0)
        
        # Should be reasonable (not extreme)
        self.assertLess(annual_return, 2.0)  # Less than 200% annually
        self.assertGreater(annual_return, -0.5)  # Greater than -50% annually
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = self.equity_curve.pct_change().dropna()
        sharpe = self.calculator._calculate_sharpe_ratio(returns, 0.02)
        
        # Check it's a reasonable value
        self.assertIsNotNone(sharpe)
        self.assertGreater(sharpe, -5)
        self.assertLess(sharpe, 5)
    
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        returns = self.equity_curve.pct_change().dropna()
        sortino = self.calculator._calculate_sortino_ratio(returns, 0.02)
        
        # Sortino should typically be higher than Sharpe
        sharpe = self.calculator._calculate_sharpe_ratio(returns, 0.02)
        
        # Both should exist
        self.assertIsNotNone(sortino)
        self.assertIsNotNone(sharpe)
    
    def test_drawdown_calculation(self):
        """Test drawdown calculation"""
        drawdown_series = self.calculator._calculate_drawdown_series(self.equity_curve)
        
        # Drawdowns should be negative or zero
        self.assertTrue(all(drawdown_series <= 0.001))  # Small tolerance for float precision
        
        # Max drawdown
        max_dd = abs(drawdown_series.min())
        self.assertGreaterEqual(max_dd, 0)
        self.assertLessEqual(max_dd, 1)  # Should be between 0 and 100%
    
    def test_drawdown_duration_calculation(self):
        """Test maximum drawdown duration calculation"""
        drawdown_series = self.calculator._calculate_drawdown_series(self.equity_curve)
        max_dd_duration = self.calculator._calculate_max_drawdown_duration(drawdown_series)
        
        # Duration should be non-negative
        self.assertGreaterEqual(max_dd_duration, 0)
        
        # Should be less than total days
        self.assertLess(max_dd_duration, len(self.equity_curve))
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        returns = self.equity_curve.pct_change().dropna()
        var_95 = self.calculator._calculate_var(returns, 0.95)
        
        # VaR should be negative (loss)
        self.assertLessEqual(var_95, 0)
        
        # Should be reasonable
        self.assertGreater(var_95, -0.1)  # Greater than -10% daily loss
    
    def test_cvar_calculation(self):
        """Test Conditional Value at Risk calculation"""
        returns = self.equity_curve.pct_change().dropna()
        cvar_95 = self.calculator._calculate_cvar(returns, 0.95)
        var_95 = self.calculator._calculate_var(returns, 0.95)
        
        # CVaR should be worse than VaR (more negative)
        self.assertLessEqual(cvar_95, var_95)
    
    def test_trade_metrics_calculation(self):
        """Test trade-based metrics calculation"""
        trade_metrics = self.calculator._calculate_trade_metrics(self.trades)
        
        # Check all expected keys exist
        expected_keys = [
            'total_trades', 'winning_trades', 'losing_trades', 'win_rate',
            'avg_win', 'avg_loss', 'best_trade', 'worst_trade', 'profit_factor'
        ]
        
        for key in expected_keys:
            self.assertIn(key, trade_metrics)
        
        # Validate values
        self.assertEqual(trade_metrics['total_trades'], len(self.trades))
        self.assertGreaterEqual(trade_metrics['win_rate'], 0)
        self.assertLessEqual(trade_metrics['win_rate'], 1)
        
        # Check win/loss counts
        winning = self.trades['pnl'] > 0
        self.assertEqual(trade_metrics['winning_trades'], winning.sum())
        self.assertEqual(trade_metrics['losing_trades'], (~winning).sum())
    
    def test_streak_calculation(self):
        """Test consecutive win/loss streak calculation"""
        max_wins, max_losses, current = self.calculator._calculate_streaks(self.trades)
        
        # Streaks should be non-negative
        self.assertGreaterEqual(max_wins, 0)
        self.assertGreaterEqual(max_losses, 0)
        
        # Current streak can be negative (losing) or positive (winning)
        self.assertIsNotNone(current)
    
    def test_omega_ratio_calculation(self):
        """Test Omega ratio calculation"""
        returns = self.equity_curve.pct_change().dropna()
        omega = self.calculator._calculate_omega_ratio(returns, threshold=0)
        
        # Omega ratio should be positive for profitable strategy
        self.assertGreater(omega, 0)
    
    def test_kelly_criterion_calculation(self):
        """Test Kelly Criterion calculation"""
        kelly = self.calculator._calculate_kelly_criterion(
            win_rate=0.6,
            avg_win=100,
            avg_loss=80
        )
        
        # Kelly should be between 0 and 0.25 (capped)
        self.assertGreaterEqual(kelly, 0)
        self.assertLessEqual(kelly, 0.25)
        
        # Test edge case
        kelly_zero = self.calculator._calculate_kelly_criterion(0.5, 100, 100)
        self.assertEqual(kelly_zero, 0)
    
    def test_tail_ratio_calculation(self):
        """Test tail ratio calculation"""
        returns = self.equity_curve.pct_change().dropna()
        tail_ratio = self.calculator._calculate_tail_ratio(returns, percentile=5)
        
        # Tail ratio should be positive for good strategies
        self.assertGreater(tail_ratio, 0)
    
    def test_complete_metrics_calculation(self):
        """Test complete metrics calculation"""
        metrics = self.calculator.calculate_all_metrics(
            equity_curve=self.equity_curve,
            trades=self.trades,
            benchmark=self.benchmark
        )
        
        # Check it returns PerformanceMetrics object
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # Check key metrics exist and are reasonable
        self.assertIsNotNone(metrics.total_return)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertIsNotNone(metrics.max_drawdown)
        self.assertIsNotNone(metrics.win_rate)
        
        # Check ranges
        self.assertGreaterEqual(metrics.max_drawdown, 0)
        self.assertLessEqual(metrics.max_drawdown, 1)
        self.assertGreaterEqual(metrics.win_rate, 0)
        self.assertLessEqual(metrics.win_rate, 1)
    
    def test_report_generation(self):
        """Test report generation"""
        metrics = self.calculator.calculate_all_metrics(
            equity_curve=self.equity_curve,
            trades=self.trades
        )
        
        report = self.calculator.create_report(metrics)
        
        # Check report contains key sections
        self.assertIn("PERFORMANCE METRICS REPORT", report)
        self.assertIn("RETURNS", report)
        self.assertIn("RISK METRICS", report)
        self.assertIn("TRADING STATISTICS", report)
        
        # Check specific metrics are in report
        self.assertIn("Sharpe Ratio:", report)
        self.assertIn("Max Drawdown:", report)
        self.assertIn("Win Rate:", report)


class TestPerformanceMetrics(unittest.TestCase):
    """Test PerformanceMetrics dataclass"""
    
    def test_metrics_initialization(self):
        """Test PerformanceMetrics initialization"""
        metrics = PerformanceMetrics(
            total_return=0.25,
            annual_return=0.30,
            monthly_return=0.025,
            daily_return=0.001,
            volatility=0.15,
            annual_volatility=0.20,
            downside_volatility=0.10,
            max_drawdown=0.12,
            max_drawdown_duration=30,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            information_ratio=0.8,
            treynor_ratio=0.12,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=150,
            avg_loss=-100,
            best_trade=500,
            worst_trade=-300,
            avg_trade=50,
            profit_factor=1.5,
            gross_profit=9000,
            gross_loss=6000,
            net_profit=3000,
            avg_holding_period=5,
            max_holding_period=20,
            min_holding_period=1,
            time_in_market=0.8,
            skewness=0.5,
            kurtosis=3.0,
            var_95=-0.02,
            cvar_95=-0.03,
            max_consecutive_wins=8,
            max_consecutive_losses=5,
            current_streak=3,
            recovery_factor=2.0,
            payoff_ratio=1.5,
            expectancy=30,
            kelly_criterion=0.15,
            optimal_f=0.20,
            omega_ratio=1.8,
            ulcer_index=0.05,
            tail_ratio=1.2,
            common_sense_ratio=1.5,
            cpc_index=2.0
        )
        
        # Check all values are set correctly
        self.assertEqual(metrics.total_return, 0.25)
        self.assertEqual(metrics.sharpe_ratio, 1.5)
        self.assertEqual(metrics.win_rate, 0.6)
        self.assertEqual(metrics.total_trades, 100)
        self.assertEqual(metrics.kelly_criterion, 0.15)
    
    def test_metadata_fields(self):
        """Test metadata fields in PerformanceMetrics"""
        start = datetime(2023, 1, 1)
        end = datetime(2023, 12, 31)
        
        metrics = PerformanceMetrics(
            total_return=0.25,
            annual_return=0.30,
            monthly_return=0.025,
            daily_return=0.001,
            volatility=0.15,
            annual_volatility=0.20,
            downside_volatility=0.10,
            max_drawdown=0.12,
            max_drawdown_duration=30,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            calmar_ratio=2.5,
            information_ratio=0.8,
            treynor_ratio=0.12,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            avg_win=150,
            avg_loss=-100,
            best_trade=500,
            worst_trade=-300,
            avg_trade=50,
            profit_factor=1.5,
            gross_profit=9000,
            gross_loss=6000,
            net_profit=3000,
            avg_holding_period=5,
            max_holding_period=20,
            min_holding_period=1,
            time_in_market=0.8,
            skewness=0.5,
            kurtosis=3.0,
            var_95=-0.02,
            cvar_95=-0.03,
            max_consecutive_wins=8,
            max_consecutive_losses=5,
            current_streak=3,
            recovery_factor=2.0,
            payoff_ratio=1.5,
            expectancy=30,
            kelly_criterion=0.15,
            optimal_f=0.20,
            omega_ratio=1.8,
            ulcer_index=0.05,
            tail_ratio=1.2,
            common_sense_ratio=1.5,
            cpc_index=2.0,
            start_date=start,
            end_date=end,
            trading_days=252,
            benchmark_correlation=0.65
        )
        
        self.assertEqual(metrics.start_date, start)
        self.assertEqual(metrics.end_date, end)
        self.assertEqual(metrics.trading_days, 252)
        self.assertEqual(metrics.benchmark_correlation, 0.65)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.calculator = MetricsCalculator()
    
    def test_empty_equity_curve(self):
        """Test with empty equity curve"""
        empty_curve = pd.Series([])
        metrics = self.calculator.calculate_all_metrics(empty_curve)
        
        # Should handle gracefully
        self.assertEqual(metrics.total_return, 0)
        self.assertEqual(metrics.total_trades, 0)
    
    def test_single_value_equity_curve(self):
        """Test with single value equity curve"""
        single_curve = pd.Series([10000])
        metrics = self.calculator.calculate_all_metrics(single_curve)
        
        # Should handle gracefully
        self.assertEqual(metrics.total_return, 0)
        self.assertEqual(metrics.volatility, 0)
    
    def test_no_trades(self):
        """Test with no trades"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        equity_curve = pd.Series(10000 * np.ones(len(dates)), index=dates)
        
        metrics = self.calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            trades=pd.DataFrame()
        )
        
        # Should handle empty trades
        self.assertEqual(metrics.total_trades, 0)
        self.assertEqual(metrics.win_rate, 0)
    
    def test_all_winning_trades(self):
        """Test with all winning trades"""
        trades = pd.DataFrame({
            'pnl': [100, 200, 150, 300],
            'duration': [5, 10, 7, 15]
        })
        
        trade_metrics = self.calculator._calculate_trade_metrics(trades)
        
        self.assertEqual(trade_metrics['win_rate'], 1.0)
        self.assertEqual(trade_metrics['losing_trades'], 0)
        self.assertEqual(trade_metrics['avg_loss'], 0)
    
    def test_all_losing_trades(self):
        """Test with all losing trades"""
        trades = pd.DataFrame({
            'pnl': [-100, -200, -150, -300],
            'duration': [5, 10, 7, 15]
        })
        
        trade_metrics = self.calculator._calculate_trade_metrics(trades)
        
        self.assertEqual(trade_metrics['win_rate'], 0.0)
        self.assertEqual(trade_metrics['winning_trades'], 0)
        self.assertEqual(trade_metrics['avg_win'], 0)


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
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