"""
Walk-Forward Analysis Module
Implements sophisticated walk-forward validation for trading strategies
Includes anchored and rolling walk-forward methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
from pathlib import Path

# Import our components
from ..models.ml_predictor import MLPredictor
from ..models.dqn_agent import DQNAgent
from ..features.feature_engineer import FeatureEngineer
from .metrics import MetricsCalculator, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis"""
    # Window settings
    training_period: int = 252  # Trading days for training
    testing_period: int = 63   # Trading days for testing
    step_size: int = 21        # Days to step forward
    
    # Method
    method: str = 'rolling'  # 'anchored' or 'rolling'
    min_training_size: int = 126  # Minimum training period
    
    # Optimization
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    parameter_grid: Dict[str, List] = None  # Parameters to optimize
    
    # Retraining
    retrain_models: bool = True
    retrain_frequency: int = 21  # Retrain every N days
    
    # Parallel processing
    use_parallel: bool = True
    n_jobs: int = -1  # Number of parallel jobs
    
    # Output
    save_results: bool = True
    results_path: str = './walk_forward_results/'
    verbose: bool = True


@dataclass
class WalkForwardSplit:
    """Single walk-forward split"""
    split_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: pd.Index
    test_indices: pd.Index


@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis"""
    # Overall metrics
    overall_metrics: PerformanceMetrics
    
    # Split-wise results
    split_results: List[Dict]
    split_metrics: List[PerformanceMetrics]
    
    # Out-of-sample performance
    oos_equity_curve: pd.Series
    oos_returns: pd.Series
    oos_trades: pd.DataFrame
    
    # In-sample vs out-of-sample
    is_sharpe: float
    oos_sharpe: float
    is_returns: float
    oos_returns_mean: float
    
    # Stability metrics
    consistency_score: float
    parameter_stability: Dict[str, float]
    
    # Best parameters per split
    best_parameters: List[Dict]
    
    # Metadata
    config: WalkForwardConfig
    total_splits: int
    runtime: float


class WalkForwardAnalyzer:
    """
    Sophisticated walk-forward analysis for strategy validation
    
    Features:
    - Rolling and anchored walk-forward
    - Parameter optimization per period
    - Model retraining
    - Parallel processing
    - Comprehensive metrics
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        """
        Initialize walk-forward analyzer
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config or WalkForwardConfig()
        self.metrics_calculator = MetricsCalculator()
        
        # Storage
        self.splits = []
        self.results = []
        
        logger.info(f"WalkForwardAnalyzer initialized with {self.config.method} method")
    
    def create_splits(self, 
                     data: pd.DataFrame,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> List[WalkForwardSplit]:
        """
        Create walk-forward splits
        
        Args:
            data: DataFrame with datetime index
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of WalkForwardSplit objects
        """
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
        
        # Filter data to date range
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        splits = []
        split_id = 0
        
        if self.config.method == 'rolling':
            splits = self._create_rolling_splits(data, split_id)
        elif self.config.method == 'anchored':
            splits = self._create_anchored_splits(data, split_id)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")
        
        self.splits = splits
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        return splits
    
    def _create_rolling_splits(self, data: pd.DataFrame, split_id: int) -> List[WalkForwardSplit]:
        """Create rolling window splits"""
        splits = []
        
        # Start from minimum training size
        current_pos = self.config.min_training_size
        
        while current_pos + self.config.testing_period < len(data):
            # Training window
            train_start_idx = max(0, current_pos - self.config.training_period)
            train_end_idx = current_pos
            
            # Testing window
            test_start_idx = current_pos
            test_end_idx = min(current_pos + self.config.testing_period, len(data))
            
            # Create split
            split = WalkForwardSplit(
                split_id=split_id,
                train_start=data.index[train_start_idx],
                train_end=data.index[train_end_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1],
                train_indices=data.index[train_start_idx:train_end_idx],
                test_indices=data.index[test_start_idx:test_end_idx]
            )
            
            splits.append(split)
            split_id += 1
            
            # Move forward
            current_pos += self.config.step_size
        
        return splits
    
    def _create_anchored_splits(self, data: pd.DataFrame, split_id: int) -> List[WalkForwardSplit]:
        """Create anchored (expanding window) splits"""
        splits = []
        
        # Start from minimum training size
        current_pos = self.config.min_training_size
        
        while current_pos + self.config.testing_period < len(data):
            # Training window (anchored at start)
            train_start_idx = 0
            train_end_idx = current_pos
            
            # Testing window
            test_start_idx = current_pos
            test_end_idx = min(current_pos + self.config.testing_period, len(data))
            
            # Create split
            split = WalkForwardSplit(
                split_id=split_id,
                train_start=data.index[train_start_idx],
                train_end=data.index[train_end_idx - 1],
                test_start=data.index[test_start_idx],
                test_end=data.index[test_end_idx - 1],
                train_indices=data.index[train_start_idx:train_end_idx],
                test_indices=data.index[test_start_idx:test_end_idx]
            )
            
            splits.append(split)
            split_id += 1
            
            # Move forward
            current_pos += self.config.step_size
        
        return splits
    
    def run_analysis(self,
                    data: pd.DataFrame,
                    features: pd.DataFrame,
                    strategy_func: Callable,
                    **strategy_kwargs) -> WalkForwardResults:
        """
        Run complete walk-forward analysis
        
        Args:
            data: OHLCV data
            features: Feature data
            strategy_func: Strategy function to test
            **strategy_kwargs: Additional arguments for strategy
            
        Returns:
            WalkForwardResults object
        """
        start_time = datetime.now()
        
        # Create splits if not already done
        if not self.splits:
            self.create_splits(data)
        
        logger.info(f"Starting walk-forward analysis with {len(self.splits)} splits")
        
        # Run analysis for each split
        if self.config.use_parallel:
            split_results = self._run_parallel_analysis(
                data, features, strategy_func, **strategy_kwargs
            )
        else:
            split_results = self._run_sequential_analysis(
                data, features, strategy_func, **strategy_kwargs
            )
        
        # Combine results
        combined_results = self._combine_results(split_results)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(combined_results)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(split_results)
        
        # Create results object
        results = WalkForwardResults(
            overall_metrics=overall_metrics,
            split_results=split_results,
            split_metrics=[r['metrics'] for r in split_results],
            oos_equity_curve=combined_results['equity_curve'],
            oos_returns=combined_results['returns'],
            oos_trades=combined_results['trades'],
            is_sharpe=self._calculate_is_sharpe(split_results),
            oos_sharpe=overall_metrics.sharpe_ratio,
            is_returns=self._calculate_is_returns(split_results),
            oos_returns_mean=combined_results['returns'].mean() if len(combined_results['returns']) > 0 else 0,
            consistency_score=stability_metrics['consistency'],
            parameter_stability=stability_metrics['parameter_stability'],
            best_parameters=[r.get('best_params', {}) for r in split_results],
            config=self.config,
            total_splits=len(self.splits),
            runtime=(datetime.now() - start_time).total_seconds()
        )
        
        # Save results if requested
        if self.config.save_results:
            self._save_results(results)
        
        logger.info(f"Walk-forward analysis completed in {results.runtime:.2f} seconds")
        
        return results
    
    def _run_sequential_analysis(self,
                                data: pd.DataFrame,
                                features: pd.DataFrame,
                                strategy_func: Callable,
                                **strategy_kwargs) -> List[Dict]:
        """Run analysis sequentially"""
        results = []
        
        for split in tqdm(self.splits, desc="Processing splits"):
            result = self._process_single_split(
                split, data, features, strategy_func, **strategy_kwargs
            )
            results.append(result)
        
        return results
    
    def _run_parallel_analysis(self,
                              data: pd.DataFrame,
                              features: pd.DataFrame,
                              strategy_func: Callable,
                              **strategy_kwargs) -> List[Dict]:
        """Run analysis in parallel"""
        results = []
        
        max_workers = self.config.n_jobs if self.config.n_jobs > 0 else None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for split in self.splits:
                future = executor.submit(
                    self._process_single_split,
                    split, data, features, strategy_func, **strategy_kwargs
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Processing splits"):
                result = future.result()
                results.append(result)
        
        # Sort by split ID to maintain order
        results.sort(key=lambda x: x['split_id'])
        
        return results
    
    def _process_single_split(self,
                            split: WalkForwardSplit,
                            data: pd.DataFrame,
                            features: pd.DataFrame,
                            strategy_func: Callable,
                            **strategy_kwargs) -> Dict:
        """Process a single walk-forward split"""
        # Get train and test data
        train_data = data.loc[split.train_indices]
        test_data = data.loc[split.test_indices]
        
        train_features = features.loc[split.train_indices] if features is not None else None
        test_features = features.loc[split.test_indices] if features is not None else None
        
        # Optimize parameters if grid provided
        if self.config.parameter_grid:
            best_params = self._optimize_parameters(
                train_data, train_features, strategy_func, **strategy_kwargs
            )
        else:
            best_params = {}
        
        # Train model with best parameters
        model = self._train_model(
            train_data, train_features, best_params, **strategy_kwargs
        )
        
        # Test on out-of-sample data
        test_results = self._test_model(
            model, test_data, test_features, **strategy_kwargs
        )
        
        # Calculate metrics
        if 'equity_curve' in test_results:
            metrics = self.metrics_calculator.calculate_all_metrics(
                test_results['equity_curve'],
                test_results.get('trades')
            )
        else:
            metrics = None
        
        # Calculate in-sample metrics for comparison
        train_results = self._test_model(
            model, train_data, train_features, **strategy_kwargs
        )
        
        if 'equity_curve' in train_results:
            is_metrics = self.metrics_calculator.calculate_all_metrics(
                train_results['equity_curve'],
                train_results.get('trades')
            )
        else:
            is_metrics = None
        
        return {
            'split_id': split.split_id,
            'train_period': (split.train_start, split.train_end),
            'test_period': (split.test_start, split.test_end),
            'best_params': best_params,
            'test_results': test_results,
            'metrics': metrics,
            'is_metrics': is_metrics,
            'model': model
        }
    
    def _optimize_parameters(self,
                            train_data: pd.DataFrame,
                            train_features: Optional[pd.DataFrame],
                            strategy_func: Callable,
                            **strategy_kwargs) -> Dict:
        """Optimize strategy parameters on training data"""
        best_params = {}
        best_score = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(self.config.parameter_grid)
        
        for params in param_combinations:
            # Train with current parameters
            model = self._train_model(
                train_data, train_features, params, **strategy_kwargs
            )
            
            # Test on training data (in-sample)
            results = self._test_model(
                model, train_data, train_features, **strategy_kwargs
            )
            
            # Calculate optimization metric
            if 'equity_curve' in results:
                metrics = self.metrics_calculator.calculate_all_metrics(
                    results['equity_curve'],
                    results.get('trades')
                )
                
                score = getattr(metrics, self.config.optimization_metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
        
        logger.debug(f"Best parameters: {best_params} with score: {best_score}")
        
        return best_params
    
    def _train_model(self,
                   train_data: pd.DataFrame,
                   train_features: Optional[pd.DataFrame],
                   params: Dict,
                   **kwargs) -> Any:
        """Train model with given parameters"""
        # This is a placeholder - would call actual training function
        # For example, could train MLPredictor or DQNAgent here
        return {'params': params, 'trained': True}
    
    def _test_model(self,
                  model: Any,
                  test_data: pd.DataFrame,
                  test_features: Optional[pd.DataFrame],
                  **kwargs) -> Dict:
        """Test model on data"""
        # This is a placeholder - would run actual strategy
        # Returns equity curve and trades
        
        # Generate dummy results for demonstration
        equity = 10000 * (1 + np.random.randn(len(test_data)).cumsum() * 0.001)
        equity_curve = pd.Series(equity, index=test_data.index)
        
        trades = pd.DataFrame({
            'pnl': np.random.randn(max(1, len(test_data) // 20)) * 100,
            'entry_time': test_data.index[0],
            'exit_time': test_data.index[-1]
        })
        
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': []
        }
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        from itertools import product
        
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _combine_results(self, split_results: List[Dict]) -> Dict:
        """Combine results from all splits"""
        all_trades = []
        equity_values = []
        returns = []
        
        for result in split_results:
            if 'test_results' in result:
                test_results = result['test_results']
                
                if 'trades' in test_results and test_results['trades'] is not None:
                    all_trades.append(test_results['trades'])
                
                if 'equity_curve' in test_results:
                    equity_values.append(test_results['equity_curve'])
                    returns.append(test_results['equity_curve'].pct_change().dropna())
        
        # Combine trades
        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()
        
        # Combine equity curves
        if equity_values:
            combined_equity = pd.concat(equity_values)
        else:
            combined_equity = pd.Series([10000])
        
        # Combine returns
        if returns:
            combined_returns = pd.concat(returns)
        else:
            combined_returns = pd.Series([0])
        
        return {
            'trades': combined_trades,
            'equity_curve': combined_equity,
            'returns': combined_returns
        }
    
    def _calculate_overall_metrics(self, combined_results: Dict) -> PerformanceMetrics:
        """Calculate overall metrics from combined results"""
        return self.metrics_calculator.calculate_all_metrics(
            combined_results['equity_curve'],
            combined_results['trades']
        )
    
    def _calculate_stability_metrics(self, split_results: List[Dict]) -> Dict:
        """Calculate stability metrics across splits"""
        # Extract metrics from each split
        sharpe_ratios = []
        returns = []
        win_rates = []
        
        for result in split_results:
            if result.get('metrics'):
                sharpe_ratios.append(result['metrics'].sharpe_ratio)
                returns.append(result['metrics'].total_return)
                win_rates.append(result['metrics'].win_rate)
        
        # Calculate consistency score
        if sharpe_ratios:
            # Consistency = 1 - coefficient of variation
            mean_sharpe = np.mean(sharpe_ratios)
            std_sharpe = np.std(sharpe_ratios)
            
            if mean_sharpe != 0:
                consistency = 1 - (std_sharpe / abs(mean_sharpe))
            else:
                consistency = 0
        else:
            consistency = 0
        
        # Parameter stability
        param_stability = self._calculate_parameter_stability(split_results)
        
        return {
            'consistency': consistency,
            'parameter_stability': param_stability,
            'sharpe_ratios': sharpe_ratios,
            'returns': returns,
            'win_rates': win_rates
        }
    
    def _calculate_parameter_stability(self, split_results: List[Dict]) -> Dict[str, float]:
        """Calculate how stable parameters are across splits"""
        param_values = {}
        
        for result in split_results:
            if 'best_params' in result:
                for param, value in result['best_params'].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)
        
        stability_scores = {}
        
        for param, values in param_values.items():
            if len(values) > 1:
                # Calculate coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if mean_val != 0:
                    stability_scores[param] = 1 - (std_val / abs(mean_val))
                else:
                    stability_scores[param] = 0
            else:
                stability_scores[param] = 1.0
        
        return stability_scores
    
    def _calculate_is_sharpe(self, split_results: List[Dict]) -> float:
        """Calculate average in-sample Sharpe ratio"""
        sharpe_ratios = []
        
        for result in split_results:
            if result.get('is_metrics'):
                sharpe_ratios.append(result['is_metrics'].sharpe_ratio)
        
        return np.mean(sharpe_ratios) if sharpe_ratios else 0
    
    def _calculate_is_returns(self, split_results: List[Dict]) -> float:
        """Calculate average in-sample returns"""
        returns = []
        
        for result in split_results:
            if result.get('is_metrics'):
                returns.append(result['is_metrics'].total_return)
        
        return np.mean(returns) if returns else 0
    
    def _save_results(self, results: WalkForwardResults):
        """Save walk-forward results"""
        output_dir = Path(self.config.results_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = output_dir / f"wf_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(self.create_report(results))
        
        # Save detailed results
        results_file = output_dir / f"wf_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {output_dir}")
    
    def create_report(self, results: WalkForwardResults) -> str:
        """Create text report of walk-forward results"""
        report = []
        report.append("=" * 60)
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 60)
        
        report.append(f"\nMethod: {self.config.method.upper()}")
        report.append(f"Total Splits: {results.total_splits}")
        report.append(f"Training Period: {self.config.training_period} days")
        report.append(f"Testing Period: {self.config.testing_period} days")
        
        report.append("\n### PERFORMANCE COMPARISON ###")
        report.append(f"In-Sample Sharpe: {results.is_sharpe:.3f}")
        report.append(f"Out-of-Sample Sharpe: {results.oos_sharpe:.3f}")
        report.append(f"IS/OOS Ratio: {results.is_sharpe/results.oos_sharpe:.3f}" if results.oos_sharpe != 0 else "N/A")
        
        report.append(f"\nIn-Sample Returns: {results.is_returns:.2%}")
        report.append(f"Out-of-Sample Returns: {results.overall_metrics.total_return:.2%}")
        
        report.append("\n### STABILITY METRICS ###")
        report.append(f"Consistency Score: {results.consistency_score:.3f}")
        
        if results.parameter_stability:
            report.append("\nParameter Stability:")
            for param, stability in results.parameter_stability.items():
                report.append(f"  {param}: {stability:.3f}")
        
        report.append("\n### OUT-OF-SAMPLE METRICS ###")
        report.append(f"Total Return: {results.overall_metrics.total_return:.2%}")
        report.append(f"Annual Return: {results.overall_metrics.annual_return:.2%}")
        report.append(f"Max Drawdown: {results.overall_metrics.max_drawdown:.2%}")
        report.append(f"Win Rate: {results.overall_metrics.win_rate:.2%}")
        report.append(f"Profit Factor: {results.overall_metrics.profit_factor:.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    data = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 101 + np.random.randn(len(dates)).cumsum(),
        'low': 99 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Configure walk-forward
    config = WalkForwardConfig(
        training_period=252,
        testing_period=63,
        step_size=21,
        method='rolling',
        parameter_grid={
            'stop_loss': [0.02, 0.03, 0.05],
            'take_profit': [0.05, 0.10, 0.15]
        }
    )
    
    # Run analysis
    analyzer = WalkForwardAnalyzer(config)
    
    # Dummy strategy function
    def dummy_strategy(**kwargs):
        return {}
    
    results = analyzer.run_analysis(data, None, dummy_strategy)
    
    # Print report
    print(analyzer.create_report(results))
    print("\n Walk-forward analyzer ready!")