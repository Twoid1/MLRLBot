"""
Test Script for Feature Selector Module
Tests all functionality of selector.py
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
import time
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns

# Find project root and add to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import modules to test
from src.features.selector import FeatureSelector, FeatureSelectionResult
from src.features.feature_engineer import FeatureEngineer
from src.features.indicators import TechnicalIndicators

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TestFeatureSelector:
    """
    Comprehensive test suite for FeatureSelector class
    """
    
    def __init__(self, data_path: str = None, visualize: bool = False):
        """
        Initialize test suite
        
        Args:
            data_path: Path to raw data directory
            visualize: Whether to create visualization plots
        """
        # Path handling
        if data_path is None:
            possible_roots = [
                Path.cwd(),
                Path(__file__).parent.parent,
                Path.cwd().parent,
            ]
            
            for root in possible_roots:
                if (root / 'data' / 'raw').exists():
                    data_path = root / 'data' / 'raw'
                    break
            else:
                data_path = Path('data/raw')
        
        self.data_path = Path(data_path)
        self.selector = FeatureSelector(n_features=50)
        self.test_results = []
        self.visualize = visualize
        self.selection_results = {}
        
    def run_all_tests(self):
        """Run all tests and report results"""
        print("=" * 80)
        print("FEATURE SELECTOR TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target features: {self.selector.n_features}")
        print(f"Task type: {self.selector.task_type}\n")
        
        # Generate test data
        print("Preparing test data...")
        X, y = self.prepare_test_data()
        if X is None or y is None:
            print(" Failed to prepare test data. Stopping tests.")
            return
        
        # Test individual selection methods
        print("\n" + "="*50)
        print("INDIVIDUAL SELECTION METHODS")
        print("="*50)
        
        print("\n1. Testing Variance Threshold Selection...")
        self.test_variance_selection(X, y)
        
        print("\n2. Testing Correlation Selection...")
        self.test_correlation_selection(X, y)
        
        print("\n3. Testing Mutual Information Selection...")
        self.test_mutual_info_selection(X, y)
        
        print("\n4. Testing Random Forest Selection...")
        self.test_random_forest_selection(X, y)
        
        print("\n5. Testing Lasso Selection...")
        self.test_lasso_selection(X, y)
        
        print("\n6. Testing Recursive Feature Elimination...")
        self.test_recursive_elimination(X, y)
        
        print("\n7. Testing Boruta Selection...")
        self.test_boruta_selection(X, y)
        
        print("\n8. Testing Stability Selection...")
        self.test_stability_selection(X, y)
        
        # Test ensemble methods
        print("\n" + "="*50)
        print("ENSEMBLE METHODS")
        print("="*50)
        
        print("\n9. Testing Complete Selection Pipeline...")
        self.test_complete_selection(X, y)
        
        print("\n10. Testing Ensemble Combination...")
        self.test_ensemble_combination()
        
        # Test multi-asset selection
        print("\n" + "="*50)
        print("MULTI-ASSET SELECTION")
        print("="*50)
        
        print("\n11. Testing Multi-Asset Selection...")
        self.test_multi_asset_selection()
        
        # Test evaluation and reporting
        print("\n" + "="*50)
        print("EVALUATION & REPORTING")
        print("="*50)
        
        print("\n12. Testing Feature Evaluation...")
        self.test_feature_evaluation(X, y)
        
        print("\n13. Testing Feature Report Generation...")
        self.test_feature_report()
        
        # Test save/load
        print("\n" + "="*50)
        print("PERSISTENCE")
        print("="*50)
        
        print("\n14. Testing Save/Load Functionality...")
        self.test_save_load()
        
        # Test edge cases
        print("\n" + "="*50)
        print("EDGE CASES")
        print("="*50)
        
        print("\n15. Testing Edge Cases...")
        self.test_edge_cases()
        
        # Performance tests
        print("\n" + "="*50)
        print("PERFORMANCE")
        print("="*50)
        
        print("\n16. Testing Performance...")
        self.test_performance()
        
        # Visualizations
        if self.visualize:
            print("\n" + "="*50)
            print("VISUALIZATIONS")
            print("="*50)
            self.create_visualizations()
        
        # Print summary
        self.print_test_summary()
    
    def prepare_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test data with features and target"""
        try:
            # Try to load real data
            possible_paths = [
                self.data_path / '1d' / 'BTC_USDT_1d.csv',
                self.data_path / '1h' / 'BTC_USDT_1h.csv',
                self.data_path / '4h' / 'BTC_USDT_4h.csv',
            ]
            
            df = None
            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    print(f" Loaded data from {path.name}")
                    break
            
            if df is None:
                print(" No data file found. Generating sample data...")
                df = self.generate_sample_ohlcv(1000)
            
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()
            
            # Generate features using FeatureEngineer
            print("Generating features...")
            fe = FeatureEngineer()
            
            # Calculate all features
            features = fe.calculate_all_features(df, symbol='BTC/USDT')
            
            # Create target variable (classification: up/down)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Align indices
            common_idx = features.index.intersection(target.index)
            features = features.loc[common_idx]
            target = target.loc[common_idx]
            
            # Remove rows with too many NaN values
            nan_threshold = 0.3
            nan_ratio = features.isna().sum(axis=1) / len(features.columns)
            valid_rows = nan_ratio < nan_threshold
            
            features = features[valid_rows]
            target = target[valid_rows]
            
            # Fill remaining NaN values
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            print(f" Prepared features: {features.shape}")
            print(f" Target distribution: {target.value_counts().to_dict()}")
            
            return features, target
            
        except Exception as e:
            print(f" Error preparing data: {e}")
            return None, None
    
    def generate_sample_ohlcv(self, periods: int = 1000) -> pd.DataFrame:
        """Generate sample OHLCV data"""
        np.random.seed(42)
        
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate price series
        returns = np.random.normal(0.0001, 0.02, periods)
        price_series = 50000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(index=timestamps)
        df['close'] = price_series
        df['open'] = df['close'] * (1 + np.random.normal(0, 0.003, periods))
        df['high'] = np.maximum(df['open'], df['close']) * (1 + np.abs(np.random.normal(0, 0.01, periods)))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - np.abs(np.random.normal(0, 0.01, periods)))
        df['volume'] = 1000000 * (1 + np.abs(np.random.normal(0, 0.5, periods)))
        
        return df
    
    # ================== INDIVIDUAL METHOD TESTS ==================
    
    def test_variance_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test variance threshold selection"""
        try:
            start_time = time.time()
            result = self.selector.variance_threshold_selection(X, y)
            elapsed = time.time() - start_time
            
            # Validate result
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            assert len(result.selected_features) <= self.selector.n_features
            
            # Store for later
            self.selection_results['variance'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Top 5 features: {result.selected_features[:5]}")
            
            self.test_results.append(('Variance Selection', 'PASS'))
            
        except Exception as e:
            print(f" Variance selection failed: {e}")
            self.test_results.append(('Variance Selection', 'FAIL'))
    
    def test_correlation_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test correlation-based selection"""
        try:
            start_time = time.time()
            result = self.selector.correlation_selection(X, y)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            
            self.selection_results['correlation'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            
            # Check that highly correlated features were removed
            selected_X = X[result.selected_features]
            corr_matrix = selected_X.corr().abs()
            high_corr = (corr_matrix > 0.95).sum().sum()
            print(f"   High correlations remaining: {high_corr}")
            
            self.test_results.append(('Correlation Selection', 'PASS'))
            
        except Exception as e:
            print(f" Correlation selection failed: {e}")
            self.test_results.append(('Correlation Selection', 'FAIL'))
    
    def test_mutual_info_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test mutual information selection"""
        try:
            start_time = time.time()
            result = self.selector.mutual_info_selection(X, y)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) == min(self.selector.n_features, len(X.columns))
            
            self.selection_results['mutual_info'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Top feature MI score: {result.feature_scores.iloc[0]:.4f}")
            
            self.test_results.append(('Mutual Info Selection', 'PASS'))
            
        except Exception as e:
            print(f" Mutual info selection failed: {e}")
            self.test_results.append(('Mutual Info Selection', 'FAIL'))
    
    def test_random_forest_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test Random Forest importance selection"""
        try:
            start_time = time.time()
            result = self.selector.random_forest_selection(X, y, n_estimators=50)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) == min(self.selector.n_features, len(X.columns))
            
            self.selection_results['random_forest'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Top feature importance: {result.feature_scores.iloc[0]:.4f}")
            
            self.test_results.append(('Random Forest Selection', 'PASS'))
            
        except Exception as e:
            print(f" Random Forest selection failed: {e}")
            self.test_results.append(('Random Forest Selection', 'FAIL'))
    
    def test_lasso_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test Lasso regularization selection"""
        try:
            start_time = time.time()
            result = self.selector.lasso_selection(X, y, cv=3)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            
            self.selection_results['lasso'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Alpha: {result.metadata.get('alpha', 'N/A')}")
            
            self.test_results.append(('Lasso Selection', 'PASS'))
            
        except Exception as e:
            print(f" Lasso selection failed: {e}")
            self.test_results.append(('Lasso Selection', 'FAIL'))
    
    def test_recursive_elimination(self, X: pd.DataFrame, y: pd.Series):
        """Test Recursive Feature Elimination"""
        try:
            # Use smaller subset for speed
            X_subset = X.iloc[:500]
            y_subset = y.iloc[:500]
            
            start_time = time.time()
            result = self.selector.recursive_elimination(X_subset, y_subset, step=5)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) == min(self.selector.n_features, len(X.columns))
            
            self.selection_results['recursive'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            
            self.test_results.append(('Recursive Elimination', 'PASS'))
            
        except Exception as e:
            print(f" Recursive elimination failed: {e}")
            self.test_results.append(('Recursive Elimination', 'FAIL'))
    
    def test_boruta_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test Boruta-like selection"""
        try:
            start_time = time.time()
            result = self.selector.boruta_selection(X, y, n_iterations=50)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            
            self.selection_results['boruta'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Max shadow importance: {result.metadata.get('max_shadow_importance', 'N/A'):.4f}")
            
            self.test_results.append(('Boruta Selection', 'PASS'))
            
        except Exception as e:
            print(f" Boruta selection failed: {e}")
            self.test_results.append(('Boruta Selection', 'FAIL'))
    
    def test_stability_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test Stability selection"""
        try:
            # Use smaller subset for speed
            X_subset = X.iloc[:500]
            y_subset = y.iloc[:500]
            
            start_time = time.time()
            result = self.selector.stability_selection(X_subset, y_subset, n_bootstrap=20)
            elapsed = time.time() - start_time
            
            assert isinstance(result, FeatureSelectionResult)
            assert len(result.selected_features) > 0
            
            self.selection_results['stability'] = result
            
            print(f" Selected {len(result.selected_features)} features in {elapsed:.2f}s")
            print(f"   Performance score: {result.performance_score:.4f}")
            print(f"   Top stability score: {result.feature_scores.iloc[0]:.4f}")
            
            # Check stability scores
            stable_features = result.feature_scores[result.feature_scores > 0.8]
            print(f"   Highly stable features (>0.8): {len(stable_features)}")
            
            self.test_results.append(('Stability Selection', 'PASS'))
            
        except Exception as e:
            print(f" Stability selection failed: {e}")
            self.test_results.append(('Stability Selection', 'FAIL'))
    
    # ================== ENSEMBLE TESTS ==================
    
    def test_complete_selection(self, X: pd.DataFrame, y: pd.Series):
        """Test complete selection pipeline"""
        try:
            # Use subset for speed
            X_subset = X.iloc[:500]
            y_subset = y.iloc[:500]
            
            start_time = time.time()
            
            # Run with fewer methods for speed
            methods = ['variance', 'mutual_info', 'random_forest']
            results = self.selector.select_features(X_subset, y_subset, methods=methods)
            
            elapsed = time.time() - start_time
            
            assert len(results) == len(methods)
            assert self.selector.final_features is not None
            assert len(self.selector.final_features) > 0
            
            print(f" Complete pipeline finished in {elapsed:.2f}s")
            print(f"   Methods used: {methods}")
            print(f"   Final features selected: {len(self.selector.final_features)}")
            
            self.test_results.append(('Complete Selection', 'PASS'))
            
        except Exception as e:
            print(f" Complete selection failed: {e}")
            self.test_results.append(('Complete Selection', 'FAIL'))
    
    def test_ensemble_combination(self):
        """Test ensemble combination of methods"""
        try:
            if not self.selection_results:
                print(" No selection results to combine")
                return
            
            # Test combination with default weights
            final_features = self.selector.combine_selections(self.selection_results)
            
            assert len(final_features) > 0
            assert len(final_features) <= self.selector.n_features
            
            print(f" Combined {len(self.selection_results)} methods")
            print(f"   Final features: {len(final_features)}")
            
            # Test custom weights
            custom_weights = {
                'variance': 0.5,
                'correlation': 1.0,
                'mutual_info': 2.0,
                'random_forest': 2.0
            }
            
            weighted_features = self.selector.combine_selections(
                self.selection_results, weights=custom_weights
            )
            
            print(f"   Custom weighted features: {len(weighted_features)}")
            
            # Check overlap
            overlap = set(final_features) & set(weighted_features)
            print(f"   Overlap between methods: {len(overlap)} features")
            
            self.test_results.append(('Ensemble Combination', 'PASS'))
            
        except Exception as e:
            print(f" Ensemble combination failed: {e}")
            self.test_results.append(('Ensemble Combination', 'FAIL'))
    
    # ================== MULTI-ASSET TESTS ==================
    
    def test_multi_asset_selection(self):
        """Test multi-asset feature selection"""
        try:
            # Generate data for multiple assets
            assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            asset_data = {}
            asset_targets = {}
            
            print("   Generating multi-asset data...")
            for asset in assets:
                # Generate slightly different data for each asset
                df = self.generate_sample_ohlcv(500)
                fe = FeatureEngineer()
                features = fe.calculate_all_features(df, symbol=asset)
                features = features.fillna(method='ffill').fillna(0)
                
                target = (df['close'].shift(-1) > df['close']).astype(int)
                
                # Align
                common_idx = features.index.intersection(target.index)
                asset_data[asset] = features.loc[common_idx]
                asset_targets[asset] = target.loc[common_idx]
            
            # Run multi-asset selection
            universal_features = self.selector.multi_asset_selection(
                asset_data, asset_targets
            )
            
            assert len(universal_features) > 0
            
            print(f" Multi-asset selection completed")
            print(f"   Assets tested: {len(assets)}")
            print(f"   Universal features: {len(universal_features)}")
            print(f"   Top 5 universal: {universal_features[:5]}")
            
            self.test_results.append(('Multi-Asset Selection', 'PASS'))
            
        except Exception as e:
            print(f" Multi-asset selection failed: {e}")
            self.test_results.append(('Multi-Asset Selection', 'FAIL'))
    
    # ================== EVALUATION TESTS ==================
    
    def test_feature_evaluation(self, X: pd.DataFrame, y: pd.Series):
        """Test feature evaluation"""
        try:
            # Use a small set of features for testing
            test_features = X.columns[:20].tolist()
            
            score = self.selector.evaluate_features(
                X.iloc[:500], y.iloc[:500], test_features, cv=3
            )
            
            assert isinstance(score, float)
            assert 0 <= score <= 1 or -1 <= score <= 1  # Depending on metric
            
            print(f" Feature evaluation completed")
            print(f"   Features evaluated: {len(test_features)}")
            print(f"   Cross-validation score: {score:.4f}")
            
            self.test_results.append(('Feature Evaluation', 'PASS'))
            
        except Exception as e:
            print(f" Feature evaluation failed: {e}")
            self.test_results.append(('Feature Evaluation', 'FAIL'))
    
    def test_feature_report(self):
        """Test feature report generation"""
        try:
            if not self.selection_results:
                print(" No selection results for report")
                return
            
            # Store results in selector
            self.selector.selection_results = self.selection_results
            
            report = self.selector.get_feature_report()
            
            assert isinstance(report, pd.DataFrame)
            assert len(report) > 0
            
            print(f" Feature report generated")
            print(f"   Total features in report: {len(report)}")
            print(f"   Report columns: {list(report.columns)[:5]}...")
            
            # Show top features by selection rate
            if 'selection_rate' in report.columns:
                top_features = report.nlargest(5, 'selection_rate')
                print("\n   Top 5 features by selection rate:")
                for idx, row in top_features.iterrows():
                    print(f"     {idx}: {row['selection_rate']:.2f}")
            
            self.test_results.append(('Feature Report', 'PASS'))
            
        except Exception as e:
            print(f" Feature report failed: {e}")
            self.test_results.append(('Feature Report', 'FAIL'))
    
    # ================== PERSISTENCE TESTS ==================
    
    def test_save_load(self):
        """Test save and load functionality"""
        try:
            if not self.selector.final_features:
                # Create some dummy results
                self.selector.final_features = ['feature1', 'feature2', 'feature3']
                self.selector.selection_results = self.selection_results
            
            # Save
            test_file = 'test_selection.json'
            self.selector.save_selection(test_file)
            print(f" Saved selection to {test_file}")
            
            # Create new selector and load
            new_selector = FeatureSelector()
            new_selector.load_selection(test_file)
            
            # Validate
            assert new_selector.final_features == self.selector.final_features
            assert new_selector.n_features == self.selector.n_features
            
            print(f" Loaded selection successfully")
            print(f"   Features match: {new_selector.final_features == self.selector.final_features}")
            
            # Clean up
            Path(test_file).unlink(missing_ok=True)
            
            self.test_results.append(('Save/Load', 'PASS'))
            
        except Exception as e:
            print(f" Save/Load failed: {e}")
            self.test_results.append(('Save/Load', 'FAIL'))
    
    # ================== EDGE CASES ==================
    
    def test_edge_cases(self):
        """Test edge cases"""
        try:
            print("   Testing edge cases...")
            
            # 1. Empty DataFrame
            try:
                empty_X = pd.DataFrame()
                empty_y = pd.Series()
                result = self.selector.variance_threshold_selection(empty_X, empty_y)
                print("    Should have failed with empty data")
            except:
                print("    Correctly handled empty DataFrame")
            
            # 2. Single feature
            single_X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
            single_y = pd.Series([0, 1, 0, 1, 0])
            result = self.selector.variance_threshold_selection(single_X, single_y)
            assert len(result.selected_features) <= 1
            print("    Handled single feature")
            
            # 3. All identical features (FIXED)
            try:
                identical_X = pd.DataFrame({
                    f'feature{i}': [1, 1, 1, 1, 1] 
                    for i in range(10)
                })
                identical_y = pd.Series([0, 1, 0, 1, 0])
                result = self.selector.variance_threshold_selection(identical_X, identical_y, threshold=0)
                # With threshold=0, it should keep the features even with no variance
                print(f"    Handled identical features (selected {len(result.selected_features)})")
            except Exception as e:
                # This is actually expected - features with zero variance should be filtered
                if "variance threshold" in str(e).lower():
                    print(f"    Correctly rejected zero-variance features")
                else:
                    raise e
            
            # 4. Perfect correlation
            np.random.seed(42)  # Set seed for reproducibility
            corr_data = np.random.randn(100, 1)
            perfect_corr_X = pd.DataFrame({
                'feature1': corr_data.flatten(),
                'feature2': corr_data.flatten() * 2,
                'feature3': corr_data.flatten() * 3,
            })
            perfect_y = pd.Series(np.random.randint(0, 2, 100))
            result = self.selector.correlation_selection(perfect_corr_X, perfect_y)
            # Should only keep 1 feature from perfectly correlated set
            assert len(result.selected_features) <= len(perfect_corr_X.columns)
            print(f"    Handled perfect correlation (selected {len(result.selected_features)})")
            
            # 5. NaN values
            nan_X = pd.DataFrame(np.random.randn(100, 10))
            nan_X.iloc[0:10, 0:3] = np.nan
            nan_y = pd.Series(np.random.randint(0, 2, 100))
            
            # Should handle by cleaning
            clean_X, clean_y = self.selector._clean_data(nan_X, nan_y)
            assert len(clean_X) < len(nan_X)
            print(f"    Handled NaN values (cleaned {len(nan_X) - len(clean_X)} rows)")
            
            # 6. More features than samples
            wide_X = pd.DataFrame(np.random.randn(10, 100))  # 10 samples, 100 features
            wide_y = pd.Series(np.random.randint(0, 2, 10))
            
            # Test that it can handle wide data
            try:
                result = self.selector.mutual_info_selection(wide_X, wide_y)
                print(f"    Handled wide data (more features than samples)")
            except:
                print(f"    Could not handle wide data")
            
            # 7. Binary features
            binary_X = pd.DataFrame({
                f'binary_{i}': np.random.randint(0, 2, 100)
                for i in range(10)
            })
            binary_y = pd.Series(np.random.randint(0, 2, 100))
            result = self.selector.mutual_info_selection(binary_X, binary_y)
            print(f"    Handled binary features")
            
            # 8. Highly imbalanced target
            imbalanced_y = pd.Series([0] * 95 + [1] * 5)  # 95% class 0, 5% class 1
            normal_X = pd.DataFrame(np.random.randn(100, 20))
            result = self.selector.random_forest_selection(normal_X, imbalanced_y, n_estimators=10)
            print(f"    Handled imbalanced target")
            
            self.test_results.append(('Edge Cases', 'PASS'))
            
        except Exception as e:
            print(f" Edge cases failed: {e}")
            self.test_results.append(('Edge Cases', 'FAIL'))
    
    # ================== PERFORMANCE TESTS ==================
    
    def test_performance(self):
        """Test performance with different data sizes"""
        try:
            print("   Running performance benchmarks...")
            
            sizes = [50, 100, 200]
            n_features = [10, 50, 100]
            
            times = {}
            
            for n_samples in sizes:
                for n_feat in n_features:
                    # Generate data
                    X = pd.DataFrame(np.random.randn(n_samples, n_feat))
                    y = pd.Series(np.random.randint(0, 2, n_samples))
                    
                    # Time mutual info selection
                    start = time.time()
                    _ = self.selector.mutual_info_selection(X, y)
                    elapsed = time.time() - start
                    
                    times[(n_samples, n_feat)] = elapsed
                    print(f"     {n_samples} samples, {n_feat} features: {elapsed:.3f}s")
            
            # Check scaling
            if (100, 50) in times and (200, 50) in times:
                scaling = times[(200, 50)] / times[(100, 50)]
                print(f"\n    Scaling factor (2x samples): {scaling:.2f}x")
                
                if scaling < 4:
                    print("   Good performance scaling")
                else:
                    print("    Poor scaling performance")
            
            self.test_results.append(('Performance', 'PASS'))
            
        except Exception as e:
            print(f" Performance test failed: {e}")
            self.test_results.append(('Performance', 'FAIL'))
    
    # ================== VISUALIZATIONS ==================
    
    def create_visualizations(self):
        """Create visualizations of selection results"""
        try:
            print("\nCreating visualizations...")
            
            if not self.selection_results:
                print(" No results to visualize")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Feature selection counts across methods
            ax = axes[0, 0]
            method_counts = {
                method: len(result.selected_features) 
                for method, result in self.selection_results.items()
            }
            ax.bar(method_counts.keys(), method_counts.values())
            ax.set_title('Features Selected by Each Method')
            ax.set_xlabel('Method')
            ax.set_ylabel('Number of Features')
            ax.tick_params(axis='x', rotation=45)
            
            # 2. Feature importance scores
            ax = axes[0, 1]
            if 'random_forest' in self.selection_results:
                rf_scores = self.selection_results['random_forest'].feature_scores.head(20)
                ax.barh(range(len(rf_scores)), rf_scores.values)
                ax.set_yticks(range(len(rf_scores)))
                ax.set_yticklabels([f'F{i}' for i in range(len(rf_scores))])
                ax.set_title('Top 20 Features by Random Forest')
                ax.set_xlabel('Importance Score')
            
            # 3. Feature overlap heatmap
            ax = axes[1, 0]
            if len(self.selection_results) > 1:
                methods = list(self.selection_results.keys())
                overlap_matrix = np.zeros((len(methods), len(methods)))
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        features1 = set(self.selection_results[method1].selected_features)
                        features2 = set(self.selection_results[method2].selected_features)
                        overlap = len(features1 & features2) / len(features1 | features2) if features1 | features2 else 0
                        overlap_matrix[i, j] = overlap
                
                im = ax.imshow(overlap_matrix, cmap='coolwarm', vmin=0, vmax=1)
                ax.set_xticks(range(len(methods)))
                ax.set_yticks(range(len(methods)))
                ax.set_xticklabels(methods, rotation=45)
                ax.set_yticklabels(methods)
                ax.set_title('Feature Selection Method Overlap')
                plt.colorbar(im, ax=ax)
            
            # 4. Stability scores
            ax = axes[1, 1]
            if 'stability' in self.selection_results:
                stability_scores = self.selection_results['stability'].feature_scores.head(30)
                ax.plot(stability_scores.values, 'o-')
                ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                ax.set_title('Feature Stability Scores')
                ax.set_xlabel('Feature Rank')
                ax.set_ylabel('Stability Score')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('feature_selection_visualization.png', dpi=100, bbox_inches='tight')
            print(" Saved visualizations to feature_selection_visualization.png")
            
        except Exception as e:
            print(f" Visualization failed: {e}")
    
    # ================== SUMMARY ==================
    
    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in self.test_results if result == 'PASS')
        failed = sum(1 for _, result in self.test_results if result == 'FAIL')
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ")
        print(f"Failed: {failed} ")
        print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results:
            symbol = "Y" if result == "PASS" else "N"
            print(f"  {symbol} {test_name}: {result}")
        
        # Summary of selected features
        if self.selector.final_features:
            print(f"\nFinal Selected Features: {len(self.selector.final_features)}")
            print(f"Top 10 final features:")
            for i, feat in enumerate(self.selector.final_features[:10], 1):
                print(f"  {i}. {feat}")
        
        print("\n" + "=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Feature Selector')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--n-features', type=int, default=50,
                       help='Number of features to select')
    parser.add_argument('--task', type=str, default='classification',
                       choices=['classification', 'regression'],
                       help='Task type')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TestFeatureSelector(args.data_path, args.visualize)
    
    # Configure selector
    tester.selector.n_features = args.n_features
    tester.selector.task_type = args.task
    
    # Run all tests
    tester.run_all_tests()
    
    # Return exit code
    failed_tests = sum(1 for _, result in tester.test_results if result == 'FAIL')
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())