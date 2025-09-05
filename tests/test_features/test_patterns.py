"""
Fixed Test Script for Pattern Recognition Module
Fixes encoding issues, data loading, and edge cases
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Find project root and add to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import modules to test
from src.features.patterns import PatternRecognition, PatternSignal

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Use ASCII characters instead of Unicode for Windows compatibility
CHECK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


class TestPatternRecognition:
    """
    Comprehensive test suite for PatternRecognition class
    """
    
    def __init__(self, data_path: str = None, visualize: bool = False):
        """
        Initialize test suite
        """
        # Better path handling for your data structure
        if data_path is None:
            # Look for the project root
            possible_roots = [
                Path.cwd(),
                Path(__file__).parent.parent,
                Path.cwd().parent,
            ]
            
            # Find data directory
            data_found = False
            for root in possible_roots:
                if (root / 'data' / 'raw').exists():
                    data_path = root / 'data' / 'raw'
                    data_found = True
                    break
            
            if not data_found:
                # Try current directory structure
                if Path('data/raw').exists():
                    data_path = Path('data/raw')
                else:
                    data_path = Path('.')
                    print(f"{WARN} Data directory not found, using current directory")
        
        self.data_path = Path(data_path)
        print(f"Data path set to: {self.data_path}")
        
        self.pattern_recognizer = PatternRecognition(min_pattern_bars=5)
        self.test_results = []
        self.visualize = visualize
        self.detected_patterns = {}
        
    def run_all_tests(self):
        """Run all tests and report results"""
        print("=" * 80)
        print("PATTERN RECOGNITION TEST SUITE (FIXED)")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Min pattern bars: {self.pattern_recognizer.min_pattern_bars}\n")
        
        # Load test data
        print("Loading test data...")
        df = self.load_test_data()
        if df is None:
            print(f"{FAIL} Failed to load data. Stopping tests.")
            return
        
        # Test categories
        print("\n" + "="*50)
        print("CANDLESTICK PATTERNS")
        print("="*50)
        self.test_candlestick_patterns(df)
        
        print("\n" + "="*50)
        print("CHART PATTERNS")
        print("="*50)
        self.test_chart_patterns(df)
        
        print("\n" + "="*50)
        print("SUPPORT & RESISTANCE")
        print("="*50)
        self.test_support_resistance(df)
        
        print("\n" + "="*50)
        print("TREND PATTERNS")
        print("="*50)
        self.test_trend_patterns(df)
        
        print("\n" + "="*50)
        print("VOLUME PATTERNS")
        print("="*50)
        self.test_volume_patterns(df)
        
        print("\n" + "="*50)
        print("PRICE ACTION PATTERNS")
        print("="*50)
        self.test_price_action_patterns(df)
        
        print("\n" + "="*50)
        print("COMPLETE PATTERN DETECTION")
        print("="*50)
        self.test_complete_detection(df)
        
        print("\n" + "="*50)
        print("PATTERN STATISTICS")
        print("="*50)
        self.test_pattern_statistics(df)
        
        print("\n" + "="*50)
        print("EDGE CASES")
        print("="*50)
        self.test_edge_cases()
        
        print("\n" + "="*50)
        print("PERFORMANCE")
        print("="*50)
        self.test_performance(df)
        
        # Visualizations
        if self.visualize:
            print("\n" + "="*50)
            print("VISUALIZATIONS")
            print("="*50)
            self.create_visualizations(df)
        
        # Print summary
        self.print_test_summary()
    
    def load_test_data(self) -> pd.DataFrame:
        """Load or generate test OHLCV data"""
        try:
            # Updated paths for your data structure
            possible_paths = [
                self.data_path / '1d' / 'BTC_USDT_1d.csv',
                self.data_path / '1h' / 'BTC_USDT_1h.csv',
                self.data_path / '4h' / 'BTC_USDT_4h.csv',
                self.data_path / '15m' / 'BTC_USDT_15m.csv',
                self.data_path / '5m' / 'BTC_USDT_5m.csv',
                self.data_path / '1w' / 'BTC_USDT_1w.csv',
                # Alternative paths
                self.data_path / 'BTC_USDT_1d.csv',
                self.data_path / 'BTC_USDT.csv',
            ]
            
            df = None
            print("Searching for data files...")
            for path in possible_paths:
                if path.exists():
                    print(f"Found: {path}")
                    df = pd.read_csv(path)
                    print(f"{CHECK} Loaded data from {path.name}")
                    break
            
            if df is None:
                print(f"{WARN} No data file found in:")
                for path in possible_paths[:5]:
                    print(f"  - {path}")
                print("Generating sample data...")
                df = self.generate_sample_data(1000)
            
            # Handle different column name formats
            df.columns = df.columns.str.lower()
            
            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                print(f"{FAIL} Missing required columns: {missing}")
                print(f"Available columns: {list(df.columns)}")
                return None
            
            # Set datetime index if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif df.index.dtype == 'int64':
                # Create datetime index if none exists
                df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='1H')
            
            print(f"{CHECK} Data shape: {df.shape}")
            print(f"{CHECK} Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"{FAIL} Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate realistic OHLCV data with patterns"""
        np.random.seed(42)
        
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate price with trends and patterns
        t = np.linspace(0, 4*np.pi, periods)
        
        # Base trend with cycles
        trend = 50000 + 10000 * np.sin(t/2) + 5000 * np.sin(t*2)
        
        # Add noise
        noise = np.random.normal(0, 500, periods)
        
        # Create price series
        close = trend + noise
        
        # Ensure positive prices
        close = np.abs(close) + 40000
        
        # Generate OHLCV
        df = pd.DataFrame(index=timestamps)
        df['close'] = close
        
        # Generate realistic OHLC relationships
        daily_range = np.abs(np.random.normal(0.02, 0.01, periods))
        
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        df['open'] = df['open'] * (1 + np.random.normal(0, 0.005, periods))
        
        df['high'] = np.maximum(df['open'], df['close']) * (1 + daily_range * np.random.uniform(0.5, 1, periods))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - daily_range * np.random.uniform(0.5, 1, periods))
        
        # Generate volume (higher on big moves)
        price_change = np.abs(df['close'].pct_change()).fillna(0)
        df['volume'] = 1000000 * (1 + price_change * 20) * np.random.uniform(0.8, 1.2, periods)
        
        # Only add specific patterns if we have enough data
        if periods > 200:
            # Add a hammer pattern
            hammer_idx = min(100, periods - 10)
            df.loc[df.index[hammer_idx], 'low'] = df.loc[df.index[hammer_idx], 'close'] * 0.97
            df.loc[df.index[hammer_idx], 'open'] = df.loc[df.index[hammer_idx], 'close'] * 0.995
            
            # Add an engulfing pattern
            engulf_idx = min(200, periods - 10)
            df.loc[df.index[engulf_idx-1], 'close'] = df.loc[df.index[engulf_idx-1], 'open'] * 0.99
            df.loc[df.index[engulf_idx], 'open'] = df.loc[df.index[engulf_idx-1], 'close'] * 0.995
            df.loc[df.index[engulf_idx], 'close'] = df.loc[df.index[engulf_idx-1], 'open'] * 1.01
        
        return df
    
    def test_candlestick_patterns(self, df: pd.DataFrame):
        """Test candlestick pattern detection"""
        try:
            start_time = time.time()
            patterns = self.pattern_recognizer.detect_candlestick_patterns(df)
            elapsed = time.time() - start_time
            
            # Store for later use
            self.detected_patterns['candlestick'] = patterns
            
            # Validate structure
            assert isinstance(patterns, pd.DataFrame)
            assert len(patterns) == len(df)
            
            print(f"{CHECK} Detected candlestick patterns in {elapsed:.2f}s")
            print(f"   Pattern columns: {len(patterns.columns)}")
            
            # Count patterns detected
            pattern_counts = {}
            single_patterns = ['doji', 'hammer', 'inverted_hammer', 'shooting_star', 
                              'spinning_top', 'marubozu']
            
            for pattern in single_patterns:
                if pattern in patterns.columns:
                    count = patterns[pattern].sum()
                    if count > 0:
                        pattern_counts[pattern] = count
            
            if pattern_counts:
                print("\n   Single candle patterns detected:")
                for pattern, count in pattern_counts.items():
                    print(f"     {pattern}: {count}")
            
            # Check two-candle patterns
            two_patterns = ['engulfing_bullish', 'engulfing_bearish', 'harami_bullish', 
                           'harami_bearish', 'piercing_line', 'dark_cloud_cover']
            
            two_pattern_counts = {}
            for pattern in two_patterns:
                if pattern in patterns.columns:
                    count = patterns[pattern].sum()
                    if count > 0:
                        two_pattern_counts[pattern] = count
            
            if two_pattern_counts:
                print("\n   Two-candle patterns detected:")
                for pattern, count in two_pattern_counts.items():
                    print(f"     {pattern}: {count}")
            
            # Check bullish/bearish scores
            if 'candlestick_bullish_score' in patterns.columns:
                max_bull = patterns['candlestick_bullish_score'].max()
                avg_bull = patterns['candlestick_bullish_score'].mean()
                print(f"\n   Bullish score - Max: {max_bull:.0f}, Avg: {avg_bull:.2f}")
            
            if 'candlestick_bearish_score' in patterns.columns:
                max_bear = patterns['candlestick_bearish_score'].max()
                avg_bear = patterns['candlestick_bearish_score'].mean()
                print(f"   Bearish score - Max: {max_bear:.0f}, Avg: {avg_bear:.2f}")
            
            self.test_results.append(('Candlestick Patterns', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Candlestick patterns failed: {e}")
            self.test_results.append(('Candlestick Patterns', 'FAIL'))
    
    def test_chart_patterns(self, df: pd.DataFrame):
        """Test chart pattern detection"""
        try:
            start_time = time.time()
            patterns = self.pattern_recognizer.detect_chart_patterns(df, window=50)
            elapsed = time.time() - start_time
            
            self.detected_patterns['chart'] = patterns
            
            assert isinstance(patterns, pd.DataFrame)
            assert len(patterns) == len(df)
            
            print(f"{CHECK} Detected chart patterns in {elapsed:.2f}s")
            
            # Count various patterns
            pattern_types = {
                'Triangle': ['triangle_ascending', 'triangle_descending', 'triangle_symmetrical'],
                'Wedge': ['wedge_rising', 'wedge_falling'],
                'Flag': ['flag_bullish', 'flag_bearish'],
                'Double': ['double_top', 'double_bottom'],
                'H&S': ['head_shoulders', 'inverse_head_shoulders']
            }
            
            for pattern_type, pattern_list in pattern_types.items():
                counts = {}
                for pattern in pattern_list:
                    if pattern in patterns.columns:
                        count = patterns[pattern].sum()
                        if count > 0:
                            counts[pattern] = count
                
                if counts:
                    print(f"\n   {pattern_type} patterns:")
                    for pattern, count in counts.items():
                        print(f"     {pattern}: {count}")
            
            self.test_results.append(('Chart Patterns', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Chart patterns failed: {e}")
            self.test_results.append(('Chart Patterns', 'FAIL'))
    
    def test_support_resistance(self, df: pd.DataFrame):
        """Test support and resistance detection"""
        try:
            start_time = time.time()
            sr_levels = self.pattern_recognizer.detect_support_resistance(df, window=50, num_levels=5)
            elapsed = time.time() - start_time
            
            self.detected_patterns['support_resistance'] = sr_levels
            
            print(f"{CHECK} Detected support/resistance in {elapsed:.2f}s")
            
            # Display current levels
            current_price = df['close'].iloc[-1]
            print(f"\n   Current price: ${current_price:.2f}")
            
            # Show resistance levels
            for i in range(1, 4):
                col = f'resistance_{i}'
                if col in sr_levels.columns:
                    level = sr_levels[col].iloc[-1]
                    if not pd.isna(level):
                        distance = (level - current_price) / current_price * 100
                        print(f"   Resistance {i}: ${level:.2f} ({distance:+.1f}%)")
            
            # Show support levels
            for i in range(1, 4):
                col = f'support_{i}'
                if col in sr_levels.columns:
                    level = sr_levels[col].iloc[-1]
                    if not pd.isna(level):
                        distance = (level - current_price) / current_price * 100
                        print(f"   Support {i}: ${level:.2f} ({distance:+.1f}%)")
            
            self.test_results.append(('Support/Resistance', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Support/Resistance failed: {e}")
            self.test_results.append(('Support/Resistance', 'FAIL'))
    
    def test_trend_patterns(self, df: pd.DataFrame):
        """Test trend pattern detection"""
        try:
            start_time = time.time()
            patterns = self.pattern_recognizer.detect_trend_patterns(df, window=20)
            elapsed = time.time() - start_time
            
            self.detected_patterns['trend'] = patterns
            
            print(f"{CHECK} Detected trend patterns in {elapsed:.2f}s")
            
            # Market classification
            if all(col in patterns.columns for col in ['uptrend', 'downtrend', 'sideways']):
                uptrend_pct = patterns['uptrend'].mean() * 100
                downtrend_pct = patterns['downtrend'].mean() * 100
                sideways_pct = patterns['sideways'].mean() * 100
                
                print(f"\n   Market classification:")
                print(f"     Uptrend: {uptrend_pct:.1f}%")
                print(f"     Downtrend: {downtrend_pct:.1f}%")
                print(f"     Sideways: {sideways_pct:.1f}%")
                
                # Current trend
                if patterns['uptrend'].iloc[-1]:
                    current = "UPTREND"
                elif patterns['downtrend'].iloc[-1]:
                    current = "DOWNTREND"
                else:
                    current = "SIDEWAYS"
                print(f"     Current: {current}")
            
            self.test_results.append(('Trend Patterns', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Trend patterns failed: {e}")
            self.test_results.append(('Trend Patterns', 'FAIL'))
    
    def test_volume_patterns(self, df: pd.DataFrame):
        """Test volume pattern detection"""
        try:
            start_time = time.time()
            patterns = self.pattern_recognizer.detect_volume_patterns(df)
            elapsed = time.time() - start_time
            
            self.detected_patterns['volume'] = patterns
            
            print(f"{CHECK} Detected volume patterns in {elapsed:.2f}s")
            
            # Volume analysis
            if 'volume_spike' in patterns.columns:
                spike_count = patterns['volume_spike'].sum()
                spike_pct = spike_count / len(patterns) * 100
                print(f"\n   Volume spikes: {spike_count} ({spike_pct:.1f}%)")
            
            if 'accumulation' in patterns.columns and 'distribution' in patterns.columns:
                acc_count = patterns['accumulation'].sum()
                dist_count = patterns['distribution'].sum()
                print(f"   Accumulation: {acc_count} periods")
                print(f"   Distribution: {dist_count} periods")
                
                if acc_count > dist_count:
                    print(f"   Net: ACCUMULATION (+{acc_count - dist_count})")
                else:
                    print(f"   Net: DISTRIBUTION (-{dist_count - acc_count})")
            
            self.test_results.append(('Volume Patterns', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Volume patterns failed: {e}")
            self.test_results.append(('Volume Patterns', 'FAIL'))
    
    def test_price_action_patterns(self, df: pd.DataFrame):
        """Test price action pattern detection"""
        try:
            start_time = time.time()
            patterns = self.pattern_recognizer.detect_price_action_patterns(df)
            elapsed = time.time() - start_time
            
            self.detected_patterns['price_action'] = patterns
            
            print(f"{CHECK} Detected price action patterns in {elapsed:.2f}s")
            
            # Price action summary
            pattern_types = {
                'Pin Bars': ['pin_bar_bullish', 'pin_bar_bearish'],
                'Bar Types': ['inside_bar', 'outside_bar'],
                'Rejections': ['rejection_upper', 'rejection_lower'],
                'Momentum': ['momentum_bullish', 'momentum_bearish'],
                'Gaps': ['gap_up', 'gap_down', 'gap_filled']
            }
            
            for pattern_type, pattern_list in pattern_types.items():
                counts = []
                for pattern in pattern_list:
                    if pattern in patterns.columns:
                        count = patterns[pattern].sum()
                        if count > 0:
                            counts.append(f"{pattern}: {count}")
                
                if counts:
                    print(f"\n   {pattern_type}:")
                    for count_str in counts:
                        print(f"     {count_str}")
            
            self.test_results.append(('Price Action Patterns', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Price action patterns failed: {e}")
            import traceback
            traceback.print_exc()
            self.test_results.append(('Price Action Patterns', 'FAIL'))
    
    def test_complete_detection(self, df: pd.DataFrame):
        """Test complete pattern detection"""
        try:
            start_time = time.time()
            all_patterns = self.pattern_recognizer.detect_all_patterns(df)
            elapsed = time.time() - start_time
            
            print(f"{CHECK} Complete pattern detection in {elapsed:.2f}s")
            print(f"   Total pattern features: {len(all_patterns.columns)}")
            
            # Test specific patterns are present
            expected_categories = [
                'doji', 'hammer',  # Candlestick
                'triangle_ascending',  # Chart
                'resistance_1',  # S/R
                'uptrend',  # Trend
                'volume_spike',  # Volume
                'inside_bar'  # Price action
            ]
            
            missing = []
            for pattern in expected_categories:
                if pattern not in all_patterns.columns:
                    missing.append(pattern)
            
            if missing:
                print(f"   {WARN} Missing expected patterns: {missing}")
            else:
                print(f"   {CHECK} All expected pattern categories present")
            
            self.test_results.append(('Complete Detection', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Complete detection failed: {e}")
            self.test_results.append(('Complete Detection', 'FAIL'))
    
    def test_pattern_statistics(self, df: pd.DataFrame):
        """Test pattern statistics and correlations"""
        try:
            if not self.detected_patterns:
                print(f"{WARN} No patterns detected to analyze")
                return
            
            print("Analyzing pattern statistics...")
            
            # Count total patterns
            total_patterns = 0
            for category, patterns in self.detected_patterns.items():
                if patterns is not None and not patterns.empty:
                    for col in patterns.columns:
                        if col not in ['candlestick_bullish_score', 'candlestick_bearish_score']:
                            if patterns[col].dtype in [int, float]:
                                total_patterns += patterns[col].sum()
            
            print(f"\n   Total patterns detected: {int(total_patterns)}")
            
            # Analyze pattern correlations
            if 'candlestick' in self.detected_patterns and 'volume' in self.detected_patterns:
                candle_df = self.detected_patterns['candlestick']
                volume_df = self.detected_patterns['volume']
                
                if 'hammer' in candle_df.columns and 'volume_spike' in volume_df.columns:
                    # Check if hammers often coincide with volume spikes
                    hammer_with_volume = (candle_df['hammer'] & volume_df['volume_spike']).sum()
                    total_hammers = candle_df['hammer'].sum()
                    
                    if total_hammers > 0:
                        pct = (hammer_with_volume / total_hammers) * 100
                        print(f"   Hammers with volume spikes: {hammer_with_volume}/{total_hammers} ({pct:.1f}%)")
            
            self.test_results.append(('Pattern Statistics', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Pattern statistics failed: {e}")
            self.test_results.append(('Pattern Statistics', 'FAIL'))
    
    def test_edge_cases(self):
        """Test edge cases - FIXED"""
        try:
            print("Testing edge cases...")
            
            # 1. Minimal data
            print("\n   Testing with minimal data...")
            small_df = self.generate_sample_data(10)
            patterns = self.pattern_recognizer.detect_all_patterns(small_df)
            assert len(patterns) == len(small_df)
            print(f"   {CHECK} Handled minimal data")
            
            # 2. Constant prices
            print("\n   Testing with constant prices...")
            const_df = self.generate_sample_data(100)
            const_df['open'] = const_df['close'] = const_df['high'] = const_df['low'] = 50000
            patterns = self.pattern_recognizer.detect_candlestick_patterns(const_df)
            print(f"   {CHECK} Handled constant prices")
            
            # 3. Extreme volatility
            print("\n   Testing with extreme volatility...")
            volatile_df = self.generate_sample_data(100)
            volatile_df['high'] = volatile_df['close'] * 1.5
            volatile_df['low'] = volatile_df['close'] * 0.5
            patterns = self.pattern_recognizer.detect_all_patterns(volatile_df)
            print(f"   {CHECK} Handled extreme volatility")
            
            # 4. Missing values - FIXED: Use iloc for position-based indexing
            print("\n   Testing with missing values...")
            nan_df = self.generate_sample_data(100)
            # Use iloc for position-based indexing on datetime index
            nan_df.iloc[10:20, nan_df.columns.get_loc('close')] = np.nan
            nan_df.iloc[30:35, nan_df.columns.get_loc('volume')] = np.nan
            patterns = self.pattern_recognizer.detect_all_patterns(nan_df)
            print(f"   {CHECK} Handled missing values")
            
            # 5. Extreme price levels
            print("\n   Testing with extreme price levels...")
            extreme_df = self.generate_sample_data(100)
            extreme_df *= 1000000  # Very high prices
            patterns = self.pattern_recognizer.detect_all_patterns(extreme_df)
            print(f"   {CHECK} Handled extreme price levels")
            
            self.test_results.append(('Edge Cases', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Edge cases failed: {e}")
            self.test_results.append(('Edge Cases', 'FAIL'))
    
    def test_performance(self, df: pd.DataFrame):
        """Test performance with different data sizes"""
        try:
            print("Running performance benchmarks...")
            
            sizes = [100, 500, 1000]
            times = {}
            
            for size in sizes:
                if size <= len(df):
                    test_df = df.iloc[:size].copy()
                else:
                    test_df = self.generate_sample_data(size)
                
                start = time.time()
                _ = self.pattern_recognizer.detect_all_patterns(test_df)
                elapsed = time.time() - start
                
                times[size] = elapsed
                patterns_per_sec = len(test_df) / elapsed
                
                print(f"\n   {size} rows: {elapsed:.2f}s ({patterns_per_sec:.0f} rows/sec)")
            
            # Check scaling
            if 100 in times and 500 in times:
                scaling = times[500] / times[100]
                print(f"\n   Scaling factor (100->500): {scaling:.1f}x")
                
                if scaling < 10:
                    print(f"   {CHECK} Good performance scaling")
                else:
                    print(f"   {WARN} Poor scaling performance")
            
            # Memory usage test
            print("\n   Testing memory efficiency...")
            import tracemalloc
            tracemalloc.start()
            
            test_df = self.generate_sample_data(1000)
            _ = self.pattern_recognizer.detect_all_patterns(test_df)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"   Peak memory usage: {peak / 1024 / 1024:.1f} MB")
            
            self.test_results.append(('Performance', 'PASS'))
            
        except Exception as e:
            print(f"{FAIL} Performance test failed: {e}")
            self.test_results.append(('Performance', 'FAIL'))
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create pattern visualizations"""
        try:
            print("\nCreating visualizations...")
            
            # Take last 100 periods for visualization
            viz_df = df.iloc[-min(100, len(df)):].copy()
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 9))
            
            # Plot 1: Price with patterns
            axes[0].plot(viz_df.index, viz_df['close'], 'b-', linewidth=1, label='Close')
            axes[0].fill_between(viz_df.index, viz_df['low'], viz_df['high'], alpha=0.2, color='gray')
            axes[0].set_title('Price Chart with High/Low Range')
            axes[0].set_ylabel('Price ($)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            
            # Plot 2: Volume
            colors = ['g' if c > o else 'r' for c, o in zip(viz_df['close'], viz_df['open'])]
            axes[1].bar(viz_df.index, viz_df['volume'], color=colors, alpha=0.5)
            axes[1].set_title('Volume (Green=Up, Red=Down)')
            axes[1].set_ylabel('Volume')
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Pattern detection summary
            if self.detected_patterns:
                pattern_counts = {}
                for category, patterns in self.detected_patterns.items():
                    if patterns is not None and not patterns.empty:
                        pattern_counts[category] = len(patterns.columns)
                
                if pattern_counts:
                    categories = list(pattern_counts.keys())
                    counts = list(pattern_counts.values())
                    
                    axes[2].bar(categories, counts, color='skyblue')
                    axes[2].set_title('Pattern Features by Category')
                    axes[2].set_ylabel('Number of Features')
                    axes[2].set_xlabel('Category')
                    
                    # Add value labels on bars
                    for i, (cat, count) in enumerate(zip(categories, counts)):
                        axes[2].text(i, count + 0.5, str(count), ha='center', va='bottom')
            else:
                axes[2].text(0.5, 0.5, 'No patterns detected', 
                           ha='center', va='center', transform=axes[2].transAxes)
                axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_file = 'pattern_visualization.png'
            plt.savefig(output_file, dpi=100, bbox_inches='tight')
            print(f"{CHECK} Saved visualization to {output_file}")
            
            plt.close()
            
        except Exception as e:
            print(f"{WARN} Visualization failed: {e}")
    
    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, result in self.test_results if result == 'PASS')
        failed = sum(1 for _, result in self.test_results if result == 'FAIL')
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} {CHECK}")
        print(f"Failed: {failed} {FAIL}")
        print(f"Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results:
            symbol = CHECK if result == "PASS" else FAIL
            print(f"  {symbol} {test_name}: {result}")
        
        print("\n" + "=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Pattern Recognition')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--min-bars', type=int, default=5,
                       help='Minimum bars for pattern formation')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TestPatternRecognition(args.data_path, args.visualize)
    
    # Configure pattern recognizer
    tester.pattern_recognizer.min_pattern_bars = args.min_bars
    
    # Run all tests
    tester.run_all_tests()
    
    # Return exit code
    failed_tests = sum(1 for _, result in tester.test_results if result == 'FAIL')
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())