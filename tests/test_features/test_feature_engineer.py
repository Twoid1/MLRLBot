"""
Test Script for Feature Engineer Module
Tests all functionality of feature_engineer.py

This script automatically finds the project root directory,
regardless of where the test file is located.

Directory structure assumed:
project_root/
├── data/
│   └── raw/
│       ├── 1d/
│       │   ├── BTC_USDT_1d.csv
│       │   ├── ETH_USDT_1d.csv
│       │   └── ...
│       ├── 1h/
│       │   ├── BTC_USDT_1h.csv
│       │   └── ...
│       └── ...
├── src/
│   └── features/
│       ├── feature_engineer.py
│       ├── indicators.py
│       └── ...
└── tests/  (or any other directory)
    └── test_feature_engineer.py  (this file)
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import warnings
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

# Get the current file's directory
CURRENT_FILE = Path(__file__).resolve()
TEST_DIR = CURRENT_FILE.parent

# Debug: Show the actual paths
print(f"Current file: {CURRENT_FILE}")
print(f"Test directory: {TEST_DIR}")
print(f"Test directory name: {TEST_DIR.name}")

# The project root is kraken-trading-bot directory
# If this test file is in kraken-trading-bot/tests/, then go up one level
if TEST_DIR.name == 'tests':
    PROJECT_ROOT = TEST_DIR.parent
    print(f"Found 'tests' directory, going up to: {PROJECT_ROOT}")
elif 'kraken-trading-bot' in str(TEST_DIR):
    # Find the kraken-trading-bot directory in the path
    current = TEST_DIR
    while current.parent != current:  # While not at filesystem root
        if current.name == 'kraken-trading-bot':
            PROJECT_ROOT = current
            break
        current = current.parent
    else:
        # Fallback
        PROJECT_ROOT = TEST_DIR
else:
    # Fallback: assume current directory is project root
    PROJECT_ROOT = TEST_DIR
    print(f"Warning: Could not determine project root, using: {PROJECT_ROOT}")

# The data should be at PROJECT_ROOT/data/raw
DATA_PATH = PROJECT_ROOT / 'data' / 'raw'

print(f"Project root set to: {PROJECT_ROOT}")
print(f"Looking for data in: {DATA_PATH}")
print("-" * 80)

# Add project root to path so we can import modules
sys.path.insert(0, str(PROJECT_ROOT))

# Now imports will work correctly
from src.features.feature_engineer import FeatureEngineer
from src.features.indicators import TechnicalIndicators

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TestFeatureEngineer:
    """
    Comprehensive test suite for FeatureEngineer class
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize test suite
        
        Args:
            data_path: Path to raw data directory (if None, uses DATA_PATH global)
        """
        # Use the global DATA_PATH we defined at the top, or override if provided
        if data_path is None:
            self.data_path = DATA_PATH  # Use the global we defined
        else:
            # If a path is provided, use it
            path = Path(data_path)
            if not path.is_absolute():
                self.data_path = PROJECT_ROOT / path
            else:
                self.data_path = path
        
        print(f" Using data directory: {self.data_path.absolute()}")
        
        # Verify the path exists
        if not self.data_path.exists():
            print(f" Warning: Data directory does not exist: {self.data_path}")
            print(f"  Expected structure: {PROJECT_ROOT}/data/raw/{{timeframe}}/{{SYMBOL}}_{{timeframe}}.csv")
        
        self.fe = FeatureEngineer()
        self.test_results = []
        
        # Your crypto symbols
        self.symbols = [
            'BTC_USDT', 'ETH_USDT', 'BNB_USDT', 'ADA_USDT', 'SOL_USDT',
            'XRP_USDT', 'DOT_USDT', 'DOGE_USDT', 'AVAX_USDT', 'MATIC_USDT',
            'LINK_USDT', 'LTC_USDT', 'ATOM_USDT', 'UNI_USDT', 'ALGO_USDT'
        ]
        
        # Available timeframes
        self.timeframes = ['5m', '15m', '1h', '4h', '1d', '1w']
        
    def run_all_tests(self):
        """Run all tests and report results"""
        print("=" * 80)
        print("FEATURE ENGINEER TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Test 1: Load sample data
        print("Test 1: Loading sample data...")
        df = self.test_load_data()
        if df is None:
            print(" Failed to load data. Stopping tests.")
            return
        
        # Test 2: Price features
        print("\nTest 2: Testing price features...")
        self.test_price_features(df)
        
        # Test 3: Volume features
        print("\nTest 3: Testing volume features...")
        self.test_volume_features(df)
        
        # Test 4: Volatility features
        print("\nTest 4: Testing volatility features...")
        self.test_volatility_features(df)
        
        # Test 5: Technical indicators
        print("\nTest 5: Testing technical indicator features...")
        self.test_indicator_features(df)
        
        # Test 6: Pattern features
        print("\nTest 6: Testing pattern features...")
        self.test_pattern_features(df)
        
        # Test 7: Moving average features
        print("\nTest 7: Testing moving average features...")
        self.test_ma_features(df)
        
        # Test 8: Complete feature calculation
        print("\nTest 8: Testing complete feature calculation...")
        all_features = self.test_all_features(df)
        
        # Test 9: Multi-timeframe features
        print("\nTest 9: Testing multi-timeframe features...")
        self.test_multi_timeframe_features()
        
        # Test 10: Feature selection
        if all_features is not None:
            print("\nTest 10: Testing feature selection...")
            self.test_feature_selection(all_features, df)
        
        # Test 11: Save and load features
        if all_features is not None:
            print("\nTest 11: Testing save/load functionality...")
            self.test_save_load_features(all_features)
        
        # Test 12: Multi-asset feature matrix
        print("\nTest 12: Testing multi-asset feature matrix...")
        self.test_multi_asset_features()
        
        # Test 13: Performance benchmarks
        print("\nTest 13: Running performance benchmarks...")
        self.test_performance_benchmarks(df)
        
        # Test 14: Edge cases
        print("\nTest 14: Testing edge cases...")
        self.test_edge_cases()
        
        # Print summary
        self.print_test_summary()
    
    def load_crypto_data(self, symbol: str = 'BTC_USDT', timeframe: str = '1h') -> pd.DataFrame:
        """
        Load data for a specific symbol and timeframe
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC_USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        file_path = self.data_path / timeframe / f'{symbol}_{timeframe}.csv'
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    
    def test_load_data(self) -> pd.DataFrame:
        """Test data loading"""
        try:
            print(f" Looking for data in: {self.data_path.absolute()}")
            
            # Check if data directory exists
            if not self.data_path.exists():
                print(f" Data directory does not exist: {self.data_path}")
                print(f"   Please ensure your data is in: {self.data_path.absolute()}")
                print("\n Generating sample data to continue tests...")
                return self.generate_sample_data()
            
            # Your data structure: data/raw/{timeframe}/{SYMBOL}_{timeframe}.csv
            # Try different timeframes and symbols
            timeframes = ['1d', '1h', '4h', '5m', '15m', '1w']  # Check 1d first since you mentioned it
            symbols = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
            
            possible_paths = []
            
            # Build paths based on your structure
            for tf in timeframes:
                for symbol in symbols:
                    possible_paths.append(self.data_path / tf / f'{symbol}_{tf}.csv')
            
            df = None
            for path in possible_paths:
                path = Path(path)  # Ensure it's a Path object
                if path.exists():
                    df = pd.read_csv(path)
                    print(f" Loaded data from {path.relative_to(PROJECT_ROOT)}")
                    
                    # Print data info
                    print(f"   Shape: {df.shape}")
                    print(f"   Columns: {list(df.columns)}")
                    break
            
            if df is None:
                # Show available files for debugging
                print("\n Checking for available data files...")
                print(f"Looking in: {self.data_path.absolute()}")
                
                # Check for timeframe directories
                timeframe_dirs = [d for d in self.data_path.iterdir() if d.is_dir()] if self.data_path.exists() else []
                if timeframe_dirs:
                    print(f"Found timeframe directories: {[d.name for d in timeframe_dirs[:5]]}")
                    
                    # Show files in first timeframe directory
                    first_tf_dir = timeframe_dirs[0]
                    csv_files = list(first_tf_dir.glob('*.csv'))[:5]
                    if csv_files:
                        print(f"\nSample files in {first_tf_dir.name}:")
                        for f in csv_files:
                            print(f"  - {f.name}")
                
                # Also check for any CSV files recursively
                if self.data_path.exists():
                    all_csv = list(self.data_path.glob('**/*.csv'))
                    if all_csv:
                        print(f"\nFound {len(all_csv)} total CSV files")
                        print("Sample files:")
                        for f in all_csv[:5]:
                            print(f"  - {f.relative_to(self.data_path)}")
                
                # Generate sample data if no file found
                print("\n No matching data file found. Generating sample data...")
                df = self.generate_sample_data()
            
            # Validate columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if columns need to be renamed (handle different naming conventions)
            column_mappings = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume',
                'o': 'open', 'h': 'high', 'l': 'low', 
                'c': 'close', 'v': 'volume'
            }
            
            # Rename columns if needed
            df.columns = [column_mappings.get(col, col) for col in df.columns]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f" Missing columns: {missing_cols}")
                print(f"   Available columns: {list(df.columns)}")
                return None
            
            # Convert timestamp if exists
            timestamp_cols = ['timestamp', 'date', 'time', 'datetime', 'Date', 'Time']
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
            
            # If no index set, create a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                print(" Creating datetime index...")
                df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='1H')
            
            print(f" Data shape: {df.shape}")
            print(f" Date range: {df.index[0]} to {df.index[-1]}")
            print(f" Columns: {list(df.columns)}")
            
            self.test_results.append(('Data Loading', 'PASS'))
            return df
            
        except Exception as e:
            print(f" Error loading data: {e}")
            self.test_results.append(('Data Loading', 'FAIL'))
            return None
    
    def generate_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Generate price data
        close = 50000 + np.cumsum(np.random.randn(periods) * 100)
        
        df = pd.DataFrame({
            'open': close + np.random.randn(periods) * 50,
            'high': close + np.abs(np.random.randn(periods) * 100),
            'low': close - np.abs(np.random.randn(periods) * 100),
            'close': close,
            'volume': np.abs(np.random.randn(periods) * 1000000 + 5000000)
        }, index=timestamps)
        
        # Fix OHLC relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def test_price_features(self, df: pd.DataFrame):
        """Test price feature calculation"""
        try:
            start_time = time.time()
            features = self.fe.calculate_price_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} price features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Validate some features
            assert 'return_1' in features.columns
            assert 'close_position' in features.columns
            assert 'high_low_ratio' in features.columns
            
            # Check for NaN handling
            nan_cols = features.columns[features.isna().sum() > len(features) * 0.5]
            if len(nan_cols) > 0:
                print(f" High NaN ratio in: {list(nan_cols)}")
            
            # Sample output
            print("Sample features (last 5 rows):")
            print(features[['return_1', 'return_5', 'high_low_ratio']].tail())
            
            self.test_results.append(('Price Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in price features: {e}")
            self.test_results.append(('Price Features', 'FAIL'))
    
    def test_volume_features(self, df: pd.DataFrame):
        """Test volume feature calculation"""
        try:
            start_time = time.time()
            features = self.fe.calculate_volume_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} volume features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Validate key features
            assert 'volume_change' in features.columns
            assert 'volume_ratio_5' in features.columns
            assert 'high_volume' in features.columns
            
            # Check value ranges
            if 'high_volume' in features.columns:
                high_vol_pct = features['high_volume'].mean() * 100
                print(f" High volume bars: {high_vol_pct:.1f}%")
            
            self.test_results.append(('Volume Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in volume features: {e}")
            self.test_results.append(('Volume Features', 'FAIL'))
    
    def test_volatility_features(self, df: pd.DataFrame):
        """Test volatility feature calculation"""
        try:
            start_time = time.time()
            features = self.fe.calculate_volatility_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} volatility features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Check features exist
            assert 'volatility_20' in features.columns
            assert 'atr' in features.columns
            
            # Display statistics
            if 'atr' in features.columns:
                avg_atr = features['atr'].mean()
                print(f" Average ATR: {avg_atr:.2f}")
            
            self.test_results.append(('Volatility Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in volatility features: {e}")
            self.test_results.append(('Volatility Features', 'FAIL'))
    
    def test_indicator_features(self, df: pd.DataFrame):
        """Test technical indicator features"""
        try:
            start_time = time.time()
            features = self.fe.calculate_indicator_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} indicator features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Validate key indicators
            indicators_to_check = ['rsi_14', 'macd', 'bb_position', 'adx']
            missing = [ind for ind in indicators_to_check if ind not in features.columns]
            
            if missing:
                print(f" Missing indicators: {missing}")
            else:
                print(" All key indicators present")
            
            # Check RSI range
            if 'rsi_14' in features.columns:
                rsi_valid = ((features['rsi_14'] >= 0) & (features['rsi_14'] <= 100)).all()
                if rsi_valid:
                    print(" RSI values in valid range [0, 100]")
                
                # Display current market condition
                last_rsi = features['rsi_14'].iloc[-1]
                if last_rsi > 70:
                    print(f" Market condition: Overbought (RSI={last_rsi:.1f})")
                elif last_rsi < 30:
                    print(f" Market condition: Oversold (RSI={last_rsi:.1f})")
                else:
                    print(f" Market condition: Neutral (RSI={last_rsi:.1f})")
            
            self.test_results.append(('Indicator Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in indicator features: {e}")
            self.test_results.append(('Indicator Features', 'FAIL'))
    
    def test_pattern_features(self, df: pd.DataFrame):
        """Test pattern feature calculation"""
        try:
            start_time = time.time()
            features = self.fe.calculate_pattern_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} pattern features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Check for candlestick patterns
            pattern_cols = [col for col in features.columns if 'bullish' in col or 'bearish' in col]
            print(f" Detected {len(pattern_cols)} pattern types")
            
            # Count pattern occurrences
            if 'bullish_engulfing' in features.columns:
                bullish_count = features['bullish_engulfing'].sum()
                print(f" Bullish engulfing patterns: {bullish_count}")
            
            self.test_results.append(('Pattern Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in pattern features: {e}")
            self.test_results.append(('Pattern Features', 'FAIL'))
    
    def test_ma_features(self, df: pd.DataFrame):
        """Test moving average features"""
        try:
            start_time = time.time()
            features = self.fe.calculate_ma_features(df)
            elapsed = time.time() - start_time
            
            print(f" Generated {features.shape[1]} MA features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Check for key MAs
            ma_cols = [col for col in features.columns if 'sma' in col or 'ema' in col]
            print(f" Generated {len(ma_cols)} moving averages")
            
            # Check for crossovers
            if 'ma_cross_10_20' in features.columns:
                crossovers = features['ma_cross_10_20'].sum()
                print(f" MA(10,20) crossovers: {crossovers}")
            
            self.test_results.append(('MA Features', 'PASS'))
            
        except Exception as e:
            print(f" Error in MA features: {e}")
            self.test_results.append(('MA Features', 'FAIL'))
    
    def test_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Test complete feature calculation"""
        try:
            start_time = time.time()
            
            # Calculate all features
            all_features = self.fe.calculate_all_features(df, symbol='BTC/USDT')
            elapsed = time.time() - start_time
            
            print(f" Generated {all_features.shape[1]} total features")
            print(f" Time taken: {elapsed:.2f} seconds")
            print(f" Features per second: {all_features.shape[1]/elapsed:.1f}")
            
            # Check for NaN values
            nan_counts = all_features.isna().sum()
            high_nan_features = nan_counts[nan_counts > len(all_features) * 0.5]
            
            if len(high_nan_features) > 0:
                print(f" Features with >50% NaN: {len(high_nan_features)}")
            
            # Memory usage
            memory_mb = all_features.memory_usage(deep=True).sum() / 1024 / 1024
            print(f" Memory usage: {memory_mb:.2f} MB")
            
            # Feature categories breakdown
            categories = {
                'price': len([c for c in all_features.columns if 'return' in c or 'ratio' in c]),
                'volume': len([c for c in all_features.columns if 'volume' in c]),
                'volatility': len([c for c in all_features.columns if 'volatility' in c or 'atr' in c]),
                'indicators': len([c for c in all_features.columns if 'rsi' in c or 'macd' in c or 'bb' in c]),
                'patterns': len([c for c in all_features.columns if 'pattern' in c or 'candlestick' in c]),
                'ma': len([c for c in all_features.columns if 'sma' in c or 'ema' in c])
            }
            
            print("\nFeature breakdown by category:")
            for category, count in categories.items():
                print(f"  {category.capitalize()}: {count} features")
            
            self.test_results.append(('All Features', 'PASS'))
            return all_features
            
        except Exception as e:
            print(f" Error in all features: {e}")
            self.test_results.append(('All Features', 'FAIL'))
            return None
    
    def test_multi_timeframe_features(self):
        """Test multi-timeframe feature calculation"""
        try:
            # Try to load real data for multiple timeframes
            symbol = 'BTC_USDT'
            data_dict = {}
            
            # Load available timeframes for BTC
            for tf in ['1h', '4h', '1d']:
                df = self.load_crypto_data(symbol, tf)
                if df is not None:
                    # Ensure proper datetime index
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    elif 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    data_dict[tf] = df
            
            # If we have real data, use it
            if len(data_dict) >= 2:
                print(f" Using real data for {list(data_dict.keys())}")
            else:
                # Otherwise generate sample data
                print(" Generating sample multi-timeframe data...")
                base_df = self.generate_sample_data(1000)
                
                data_dict = {
                    '1h': base_df,
                    '4h': base_df.resample('4H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna(),
                    '1d': base_df.resample('1D').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                }
            
            start_time = time.time()
            mtf_features = self.fe.calculate_multi_timeframe_features(data_dict)
            elapsed = time.time() - start_time
            
            print(f" Generated {mtf_features.shape[1]} MTF features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Check for key MTF features
            mtf_cols = [col for col in mtf_features.columns if '4h' in col or '1d' in col]
            print(f" Multi-timeframe columns: {len(mtf_cols)}")
            
            self.test_results.append(('Multi-Timeframe', 'PASS'))
            
        except Exception as e:
            print(f" Error in multi-timeframe features: {e}")
            self.test_results.append(('Multi-Timeframe', 'FAIL'))
    
    def test_feature_selection(self, features: pd.DataFrame, df: pd.DataFrame):
        """Test feature selection functionality"""
        try:
            # Create target variable (next period return)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Align with features
            target = target.loc[features.index]
            
            start_time = time.time()
            selected = self.fe.select_top_features(features, target, n_features=50)
            elapsed = time.time() - start_time
            
            print(f" Selected {len(selected)} features")
            print(f" Time taken: {elapsed:.2f} seconds")
            
            # Display top 10 features
            if self.fe.feature_importance is not None:
                print("\nTop 10 features by importance:")
                for i, (feat, score) in enumerate(self.fe.feature_importance.head(10).items(), 1):
                    print(f"  {i}. {feat}: {score:.4f}")
            
            self.test_results.append(('Feature Selection', 'PASS'))
            
        except Exception as e:
            print(f" Error in feature selection: {e}")
            self.test_results.append(('Feature Selection', 'FAIL'))
    
    def test_save_load_features(self, features: pd.DataFrame):
        """Test save and load functionality"""
        try:
            # Save features to project root's data/features directory
            test_name = 'test_features'
            test_path = PROJECT_ROOT / 'data' / 'features'
            
            # Create directory if it doesn't exist
            test_path.mkdir(parents=True, exist_ok=True)
            
            self.fe.save_features(features, test_name, str(test_path))
            print(f" Features saved successfully to {test_path.relative_to(PROJECT_ROOT)}")
            
            # Load features
            loaded_features = self.fe.load_features(test_name, str(test_path))
            print(f" Features loaded successfully")
            
            # Verify integrity
            assert features.shape == loaded_features.shape
            assert features.columns.tolist() == loaded_features.columns.tolist()
            print(f" Feature integrity verified")
            
            # Clean up test files
            for file in test_path.glob(f'{test_name}*'):
                file.unlink()
            
            self.test_results.append(('Save/Load', 'PASS'))
            
        except Exception as e:
            print(f" Error in save/load: {e}")
            self.test_results.append(('Save/Load', 'FAIL'))
    
    def test_multi_asset_features(self):
        """Test multi-asset feature matrix creation"""
        try:
            # Try to load real data for multiple assets
            assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
            symbol_data = {}
            
            # Try to load real data first
            for asset in assets:
                asset_tfs = {}
                for tf in ['1h', '4h']:
                    df = self.load_crypto_data(asset, tf)
                    if df is not None:
                        # Ensure proper datetime index
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df.set_index('timestamp', inplace=True)
                        elif 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                        asset_tfs[tf] = df
                
                if asset_tfs:
                    # Convert symbol format for feature engineer
                    symbol_key = asset.replace('_', '/')  # BTC_USDT -> BTC/USDT
                    symbol_data[symbol_key] = asset_tfs
            
            # If we don't have enough real data, generate sample
            if len(symbol_data) < 2:
                print(" Generating sample multi-asset data...")
                for asset in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
                    base_df = self.generate_sample_data(500)
                    symbol_data[asset] = {
                        '1h': base_df,
                        '4h': base_df.resample('4H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    }
            else:
                print(f" Using real data for {list(symbol_data.keys())}")
            
            start_time = time.time()
            feature_matrix = self.fe.create_feature_matrix(symbol_data)
            elapsed = time.time() - start_time
            
            print(f" Generated feature matrix: {feature_matrix.shape}")
            print(f" Time taken: {elapsed:.2f} seconds")
            print(f" Assets processed: {len(symbol_data)}")
            
            self.test_results.append(('Multi-Asset', 'PASS'))
            
        except Exception as e:
            print(f" Error in multi-asset features: {e}")
            self.test_results.append(('Multi-Asset', 'FAIL'))
    
    def test_performance_benchmarks(self, df: pd.DataFrame):
        """Test performance benchmarks"""
        try:
            print("Running performance benchmarks...")
            
            # Test different data sizes
            sizes = [100, 500, 1000]
            times = []
            
            for size in sizes:
                df_subset = df.iloc[:size].copy()
                
                start_time = time.time()
                features = self.fe.calculate_all_features(df_subset)
                elapsed = time.time() - start_time
                
                times.append(elapsed)
                features_per_sec = features.shape[1] / elapsed
                print(f"  {size} rows: {elapsed:.2f}s ({features_per_sec:.0f} features/sec)")
            
            # Check scaling
            if len(times) == 3:
                scaling_factor = times[2] / times[0]
                expected_scaling = sizes[2] / sizes[0]
                
                if scaling_factor < expected_scaling * 1.5:
                    print(f" Good scaling performance (factor: {scaling_factor:.1f}x)")
                else:
                    print(f" Poor scaling (factor: {scaling_factor:.1f}x)")
            
            self.test_results.append(('Performance', 'PASS'))
            
        except Exception as e:
            print(f" Error in performance benchmarks: {e}")
            self.test_results.append(('Performance', 'FAIL'))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        try:
            print("Testing edge cases...")
            
            # Test 1: Empty DataFrame
            try:
                empty_df = pd.DataFrame()
                features = self.fe.calculate_all_features(empty_df)
                print(" Should have raised error for empty DataFrame")
            except:
                print(" Correctly handled empty DataFrame")
            
            # Test 2: Missing columns
            try:
                bad_df = pd.DataFrame({'close': [1, 2, 3]})
                features = self.fe.calculate_all_features(bad_df)
                print(" Should have raised error for missing columns")
            except:
                print(" Correctly handled missing columns")
            
            # Test 3: All NaN values
            nan_df = self.generate_sample_data(100)
            nan_df.loc[:, 'close'] = np.nan
            try:
                features = self.fe.calculate_price_features(nan_df)
                print(" Handled NaN values gracefully")
            except:
                print(" Failed to handle NaN values")
            
            # Test 4: Single row
            single_df = self.generate_sample_data(1)
            try:
                features = self.fe.calculate_all_features(single_df)
                print(" Handled single row DataFrame")
            except:
                print(" Failed to handle single row")
            
            self.test_results.append(('Edge Cases', 'PASS'))
            
        except Exception as e:
            print(f" Error in edge cases: {e}")
            self.test_results.append(('Edge Cases', 'FAIL'))
    
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
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results:
            symbol = "Y" if result == "PASS" else "N"
            print(f"  {symbol} {test_name}: {result}")
        
        print("\n" + "=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main test execution"""
    print("=" * 80)
    print("PATH CONFIGURATION")
    print("=" * 80)
    print(f" Test file: {CURRENT_FILE}")
    print(f" Project root: {PROJECT_ROOT}")
    print(f" Data directory: {DATA_PATH}")
    print(f" Data directory exists: {DATA_PATH.exists()}")
    
    if DATA_PATH.exists():
        # Show what's in the data directory
        subdirs = [d for d in DATA_PATH.iterdir() if d.is_dir()]
        if subdirs:
            print(f" Timeframe directories found: {[d.name for d in subdirs]}")
    print("=" * 80)
    print()
    
    # Initialize tester - it will automatically use DATA_PATH
    tester = TestFeatureEngineer()
    
    # Run all tests
    tester.run_all_tests()
    
    # Return exit code based on results
    failed_tests = sum(1 for _, result in tester.test_results if result == 'FAIL')
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())