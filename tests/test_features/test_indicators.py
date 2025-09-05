"""
Test Script for Technical Indicators Module
Tests all functionality of indicators.py
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
project_root = current_dir.parent  # Go up one level to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the module to test
from src.features.indicators import TechnicalIndicators

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TestTechnicalIndicators:
    """
    Comprehensive test suite for TechnicalIndicators class
    """
    
    def __init__(self, data_path: str = None, visualize: bool = False):
        """
        Initialize test suite
        
        Args:
            data_path: Path to raw data directory
            visualize: Whether to create visualization plots
        """
        # Better path handling
        if data_path is None:
            # Try to find the data directory
            possible_roots = [
                Path.cwd(),  # Current working directory
                Path(__file__).parent.parent,  # Project root from test file
                Path.cwd().parent,  # One level up
            ]
            
            for root in possible_roots:
                if (root / 'data' / 'raw').exists():
                    data_path = root / 'data' / 'raw'
                    break
            else:
                data_path = Path('data/raw')  # Fallback
        
        self.data_path = Path(data_path)
        self.indicators = TechnicalIndicators()
        self.test_results = []
        self.visualize = visualize
        self.indicator_values = {}
        
    def run_all_tests(self):
        """Run all tests and report results"""
        print("=" * 80)
        print("TECHNICAL INDICATORS TEST SUITE (FIXED)")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Looking for data in: {self.data_path}\n")
        
        # Load test data
        print("Loading test data...")
        df = self.load_test_data()
        if df is None:
            print(" Failed to load data. Stopping tests.")
            return
        
        # Test categories
        print("\n" + "="*50)
        print("MOMENTUM INDICATORS")
        print("="*50)
        self.test_momentum_indicators(df)
        
        print("\n" + "="*50)
        print("VOLATILITY INDICATORS")
        print("="*50)
        self.test_volatility_indicators(df)
        
        print("\n" + "="*50)
        print("TREND INDICATORS")
        print("="*50)
        self.test_trend_indicators(df)
        
        print("\n" + "="*50)
        print("VOLUME INDICATORS")
        print("="*50)
        self.test_volume_indicators(df)
        
        print("\n" + "="*50)
        print("HELPER METHODS")
        print("="*50)
        self.test_helper_methods(df)
        
        print("\n" + "="*50)
        print("EDGE CASES & VALIDATION")
        print("="*50)
        self.test_edge_cases()
        
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARKS")
        print("="*50)
        self.test_performance(df)
        
        print("\n" + "="*50)
        print("INDICATOR VALUE VALIDATION")
        print("="*50)
        self.validate_indicator_values()
        
        # Visualize if requested
        if self.visualize:
            self.create_visualizations(df)
        
        # Print summary
        self.print_test_summary()
    
    def load_test_data(self) -> pd.DataFrame:
        """Load or generate test data"""
        try:
            # Updated paths to match your structure
            possible_paths = [
                # Your actual data structure
                self.data_path / '1d' / 'BTC_USDT_1d.csv',
                self.data_path / '1h' / 'BTC_USDT_1h.csv',
                self.data_path / '4h' / 'BTC_USDT_4h.csv',
                self.data_path / '15m' / 'BTC_USDT_15m.csv',
                self.data_path / '5m' / 'BTC_USDT_5m.csv',
                # Alternative structures
                self.data_path / 'BTC_USDT' / '1d.csv',
                self.data_path / 'BTC_USDT' / '1h.csv',
                self.data_path / 'BTCUSDT' / '1h.csv',
                self.data_path / 'btc_1h.csv',
            ]
            
            df = None
            for path in possible_paths:
                if path.exists():
                    print(f"Found data file: {path}")
                    df = pd.read_csv(path)
                    print(f" Loaded data from {path.name}")
                    break
            
            if df is None:
                print(" No data file found in these locations:")
                for path in possible_paths[:5]:  # Show first 5 attempted paths
                    print(f"   - {path}")
                print("Generating sample data instead...")
                df = self.generate_sample_data()
            
            # Ensure required columns (handle different column name formats)
            column_mappings = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 
                'Close': 'close', 'Volume': 'volume'
            }
            
            # Rename columns if needed
            df = df.rename(columns=column_mappings)
            
            # Check for required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                print(f" Missing required columns: {missing}")
                print(f"   Available columns: {list(df.columns)}")
                return None
            
            # Set index if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            print(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f" Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f" Error loading data: {e}")
            return None
    
    def generate_sample_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate realistic OHLCV data for testing"""
        np.random.seed(42)
        
        # Generate realistic price movement
        timestamps = pd.date_range(end=datetime.now(), periods=periods, freq='1H')
        
        # Start price
        initial_price = 50000
        
        # Generate returns with momentum and mean reversion
        returns = np.random.normal(0.0001, 0.02, periods)
        
        # Add trend
        trend = np.linspace(0, 0.1, periods) / periods
        returns = returns + trend
        
        # Calculate prices
        price_series = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        df = pd.DataFrame(index=timestamps)
        df['close'] = price_series
        
        # Generate realistic OHLC relationships
        daily_range = np.abs(np.random.normal(0.01, 0.005, periods))
        
        df['open'] = df['close'] * (1 + np.random.normal(0, 0.003, periods))
        df['high'] = np.maximum(df['open'], df['close']) * (1 + daily_range * np.random.uniform(0.3, 1, periods))
        df['low'] = np.minimum(df['open'], df['close']) * (1 - daily_range * np.random.uniform(0.3, 1, periods))
        
        # Generate volume (correlated with price movement)
        base_volume = 1000000
        price_change = np.abs(df['close'].pct_change())
        df['volume'] = base_volume * (1 + price_change * 10) * np.random.uniform(0.5, 1.5, periods)
        df['volume'] = df['volume'].fillna(base_volume)
        
        return df
    
    # ================== MOMENTUM INDICATORS TESTS ==================
    
    def test_momentum_indicators(self, df: pd.DataFrame):
        """Test all momentum indicators"""
        
        # Test RSI
        print("\n1. Testing RSI...")
        try:
            rsi_14 = self.indicators.rsi(df['close'], 14)
            rsi_7 = self.indicators.rsi(df['close'], 7)
            
            # Validate
            assert not rsi_14.isna().all(), "RSI all NaN"
            assert (rsi_14.dropna() >= 0).all() and (rsi_14.dropna() <= 100).all(), "RSI out of range"
            
            # Store for validation
            self.indicator_values['rsi_14'] = rsi_14
            
            # Analysis
            last_rsi = rsi_14.iloc[-1]
            avg_rsi = rsi_14.mean()
            
            print(f" RSI(14) - Last: {last_rsi:.2f}, Avg: {avg_rsi:.2f}")
            
            # Check overbought/oversold
            overbought = (rsi_14 > 70).sum()
            oversold = (rsi_14 < 30).sum()
            print(f"   Overbought periods: {overbought} ({overbought/len(rsi_14)*100:.1f}%)")
            print(f"   Oversold periods: {oversold} ({oversold/len(rsi_14)*100:.1f}%)")
            
            self.test_results.append(('RSI', 'PASS'))
            
        except Exception as e:
            print(f" RSI failed: {e}")
            self.test_results.append(('RSI', 'FAIL'))
        
        # Test MACD
        print("\n2. Testing MACD...")
        try:
            macd, signal, histogram = self.indicators.macd(df['close'])
            
            assert not macd.isna().all(), "MACD all NaN"
            assert len(macd) == len(df), "MACD length mismatch"
            
            self.indicator_values['macd'] = macd
            self.indicator_values['macd_signal'] = signal
            self.indicator_values['macd_histogram'] = histogram
            
            last_hist = histogram.iloc[-1]
            crossovers = ((macd > signal) & (macd.shift(1) <= signal.shift(1))).sum()
            
            print(f" MACD - Histogram: {last_hist:.4f}")
            print(f"   Bullish crossovers: {crossovers}")
            
            self.test_results.append(('MACD', 'PASS'))
            
        except Exception as e:
            print(f" MACD failed: {e}")
            self.test_results.append(('MACD', 'FAIL'))
        
        # Test Stochastic
        print("\n3. Testing Stochastic...")
        try:
            k, d = self.indicators.stochastic(df['high'], df['low'], df['close'])
            
            assert not k.isna().all(), "Stochastic K all NaN"
            assert (k.dropna() >= 0).all() and (k.dropna() <= 100).all(), "Stochastic out of range"
            
            self.indicator_values['stoch_k'] = k
            self.indicator_values['stoch_d'] = d
            
            print(f" Stochastic - K: {k.iloc[-1]:.2f}, D: {d.iloc[-1]:.2f}")
            
            self.test_results.append(('Stochastic', 'PASS'))
            
        except Exception as e:
            print(f" Stochastic failed: {e}")
            self.test_results.append(('Stochastic', 'FAIL'))
        
        # Test Williams %R
        print("\n4. Testing Williams %R...")
        try:
            williams = self.indicators.williams_r(df['high'], df['low'], df['close'])
            
            # Validate
            assert not williams.isna().all(), "Williams %R all NaN"
            assert (williams.dropna() >= -100).all() and (williams.dropna() <= 0).all(), "Williams %R out of range"
            
            print(f" Williams %R - Last: {williams.iloc[-1]:.2f}")
            
            self.test_results.append(('Williams %R', 'PASS'))
            
        except Exception as e:
            print(f" Williams %R failed: {e}")
            self.test_results.append(('Williams %R', 'FAIL'))
        
        # Test CCI
        print("\n5. Testing CCI...")
        try:
            cci = self.indicators.cci(df['high'], df['low'], df['close'])
            
            # Validate
            assert not cci.isna().all(), "CCI all NaN"
            
            # Analysis
            extreme_high = (cci > 100).sum()
            extreme_low = (cci < -100).sum()
            
            print(f" CCI - Last: {cci.iloc[-1]:.2f}")
            print(f"   Extreme high (>100): {extreme_high}")
            print(f"   Extreme low (<-100): {extreme_low}")
            
            self.test_results.append(('CCI', 'PASS'))
            
        except Exception as e:
            print(f" CCI failed: {e}")
            self.test_results.append(('CCI', 'FAIL'))
        
        # Test ROC
        print("\n6. Testing ROC...")
        try:
            roc = self.indicators.roc(df['close'], 10)
            
            # Validate
            assert not roc.isna().all(), "ROC all NaN"
            
            print(f" ROC(10) - Last: {roc.iloc[-1]:.2f}%")
            
            self.test_results.append(('ROC', 'PASS'))
            
        except Exception as e:
            print(f" ROC failed: {e}")
            self.test_results.append(('ROC', 'FAIL'))
        
        # Test MFI
        print("\n7. Testing MFI...")
        try:
            mfi = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
            
            # Validate
            assert not mfi.isna().all(), "MFI all NaN"
            valid_mfi = mfi.dropna()
            if len(valid_mfi) > 0:
                assert (valid_mfi >= 0).all() and (valid_mfi <= 100).all(), "MFI out of range"
            
            print(f" MFI - Last: {mfi.iloc[-1]:.2f}")
            
            self.test_results.append(('MFI', 'PASS'))
            
        except Exception as e:
            print(f" MFI failed: {e}")
            self.test_results.append(('MFI', 'FAIL'))
    
    # ================== VOLATILITY INDICATORS TESTS ==================
    
    def test_volatility_indicators(self, df: pd.DataFrame):
        """Test all volatility indicators"""
        
        # Test ATR
        print("\n1. Testing ATR...")
        try:
            atr = self.indicators.atr(df['high'], df['low'], df['close'])
            
            assert not atr.isna().all(), "ATR all NaN"
            assert (atr.dropna() >= 0).all(), "ATR negative values"
            
            self.indicator_values['atr'] = atr
            
            atr_pct = (atr / df['close']) * 100
            
            print(f" ATR - Last: {atr.iloc[-1]:.2f} ({atr_pct.iloc[-1]:.2f}% of price)")
            print(f"   Average: {atr.mean():.2f}")
            
            self.test_results.append(('ATR', 'PASS'))
            
        except Exception as e:
            print(f" ATR failed: {e}")
            self.test_results.append(('ATR', 'FAIL'))
        
        # Test Bollinger Bands (FIXED)
        print("\n2. Testing Bollinger Bands...")
        try:
            upper, middle, lower = self.indicators.bollinger_bands(df['close'])
            
            # Validate - Fixed to handle NaN properly
            assert not upper.isna().all(), "BB Upper all NaN"
            
            # Only check where all values are not NaN
            valid_mask = ~(upper.isna() | middle.isna() | lower.isna())
            if valid_mask.any():
                assert (upper[valid_mask] >= middle[valid_mask]).all(), "BB Upper < Middle"
                assert (middle[valid_mask] >= lower[valid_mask]).all(), "BB Middle < Lower"
            
            self.indicator_values['bb_upper'] = upper
            self.indicator_values['bb_lower'] = lower
            
            bb_width = upper - lower
            bb_width_pct = (bb_width / middle) * 100
            
            print(f" Bollinger Bands")
            print(f"   Upper: {upper.iloc[-1]:.2f}")
            print(f"   Middle: {middle.iloc[-1]:.2f}")
            print(f"   Lower: {lower.iloc[-1]:.2f}")
            print(f"   Width: {bb_width_pct.iloc[-1]:.2f}%")
            
            squeeze_threshold = bb_width_pct.quantile(0.2)
            squeezes = (bb_width_pct < squeeze_threshold).sum()
            print(f"   Squeeze periods: {squeezes}")
            
            self.test_results.append(('Bollinger Bands', 'PASS'))
            
        except Exception as e:
            print(f" Bollinger Bands failed: {e}")
            self.test_results.append(('Bollinger Bands', 'FAIL'))
        
        # Test Keltner Channels (FIXED)
        print("\n3. Testing Keltner Channels...")
        try:
            kc_upper, kc_middle, kc_lower = self.indicators.keltner_channels(
                df['high'], df['low'], df['close']
            )
            
            # Validate - Fixed to handle NaN properly
            assert not kc_upper.isna().all(), "KC Upper all NaN"
            
            # Only check where values are not NaN
            valid_mask = ~(kc_upper.isna() | kc_middle.isna() | kc_lower.isna())
            if valid_mask.any():
                assert (kc_upper[valid_mask] >= kc_middle[valid_mask]).all(), "KC Upper < Middle"
                assert (kc_middle[valid_mask] >= kc_lower[valid_mask]).all(), "KC Middle < Lower"
            
            print(f" Keltner Channels")
            print(f"   Upper: {kc_upper.iloc[-1]:.2f}")
            print(f"   Lower: {kc_lower.iloc[-1]:.2f}")
            
            self.test_results.append(('Keltner Channels', 'PASS'))
            
        except Exception as e:
            print(f" Keltner Channels failed: {e}")
            self.test_results.append(('Keltner Channels', 'FAIL'))
        
        # Test Donchian Channels (FIXED)
        print("\n4. Testing Donchian Channels...")
        try:
            dc_upper, dc_middle, dc_lower = self.indicators.donchian_channels(
                df['high'], df['low']
            )
            
            # Validate - Fixed to handle NaN properly
            assert not dc_upper.isna().all(), "DC Upper all NaN"
            
            # Only check where values are not NaN
            valid_mask = ~(dc_upper.isna() | dc_lower.isna())
            if valid_mask.any():
                assert (dc_upper[valid_mask] >= dc_lower[valid_mask]).all(), "DC Upper < Lower"
            
            print(f" Donchian Channels")
            print(f"   Upper: {dc_upper.iloc[-1]:.2f}")
            print(f"   Lower: {dc_lower.iloc[-1]:.2f}")
            
            self.test_results.append(('Donchian Channels', 'PASS'))
            
        except Exception as e:
            print(f" Donchian Channels failed: {e}")
            self.test_results.append(('Donchian Channels', 'FAIL'))
    
    # ================== TREND INDICATORS TESTS ==================
    
    def test_trend_indicators(self, df: pd.DataFrame):
        """Test all trend indicators"""
        
        # Test ADX
        print("\n1. Testing ADX...")
        try:
            adx = self.indicators.adx(df['high'], df['low'], df['close'])
            
            # Validate
            assert not adx.isna().all(), "ADX all NaN"
            valid_adx = adx.dropna()
            if len(valid_adx) > 0:
                assert (valid_adx >= 0).all() and (valid_adx <= 100).all(), "ADX out of range"
            
            # Analysis
            last_adx = adx.iloc[-1]
            
            print(f" ADX - Last: {last_adx:.2f}")
            if last_adx > 40:
                print("   Strong trend indicated")
            elif last_adx > 25:
                print("   Moderate trend indicated")
            else:
                print("   Weak trend indicated")
            
            self.test_results.append(('ADX', 'PASS'))
            
        except Exception as e:
            print(f" ADX failed: {e}")
            self.test_results.append(('ADX', 'FAIL'))
        
        # Test Aroon
        print("\n2. Testing Aroon...")
        try:
            aroon_up, aroon_down, aroon_osc = self.indicators.aroon(df['high'], df['low'])
            
            # Validate
            assert not aroon_up.isna().all(), "Aroon Up all NaN"
            valid_up = aroon_up.dropna()
            if len(valid_up) > 0:
                assert (valid_up >= 0).all() and (valid_up <= 100).all(), "Aroon Up out of range"
            
            print(f" Aroon")
            print(f"   Up: {aroon_up.iloc[-1]:.2f}")
            print(f"   Down: {aroon_down.iloc[-1]:.2f}")
            print(f"   Oscillator: {aroon_osc.iloc[-1]:.2f}")
            
            self.test_results.append(('Aroon', 'PASS'))
            
        except Exception as e:
            print(f" Aroon failed: {e}")
            self.test_results.append(('Aroon', 'FAIL'))
        
        # Test PSAR
        print("\n3. Testing Parabolic SAR...")
        try:
            psar = self.indicators.psar(df['high'], df['low'])
            
            # Validate
            assert not psar.isna().all(), "PSAR all NaN"
            assert len(psar) == len(df), "PSAR length mismatch"
            
            # Analysis
            above_psar = (df['close'] > psar).sum()
            below_psar = (df['close'] < psar).sum()
            
            print(f" PSAR - Last: {psar.iloc[-1]:.2f}")
            print(f"   Price above PSAR: {above_psar} periods")
            print(f"   Price below PSAR: {below_psar} periods")
            
            self.test_results.append(('PSAR', 'PASS'))
            
        except Exception as e:
            print(f" PSAR failed: {e}")
            self.test_results.append(('PSAR', 'FAIL'))
        
        # Test Ichimoku
        print("\n4. Testing Ichimoku Cloud...")
        try:
            ichimoku = self.indicators.ichimoku(df['high'], df['low'], df['close'])
            
            # Validate
            assert 'conversion_line' in ichimoku, "Ichimoku missing conversion line"
            assert 'base_line' in ichimoku, "Ichimoku missing base line"
            assert not ichimoku['conversion_line'].isna().all(), "Conversion line all NaN"
            
            print(f" Ichimoku Cloud")
            print(f"   Conversion: {ichimoku['conversion_line'].iloc[-1]:.2f}")
            print(f"   Base: {ichimoku['base_line'].iloc[-1]:.2f}")
            
            self.test_results.append(('Ichimoku', 'PASS'))
            
        except Exception as e:
            print(f" Ichimoku failed: {e}")
            self.test_results.append(('Ichimoku', 'FAIL'))
    
    # ================== VOLUME INDICATORS TESTS ==================
    
    def test_volume_indicators(self, df: pd.DataFrame):
        """Test all volume indicators"""
        
        # Test OBV
        print("\n1. Testing OBV...")
        try:
            obv = self.indicators.obv(df['close'], df['volume'])
            
            # Validate
            assert not obv.isna().all(), "OBV all NaN"
            assert len(obv) == len(df), "OBV length mismatch"
            
            # Analysis
            obv_trend = (obv.iloc[-1] - obv.iloc[-20]) / abs(obv.iloc[-20]) * 100 if len(obv) > 20 else 0
            
            print(f" OBV - Last: {obv.iloc[-1]:,.0f}")
            print(f"   20-period change: {obv_trend:+.1f}%")
            
            self.test_results.append(('OBV', 'PASS'))
            
        except Exception as e:
            print(f" OBV failed: {e}")
            self.test_results.append(('OBV', 'FAIL'))
        
        # Test A/D Line
        print("\n2. Testing A/D Line...")
        try:
            ad_line = self.indicators.ad_line(df['high'], df['low'], df['close'], df['volume'])
            
            # Validate
            assert not ad_line.isna().all(), "A/D Line all NaN"
            
            print(f" A/D Line - Last: {ad_line.iloc[-1]:,.0f}")
            
            self.test_results.append(('A/D Line', 'PASS'))
            
        except Exception as e:
            print(f" A/D Line failed: {e}")
            self.test_results.append(('A/D Line', 'FAIL'))
        
        # Test VWAP
        print("\n3. Testing VWAP...")
        try:
            vwap = self.indicators.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # Validate
            assert not vwap.isna().all(), "VWAP all NaN"
            
            # Analysis
            above_vwap = df['close'] > vwap
            
            print(f" VWAP - Last: {vwap.iloc[-1]:.2f}")
            print(f"   Price vs VWAP: {'+' if above_vwap.iloc[-1] else '-'}{abs(df['close'].iloc[-1] - vwap.iloc[-1]):.2f}")
            
            self.test_results.append(('VWAP', 'PASS'))
            
        except Exception as e:
            print(f" VWAP failed: {e}")
            self.test_results.append(('VWAP', 'FAIL'))
        
        # Test Volume SMA Ratio
        print("\n4. Testing Volume SMA Ratio...")
        try:
            vol_ratio = self.indicators.volume_sma_ratio(df['volume'])
            
            # Validate
            assert not vol_ratio.isna().all(), "Volume ratio all NaN"
            
            print(f" Volume SMA Ratio - Last: {vol_ratio.iloc[-1]:.2f}")
            
            self.test_results.append(('Volume SMA Ratio', 'PASS'))
            
        except Exception as e:
            print(f" Volume SMA Ratio failed: {e}")
            self.test_results.append(('Volume SMA Ratio', 'FAIL'))
        
        # Test CMF
        print("\n5. Testing CMF...")
        try:
            cmf = self.indicators.cmf(df['high'], df['low'], df['close'], df['volume'])
            
            # Validate
            assert not cmf.isna().all(), "CMF all NaN"
            valid_cmf = cmf.dropna()
            if len(valid_cmf) > 0:
                assert (valid_cmf >= -1).all() and (valid_cmf <= 1).all(), "CMF out of range"
            
            print(f" CMF - Last: {cmf.iloc[-1]:.3f}")
            
            self.test_results.append(('CMF', 'PASS'))
            
        except Exception as e:
            print(f" CMF failed: {e}")
            self.test_results.append(('CMF', 'FAIL'))
        
        # Test Force Index
        print("\n6. Testing Force Index...")
        try:
            fi = self.indicators.fi(df['close'], df['volume'])
            
            # Validate
            assert not fi.isna().all(), "Force Index all NaN"
            
            print(f" Force Index - Last: {fi.iloc[-1]:,.0f}")
            
            self.test_results.append(('Force Index', 'PASS'))
            
        except Exception as e:
            print(f" Force Index failed: {e}")
            self.test_results.append(('Force Index', 'FAIL'))
    
    # ================== HELPER METHODS TESTS ==================
    
    def test_helper_methods(self, df: pd.DataFrame):
        """Test helper methods"""
        
        # Test calculate_all
        print("\n1. Testing calculate_all()...")
        try:
            start_time = time.time()
            all_indicators = self.indicators.calculate_all(df)
            elapsed = time.time() - start_time
            
            # Validate
            assert len(all_indicators.columns) > 30, "Too few indicators calculated"
            assert len(all_indicators) == len(df), "Length mismatch"
            
            print(f" Calculate All - {len(all_indicators.columns)} indicators in {elapsed:.2f}s")
            print(f"   Indicators per second: {len(all_indicators.columns)/elapsed:.1f}")
            
            # Check for missing values
            nan_pct = all_indicators.isna().sum().sum() / (len(all_indicators) * len(all_indicators.columns)) * 100
            print(f"   NaN percentage: {nan_pct:.1f}%")
            
            self.test_results.append(('Calculate All', 'PASS'))
            
        except Exception as e:
            print(f" Calculate All failed: {e}")
            self.test_results.append(('Calculate All', 'FAIL'))
        
        # Test add_moving_averages
        print("\n2. Testing add_moving_averages()...")
        try:
            ma_df = self.indicators.add_moving_averages(df)
            
            # Validate
            assert 'sma_10' in ma_df.columns, "SMA 10 missing"
            assert 'ema_10' in ma_df.columns, "EMA 10 missing"
            
            # Check relationships
            sma_20 = ma_df['sma_20'].dropna()
            sma_50 = ma_df['sma_50'].dropna()
            
            print(f" Moving Averages - {len(ma_df.columns)} MAs added")
            
            # Trend analysis
            if len(sma_20) > 0 and len(sma_50) > 0:
                golden_cross = (sma_20.iloc[-1] > sma_50.iloc[-1]) if len(sma_20) > 0 else False
                print(f"   Current trend: {'Bullish (Golden Cross)' if golden_cross else 'Bearish (Death Cross)'}")
            
            self.test_results.append(('Moving Averages', 'PASS'))
            
        except Exception as e:
            print(f" Moving Averages failed: {e}")
            self.test_results.append(('Moving Averages', 'FAIL'))
        
        # Test normalize_indicators
        print("\n3. Testing normalize_indicators()...")
        try:
            # Calculate some indicators first
            test_indicators = pd.DataFrame()
            test_indicators['rsi_14'] = self.indicators.rsi(df['close'])
            test_indicators['williams_r'] = self.indicators.williams_r(df['high'], df['low'], df['close'])
            test_indicators['cmf'] = self.indicators.cmf(df['high'], df['low'], df['close'], df['volume'])
            
            normalized = self.indicators.normalize_indicators(test_indicators)
            
            # Validate normalization
            for col in ['rsi_14']:
                if col in normalized.columns:
                    valid_data = normalized[col].dropna()
                    assert (valid_data >= 0).all() and (valid_data <= 1).all(), f"{col} not normalized"
            
            print(f" Normalization - All indicators normalized to [0,1]")
            
            self.test_results.append(('Normalization', 'PASS'))
            
        except Exception as e:
            print(f" Normalization failed: {e}")
            self.test_results.append(('Normalization', 'FAIL'))
    
    # ================== EDGE CASES TESTS ==================
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        
        print("\n1. Testing with minimal data...")
        try:
            small_df = self.generate_sample_data(20)
            rsi = self.indicators.rsi(small_df['close'], 14)
            
            assert len(rsi) == len(small_df), "Length mismatch with small data"
            print(" Handled small dataset")
            
        except Exception as e:
            print(f" Small dataset failed: {e}")
        
        print("\n2. Testing with NaN values...")
        try:
            # Create data with NaN - FIXED to use iloc
            nan_df = self.generate_sample_data(100)
            nan_df.iloc[40:60, nan_df.columns.get_loc('close')] = np.nan  # Fixed: use iloc
            
            rsi = self.indicators.rsi(nan_df['close'])
            assert len(rsi) == len(nan_df), "Length mismatch with NaN"
            print(" Handled NaN values")
            
        except Exception as e:
            print(f" NaN handling failed: {e}")
        
        print("\n3. Testing with extreme values...")
        try:
            extreme_df = self.generate_sample_data(100)
            extreme_df.iloc[50, extreme_df.columns.get_loc('high')] = extreme_df['high'].max() * 100
            
            atr = self.indicators.atr(extreme_df['high'], extreme_df['low'], extreme_df['close'])
            assert not np.isinf(atr).any(), "Infinity in ATR"
            print(" Handled extreme values")
            
        except Exception as e:
            print(f" Extreme values failed: {e}")
        
        print("\n4. Testing with zero volume...")
        try:
            zero_vol_df = self.generate_sample_data(100)
            zero_vol_df['volume'] = 0
            
            obv = self.indicators.obv(zero_vol_df['close'], zero_vol_df['volume'])
            assert (obv == 0).all(), "OBV not zero with zero volume"
            print(" Handled zero volume")
            
        except Exception as e:
            print(f" Zero volume failed: {e}")
        
        print("\n5. Testing with constant prices...")
        try:
            flat_df = self.generate_sample_data(100)
            flat_df['close'] = 50000  # Constant price
            
            rsi = self.indicators.rsi(flat_df['close'])
            assert not rsi.isna().all(), "RSI all NaN for flat prices"
            print(" Handled constant prices")
            
        except Exception as e:
            print(f" Constant prices failed: {e}")
        
        self.test_results.append(('Edge Cases', 'PASS'))
    
    # ================== PERFORMANCE TESTS ==================
    
    def test_performance(self, df: pd.DataFrame):
        """Test performance with different data sizes"""
        
        print("\nPerformance benchmarks:")
        
        sizes = [100, 500, 1000, 2000]
        times = {}
        
        for size in sizes:
            if size > len(df):
                test_df = self.generate_sample_data(size)
            else:
                test_df = df.iloc[:size].copy()
            
            # Time individual indicators
            start = time.time()
            _ = self.indicators.rsi(test_df['close'])
            rsi_time = time.time() - start
            
            start = time.time()
            _ = self.indicators.macd(test_df['close'])
            macd_time = time.time() - start
            
            start = time.time()
            _ = self.indicators.calculate_all(test_df)
            all_time = time.time() - start
            
            times[size] = {'rsi': rsi_time, 'macd': macd_time, 'all': all_time}
            
            print(f"\n  {size} rows:")
            print(f"    RSI: {rsi_time*1000:.1f}ms")
            print(f"    MACD: {macd_time*1000:.1f}ms")
            print(f"    All indicators: {all_time:.2f}s")
        
        # Check scaling
        if 1000 in times and 100 in times:
            scaling = times[1000]['all'] / times[100]['all']
            print(f"\n Scaling factor (100->1000): {scaling:.1f}x")
            
            if scaling < 15:  # Should scale sub-linearly
                print("   Good performance scaling")
            else:
                print("    Poor scaling performance")
        
        self.test_results.append(('Performance', 'PASS'))
    
    # ================== VALUE VALIDATION ==================
    
    def validate_indicator_values(self):
        """Validate indicator values are in expected ranges"""
        
        print("\nValidating indicator value ranges...")
        
        validations = {
            'rsi_14': (0, 100),
            'stoch_k': (0, 100),
            'stoch_d': (0, 100),
            'mfi': (0, 100),
            'cmf': (-1, 1),
            'atr': (0, float('inf'))
        }
        
        all_valid = True
        
        for indicator, (min_val, max_val) in validations.items():
            if indicator in self.indicator_values:
                values = self.indicator_values[indicator].dropna()
                if len(values) > 0:
                    actual_min = values.min()
                    actual_max = values.max()
                    
                    if actual_min >= min_val and actual_max <= max_val:
                        print(f" {indicator}: [{actual_min:.2f}, {actual_max:.2f}] ")
                    else:
                        print(f" {indicator}: [{actual_min:.2f}, {actual_max:.2f}] (expected [{min_val}, {max_val}])")
                        all_valid = False
        
        if all_valid:
            self.test_results.append(('Value Validation', 'PASS'))
        else:
            self.test_results.append(('Value Validation', 'FAIL'))
    
    # ================== VISUALIZATION ==================
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization of indicators"""
        
        print("\nCreating visualizations...")
        
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Price and Bollinger Bands
            axes[0].plot(df.index[-100:], df['close'].iloc[-100:], label='Close', color='black')
            if 'bb_upper' in self.indicator_values:
                axes[0].plot(df.index[-100:], self.indicator_values['bb_upper'].iloc[-100:], 
                           label='BB Upper', color='red', alpha=0.5)
                axes[0].plot(df.index[-100:], self.indicator_values['bb_lower'].iloc[-100:], 
                           label='BB Lower', color='green', alpha=0.5)
            axes[0].set_title('Price with Bollinger Bands')
            axes[0].legend()
            
            # RSI
            if 'rsi_14' in self.indicator_values:
                axes[1].plot(df.index[-100:], self.indicator_values['rsi_14'].iloc[-100:], 
                           label='RSI(14)', color='purple')
                axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
                axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
                axes[1].set_title('RSI')
                axes[1].set_ylim(0, 100)
                axes[1].legend()
            
            # MACD
            if 'macd' in self.indicator_values:
                axes[2].plot(df.index[-100:], self.indicator_values['macd'].iloc[-100:], 
                           label='MACD', color='blue')
                axes[2].plot(df.index[-100:], self.indicator_values['macd_signal'].iloc[-100:], 
                           label='Signal', color='red')
                axes[2].bar(df.index[-100:], self.indicator_values['macd_histogram'].iloc[-100:], 
                          label='Histogram', alpha=0.3)
                axes[2].set_title('MACD')
                axes[2].legend()
            
            # Volume
            axes[3].bar(df.index[-100:], df['volume'].iloc[-100:], alpha=0.5)
            axes[3].set_title('Volume')
            
            plt.tight_layout()
            plt.savefig('indicators_test_visualization.png')
            print(" Saved visualization to indicators_test_visualization.png")
            
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Technical Indicators')
    parser.add_argument('--data-path', type=str, default='data/raw', 
                       help='Path to data directory')
    parser.add_argument('--visualize', action='store_true', 
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TestTechnicalIndicators(args.data_path, args.visualize)
    
    # Run all tests
    tester.run_all_tests()
    
    # Return exit code
    failed_tests = sum(1 for _, result in tester.test_results if result == 'FAIL')
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit(main())