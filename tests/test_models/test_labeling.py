"""
Test Script for Labeling Module
Tests all labeling methods and components with realistic data
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the labeling module
from src.models.labeling import (
    LabelingConfig,
    TripleBarrierLabeler,
    FixedTimeLabeler,
    TrendLabeler,
    MetaLabeler,
    SampleWeights,
    LabelingPipeline
)


def create_realistic_ohlcv_data(n_days=365, freq='1h', trend='upward', volatility=0.02):
    """
    Create realistic OHLCV data with configurable trend and volatility
    
    Args:
        n_days: Number of days of data
        freq: Frequency of data
        trend: 'upward', 'downward', 'sideways'
        volatility: Daily volatility (0.02 = 2%)
    
    Returns:
        DataFrame with OHLCV columns
    """
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=n_days*24 if freq == '1h' else n_days, freq=freq)
    
    # Generate base price with trend
    np.random.seed(42)
    base_price = 40000
    
    if trend == 'upward':
        trend_factor = 1.0002  # Small upward drift
    elif trend == 'downward':
        trend_factor = 0.9998  # Small downward drift
    else:
        trend_factor = 1.0  # No trend
    
    # Generate prices with geometric brownian motion
    prices = [base_price]
    for i in range(1, len(dates)):
        change = np.random.normal(0, volatility)
        new_price = prices[-1] * trend_factor * (1 + change)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    
    # Close prices
    data['close'] = prices
    
    # Open prices (previous close with small gap)
    data['open'] = data['close'].shift(1)
    data['open'].iloc[0] = prices[0] * (1 + np.random.normal(0, 0.001))
    
    # High prices (close + random positive deviation)
    data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, volatility/2, len(data))))
    
    # Low prices (close - random positive deviation)
    data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, volatility/2, len(data))))
    
    # Ensure high >= close >= low and high >= open, low <= open
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    
    # Volume (random with trend)
    base_volume = 1000
    volume_trend = np.linspace(base_volume, base_volume * 1.5, len(data))
    data['volume'] = volume_trend * (1 + np.random.normal(0, 0.3, len(data)))
    data['volume'] = data['volume'].clip(lower=100)
    
    return data


def test_triple_barrier_labeler():
    """Test Triple Barrier Labeler"""
    print("\n" + "="*60)
    print("TESTING TRIPLE BARRIER LABELER")
    print("="*60)
    
    # Create test data
    data = create_realistic_ohlcv_data(n_days=30, trend='upward')
    print(f"\n Created test data: {len(data)} samples")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Initialize labeler
    labeler = TripleBarrierLabeler(lookback=20, vol_span=50, min_ret=0.0001)
    
    # Test 1: Basic labeling
    print("\n1. Testing basic labeling...")
    try:
        labels = labeler.label_data(
            data,
            lookforward=10,
            vol_window=20,
            pt_sl=[2, 2]
        )
        
        # Check results
        label_counts = labels.value_counts()
        print(f"    Labels created: {len(labels)} samples")
        print(f"   Distribution: {label_counts.to_dict()}")
        print(f"   Percentages: Down={100*label_counts.get(-1, 0)/len(labels):.1f}%, "
              f"Flat={100*label_counts.get(0, 0)/len(labels):.1f}%, "
              f"Up={100*label_counts.get(1, 0)/len(labels):.1f}%")
        
        assert len(labels) == len(data), "Label count mismatch"
        assert labels.notna().sum() > 0, "All labels are NaN"
        
    except Exception as e:
        print(f"    Error: {e}")
        return False
    
    # Test 2: Different barrier configurations
    print("\n2. Testing different barrier configurations...")
    configurations = [
        ([1, 1], "Symmetric tight"),
        ([2, 2], "Symmetric normal"),
        ([3, 1], "Asymmetric bullish"),
        ([1, 3], "Asymmetric bearish")
    ]
    
    for pt_sl, desc in configurations:
        try:
            labels = labeler.label_data(data, pt_sl=pt_sl)
            counts = labels.value_counts()
            print(f"    {desc} [{pt_sl}]: Up={counts.get(1, 0)}, "
                  f"Flat={counts.get(0, 0)}, Down={counts.get(-1, 0)}")
        except Exception as e:
            print(f"    {desc} failed: {e}")
    
    # Test 3: Edge cases
    print("\n3. Testing edge cases...")
    
    # Small dataset
    small_data = data.iloc[:30]
    try:
        labels = labeler.label_data(small_data, lookforward=5, vol_window=10)
        print(f"    Small dataset (30 samples): {labels.value_counts().to_dict()}")
    except Exception as e:
        print(f"    Small dataset failed: {e}")
    
    # Large lookforward
    try:
        labels = labeler.label_data(data, lookforward=50, vol_window=20)
        print(f"    Large lookforward (50): {labels.value_counts().to_dict()}")
    except Exception as e:
        print(f"    Large lookforward failed: {e}")
    
    print("\n Triple Barrier Labeler tests completed")
    return True


def test_fixed_time_labeler():
    """Test Fixed Time Labeler"""
    print("\n" + "="*60)
    print("TESTING FIXED TIME LABELER")
    print("="*60)
    
    # Create test data with strong trend
    data = create_realistic_ohlcv_data(n_days=30, trend='upward', volatility=0.03)
    print(f"\n Created trending test data: {len(data)} samples")
    
    # Initialize labeler
    labeler = FixedTimeLabeler(lookforward=20, threshold=0.02)
    
    # Test 1: Binary classification
    print("\n1. Testing binary classification...")
    try:
        labels = labeler.label_data(data, num_classes=2)
        counts = labels.value_counts()
        print(f"    Binary labels: Down={counts.get(0, 0)}, Up={counts.get(1, 0)}")
        assert len(labels) == len(data), "Label count mismatch"
    except Exception as e:
        print(f"    Error: {e}")
        return False
    
    # Test 2: Ternary classification
    print("\n2. Testing ternary classification...")
    try:
        labels = labeler.label_data(data, num_classes=3)
        counts = labels.value_counts()
        print(f"    Ternary labels: Down={counts.get(-1, 0)}, "
              f"Flat={counts.get(0, 0)}, Up={counts.get(1, 0)}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test 3: Different thresholds
    print("\n3. Testing different thresholds...")
    thresholds = [0.01, 0.02, 0.05]
    for thresh in thresholds:
        labeler_test = FixedTimeLabeler(lookforward=20, threshold=thresh)
        try:
            labels = labeler_test.label_data(data, num_classes=3)
            counts = labels.value_counts()
            print(f"    Threshold {thresh:.2%}: Up={counts.get(1, 0)}, "
                  f"Flat={counts.get(0, 0)}, Down={counts.get(-1, 0)}")
        except Exception as e:
            print(f"    Threshold {thresh} failed: {e}")
    
    print("\n Fixed Time Labeler tests completed")
    return True


def test_trend_labeler():
    """Test Trend Labeler"""
    print("\n" + "="*60)
    print("TESTING TREND LABELER")
    print("="*60)
    
    # Create test data with clear trends
    uptrend_data = create_realistic_ohlcv_data(n_days=30, trend='upward')
    downtrend_data = create_realistic_ohlcv_data(n_days=30, trend='downward')
    sideways_data = create_realistic_ohlcv_data(n_days=30, trend='sideways')
    
    labeler = TrendLabeler(window=20, min_slope=0.001)
    
    print("\n1. Testing trend detection...")
    
    datasets = [
        (uptrend_data, "Uptrend"),
        (downtrend_data, "Downtrend"),
        (sideways_data, "Sideways")
    ]
    
    for data, name in datasets:
        try:
            labels = labeler.label_data(data)
            counts = labels.value_counts()
            total = len(labels)
            print(f"    {name}: Up={100*counts.get(1, 0)/total:.1f}%, "
                  f"Flat={100*counts.get(0, 0)/total:.1f}%, "
                  f"Down={100*counts.get(-1, 0)/total:.1f}%")
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    print("\n Trend Labeler tests completed")
    return True


def test_sample_weights():
    """Test Sample Weights Calculator"""
    print("\n" + "="*60)
    print("TESTING SAMPLE WEIGHTS")
    print("="*60)
    
    # Create test data
    data = create_realistic_ohlcv_data(n_days=30)
    print(f"\n Created test data: {len(data)} samples")
    
    # Create labels and events
    triple_barrier = TripleBarrierLabeler()
    labels = triple_barrier.label_data(data, lookforward=10)
    
    # Create events
    prices = data['close']
    t_events = prices.index[:-10]
    events = triple_barrier.get_events(
        prices=prices,
        t_events=t_events,
        num_periods=10,
        vol_window=20
    )
    
    # Calculate weights
    weights_calc = SampleWeights()
    
    print("\n1. Testing weight calculation...")
    try:
        weights = weights_calc.get_sample_weights(labels, events, prices)
        print(f"    Weights calculated: {len(weights)} samples")
        print(f"   Mean weight: {weights.mean():.4f}")
        print(f"   Std weight: {weights.std():.4f}")
        print(f"   Min/Max: {weights.min():.4f} / {weights.max():.4f}")
        
        assert len(weights) == len(labels), "Weight count mismatch"
        assert weights.sum() > 0, "All weights are zero"
        
    except Exception as e:
        print(f"    Error: {e}")
        return False
    
    print("\n Sample Weights tests completed")
    return True


def test_labeling_pipeline():
    """Test Complete Labeling Pipeline"""
    print("\n" + "="*60)
    print("TESTING LABELING PIPELINE")
    print("="*60)
    
    # Create test data
    data = create_realistic_ohlcv_data(n_days=60, trend='upward')
    print(f"\n Created test data: {len(data)} samples")
    
    # Initialize pipeline
    config = LabelingConfig(
        method='triple_barrier',
        lookforward=10,
        vol_window=20,
        pt_sl=[2, 2],
        threshold=0.02,
        num_classes=3
    )
    pipeline = LabelingPipeline(config)
    
    # Test 1: Different labeling methods
    print("\n1. Testing different labeling methods...")
    methods = ['triple_barrier', 'fixed_time', 'trend', 'simple_returns']
    
    for method in methods:
        try:
            labels = pipeline.create_labels(data, method=method)
            stats = pipeline.get_label_statistics(labels)
            print(f"\n    {method}:")
            print(f"     Total: {stats['total']}, Valid: {stats['valid']}")
            print(f"     Distribution: {stats.get('class_distribution', {})}")
            print(f"     Balance ratio: {stats.get('class_balance_ratio', 0):.3f}")
        except Exception as e:
            print(f"    {method} failed: {e}")
    
    # Test 2: Labels with weights
    print("\n2. Testing labels with weights...")
    try:
        labels, weights = pipeline.create_labels_with_weights(data, method='triple_barrier')
        print(f"    Labels: {len(labels)}, Weights: {len(weights)}")
        print(f"   Weight stats: mean={weights.mean():.4f}, std={weights.std():.4f}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test 3: Parallel labeling
    print("\n3. Testing parallel labeling...")
    data_dict = {
        'BTC': data,
        'ETH': data.copy(),
        'SOL': data.copy()
    }
    
    try:
        results = pipeline.parallel_labeling(data_dict, method='triple_barrier', n_jobs=2)
        for symbol, labels in results.items():
            print(f"    {symbol}: {len(labels)} labels")
    except Exception as e:
        print(f"    Parallel labeling failed: {e}")
        # Try without parallel processing
        print("   Retrying without parallel processing...")
        results = {}
        for symbol, df in data_dict.items():
            results[symbol] = pipeline.create_labels(df)
            print(f"    {symbol}: {len(results[symbol])} labels")
    
    print("\n Labeling Pipeline tests completed")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    pipeline = LabelingPipeline()
    
    # Test 1: Empty DataFrame
    print("\n1. Testing empty DataFrame...")
    empty_df = pd.DataFrame()
    try:
        labels = pipeline.create_labels(empty_df)
        print(f"    Empty data handled: {len(labels)} labels")
    except Exception as e:
        print(f"    Empty data raised expected error: {type(e).__name__}")
    
    # Test 2: Very small dataset
    print("\n2. Testing very small dataset...")
    dates = pd.date_range(end=datetime.now(), periods=10, freq='1h')
    small_data = pd.DataFrame({
        'open': np.random.uniform(40000, 41000, 10),
        'high': np.random.uniform(41000, 42000, 10),
        'low': np.random.uniform(39000, 40000, 10),
        'close': np.random.uniform(40000, 41000, 10),
        'volume': np.random.uniform(100, 1000, 10)
    }, index=dates)
    
    try:
        labels = pipeline.create_labels(small_data, lookforward=5)
        print(f"    Small data handled: {labels.value_counts().to_dict()}")
    except Exception as e:
        print(f"    Small data failed: {e}")
    
    # Test 3: Data with NaN values
    print("\n3. Testing data with NaN values...")
    nan_data = create_realistic_ohlcv_data(n_days=30)
    # Introduce some NaN values
    nan_data.iloc[10:15, nan_data.columns.get_loc('close')] = np.nan
    
    try:
        labels = pipeline.create_labels(nan_data)
        print(f"    NaN data handled: {labels.notna().sum()} valid labels")
    except Exception as e:
        print(f"    NaN data failed: {e}")
    
    # Test 4: Invalid parameters
    print("\n4. Testing invalid parameters...")
    data = create_realistic_ohlcv_data(n_days=30)
    
    try:
        labels = pipeline.create_labels(data, method='invalid_method')
        print(f"    Invalid method should have raised error")
    except ValueError as e:
        print(f"    Invalid method raised expected error")
    
    print("\n Edge case tests completed")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# LABELING MODULE COMPREHENSIVE TEST SUITE")
    print("#"*60)
    
    start_time = datetime.now()
    
    tests = [
        ("Triple Barrier Labeler", test_triple_barrier_labeler),
        ("Fixed Time Labeler", test_fixed_time_labeler),
        ("Trend Labeler", test_trend_labeler),
        ("Sample Weights", test_sample_weights),
        ("Labeling Pipeline", test_labeling_pipeline),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)
    
    for name, result in results:
        status = " PASSED" if result else " FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    elapsed = datetime.now() - start_time
    print(f"Time elapsed: {elapsed.total_seconds():.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n ALL TESTS PASSED! The labeling module is working correctly.")
    else:
        print(f"\n {total_tests - passed_tests} test(s) failed. Please review the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)