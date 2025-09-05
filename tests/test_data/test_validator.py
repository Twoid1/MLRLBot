"""
Test suite for DataValidator module

This module contains comprehensive tests for the DataValidator class,
including data validation, anomaly detection, and data cleaning functions.

Run with: pytest tests/test_data/test_validator.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.validator import DataValidator


class TestDataValidator:
    """Test suite for DataValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a DataValidator instance."""
        return DataValidator()
    
    @pytest.fixture
    def valid_ohlcv_data(self):
        """Generate valid OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure valid OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1) * 1.01
        data['low'] = data[['open', 'low', 'close']].min(axis=1) * 0.99
        
        return data
    
    @pytest.fixture
    def data_with_missing_values(self, valid_ohlcv_data):
        """Create data with missing values."""
        data = valid_ohlcv_data.copy()
        # Add random missing values
        data.iloc[5:8, data.columns.get_loc('close')] = np.nan
        data.iloc[15:17, data.columns.get_loc('volume')] = np.nan
        data.iloc[25, data.columns.get_loc('high')] = np.nan
        return data
    
    @pytest.fixture
    def data_with_anomalies(self, valid_ohlcv_data):
        """Create data with various anomalies."""
        data = valid_ohlcv_data.copy()
        
        # Add price spike (500% change)
        data.iloc[10, data.columns.get_loc('close')] = data.iloc[9]['close'] * 6
        
        # Add negative price
        data.iloc[20, data.columns.get_loc('low')] = -1000
        
        # Add invalid OHLC relationship (High < Low)
        data.iloc[30, data.columns.get_loc('high')] = 30000
        data.iloc[30, data.columns.get_loc('low')] = 50000
        
        # Add close outside high-low range
        data.iloc[40, data.columns.get_loc('close')] = 60000
        data.iloc[40, data.columns.get_loc('high')] = 45000
        
        # Add zero price
        data.iloc[50, data.columns.get_loc('open')] = 0
        
        return data
    
    @pytest.fixture
    def data_with_volume_issues(self, valid_ohlcv_data):
        """Create data with volume issues."""
        data = valid_ohlcv_data.copy()
        
        # Add negative volume
        data.iloc[5, data.columns.get_loc('volume')] = -100
        
        # Add extreme volume spike (100x average)
        avg_volume = data['volume'].mean()
        data.iloc[10, data.columns.get_loc('volume')] = avg_volume * 100
        
        # Add zero volumes
        data.iloc[20:25, data.columns.get_loc('volume')] = 0
        
        return data
    
    @pytest.fixture
    def data_with_timestamp_issues(self, valid_ohlcv_data):
        """Create data with timestamp issues."""
        data = valid_ohlcv_data.copy()
        
        # Create duplicate timestamps
        data = pd.concat([data, data.iloc[[10]]])
        
        # Shuffle to break monotonic order
        data = data.sample(frac=1)
        
        return data
    
    def test_initialization(self):
        """Test DataValidator initialization."""
        # Default initialization
        validator = DataValidator()
        assert validator is not None
        assert validator.max_price_change == 0.5
        
        # Custom config
        custom_config = {
            'max_price_change': 0.3,
            'max_volume_spike': 5,
            'min_volume': 10
        }
        validator_custom = DataValidator(custom_config)
        assert validator_custom.max_price_change == 0.3
        assert validator_custom.max_volume_spike == 5
    
    def test_validate_valid_data(self, validator, valid_ohlcv_data):
        """Test validation of valid OHLCV data."""
        is_valid, issues = validator.validate_ohlcv(valid_ohlcv_data)
        
        assert is_valid == True or len(issues) == 0  # Should be valid or have minor issues
        print(f"Valid data validation: {is_valid}, Issues: {issues}")
    
    def test_check_missing_data(self, validator, data_with_missing_values):
        """Test detection of missing data."""
        missing_stats = validator.check_missing_data(data_with_missing_values)
        
        assert missing_stats['total_missing'] > 0
        assert 'close' in missing_stats['by_column']
        assert 'volume' in missing_stats['by_column']
        assert len(missing_stats['missing_rows']) > 0
        assert missing_stats['percentage'] > 0
    
    def test_fix_missing_data(self, validator, data_with_missing_values):
        """Test fixing missing data with different methods."""
        # Test interpolation
        fixed_interp = validator.fix_missing_data(data_with_missing_values, method='interpolate')
        assert fixed_interp.isna().sum().sum() == 0
        
        # Test forward fill
        fixed_ffill = validator.fix_missing_data(data_with_missing_values, method='forward')
        assert fixed_ffill.isna().sum().sum() == 0
        
        # Test backward fill
        fixed_bfill = validator.fix_missing_data(data_with_missing_values, method='backward')
        assert fixed_bfill.isna().sum().sum() == 0
        
        # Test mean fill
        fixed_mean = validator.fix_missing_data(data_with_missing_values, method='mean')
        assert fixed_mean.isna().sum().sum() == 0
    
    def test_check_price_anomalies(self, validator, data_with_anomalies):
        """Test detection of price anomalies."""
        anomalies = validator.check_price_anomalies(data_with_anomalies)
        
        assert len(anomalies) > 0
        # Check that anomalies were detected at expected positions
        anomaly_positions = [data_with_anomalies.index.get_loc(idx) if hasattr(idx, 'timestamp') else idx for idx in anomalies]
        # Should detect anomalies around positions 10, 20, 50
        assert any(8 <= pos <= 12 for pos in anomaly_positions if isinstance(pos, int))  # Near position 10
    
    def test_detect_outliers_iqr(self, validator, valid_ohlcv_data):
        """Test IQR outlier detection."""
        # Add outliers
        data = valid_ohlcv_data.copy()
        data.iloc[5, data.columns.get_loc('close')] = 100000  # Extreme high
        data.iloc[10, data.columns.get_loc('close')] = 1000    # Extreme low
        
        outliers = validator._detect_outliers_iqr(data['close'], multiplier=1.5)
        
        assert len(outliers) >= 1  # Should detect at least one outlier
        # Check that outliers include the extreme values
        outlier_indices = [data.index.get_loc(idx) if hasattr(data.index, 'get_loc') else idx for idx in outliers]
        assert any(idx in [5, 10] or (hasattr(idx, '__iter__') and (5 in idx or 10 in idx)) for idx in outlier_indices)
    
    def test_detect_outliers_zscore(self, validator, valid_ohlcv_data):
        """Test Z-score outlier detection."""
        # Add outliers
        data = valid_ohlcv_data.copy()
        data.iloc[5, data.columns.get_loc('close')] = 100000  # Extreme high
        
        outliers = validator._detect_outliers_zscore(data['close'], threshold=2.0)
        
        assert len(outliers) >= 1
        # Check that the outlier at position 5 was detected
        if outliers:
            outlier_positions = [data.index.get_loc(idx) if hasattr(data.index, 'get_loc') else idx for idx in outliers]
            assert any(pos == 5 or (hasattr(pos, '__iter__') and 5 in pos) for pos in outlier_positions)
    
    def test_validate_volume(self, validator, data_with_volume_issues):
        """Test volume validation."""
        is_valid = validator.validate_volume(data_with_volume_issues)
        
        assert is_valid == False  # Should fail due to negative volume
    
    def test_fix_volume_issues(self, validator, data_with_volume_issues):
        """Test fixing volume issues."""
        fixed = validator._fix_volume_issues(data_with_volume_issues)
        
        # Check no negative volumes
        assert (fixed['volume'] >= 0).all()
        
        # Check extreme spikes are capped
        volume_mean = fixed['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = fixed['volume'] / volume_mean
        assert volume_ratio.max() < 20  # Should be capped
    
    def test_validate_timestamps(self, validator, data_with_timestamp_issues):
        """Test timestamp validation."""
        is_valid = validator.validate_timestamps(data_with_timestamp_issues)
        
        assert is_valid == False  # Should fail due to duplicates and non-monotonic
    
    def test_ensure_data_continuity(self, validator, valid_ohlcv_data):
        """Test ensuring data continuity."""
        # Remove some rows to create gaps
        data_with_gaps = valid_ohlcv_data.drop(valid_ohlcv_data.index[10:15])
        
        # Ensure continuity
        continuous_data = validator.ensure_data_continuity(data_with_gaps, '1h')
        
        # Check no gaps
        time_diff = pd.Series(continuous_data.index).diff()
        assert time_diff.max() <= timedelta(hours=1)
    
    def test_validate_ohlc_relationships(self, validator, data_with_anomalies):
        """Test validation of OHLC relationships."""
        issues = validator._validate_ohlc_relationships(data_with_anomalies)
        
        assert len(issues) > 0
        assert any('High < Low' in issue for issue in issues)
    
    def test_fix_ohlc_relationships(self, validator, data_with_anomalies):
        """Test fixing OHLC relationships."""
        fixed = validator._fix_ohlc_relationships(data_with_anomalies)
        
        # Check all relationships are valid
        assert (fixed['high'] >= fixed['low']).all()
        assert (fixed['high'] >= fixed['open']).all()
        assert (fixed['high'] >= fixed['close']).all()
        assert (fixed['low'] <= fixed['open']).all()
        assert (fixed['low'] <= fixed['close']).all()
    
    def test_remove_outliers(self, validator, data_with_anomalies):
        """Test outlier removal."""
        # Remove outliers using IQR
        cleaned_iqr = validator.remove_outliers(data_with_anomalies, method='iqr')
        assert len(cleaned_iqr) < len(data_with_anomalies)
        
        # Remove outliers using Z-score
        cleaned_zscore = validator.remove_outliers(data_with_anomalies, method='zscore')
        assert len(cleaned_zscore) < len(data_with_anomalies)
    
    def test_comprehensive_validation(self, validator, data_with_anomalies):
        """Test comprehensive validation with auto-fix."""
        # Validate without fixing
        is_valid_before, issues_before = validator.validate_ohlcv(data_with_anomalies, auto_fix=False)
        assert is_valid_before == False
        assert len(issues_before) > 0
        
        # Validate with auto-fix
        is_valid_after, issues_after = validator.validate_ohlcv(data_with_anomalies, auto_fix=True)
        # After auto-fix, should have fewer issues
        assert len(issues_after) <= len(issues_before)
    
    def test_generate_validation_report(self, validator, data_with_anomalies):
        """Test validation report generation."""
        report = validator.generate_validation_report(data_with_anomalies)
        
        assert 'timestamp' in report
        assert 'data_shape' in report
        assert 'date_range' in report
        assert 'validation_results' in report
        assert 'statistics' in report
        assert 'recommendations' in report
        
        # Check statistics are calculated
        assert 'close' in report['statistics']
        assert 'mean' in report['statistics']['close']
        assert 'std' in report['statistics']['close']
        
        # Should have recommendations due to anomalies
        assert len(report['recommendations']) > 0
    
    def test_edge_cases(self, validator):
        """Test edge cases and error handling."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        is_valid, issues = validator.validate_ohlcv(empty_df)
        assert is_valid == False
        assert 'empty' in issues[0].lower()
        
        # DataFrame without required columns
        bad_columns_df = pd.DataFrame({
            'price': [100, 200, 300],
            'amount': [10, 20, 30]
        })
        # This should fail early due to missing columns
        is_valid, issues = validator.validate_ohlcv(bad_columns_df)
        assert is_valid == False
        # Should detect missing columns before trying to validate relationships
        assert any('Missing columns' in issue or 'missing' in issue.lower() for issue in issues)
        
        # All NaN data
        all_nan_df = pd.DataFrame({
            'open': [np.nan] * 10,
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'close': [np.nan] * 10,
            'volume': [np.nan] * 10
        })
        fixed = validator.fix_missing_data(all_nan_df, method='mean')
        # Should handle gracefully even if all values are NaN
        assert fixed is not None
    
    def test_30m_timeframe_support(self, validator):
        """Test support for 30-minute timeframe data."""
        # Create 30-minute data
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='30min')
        data_30m = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure continuity for 30m timeframe
        continuous = validator.ensure_data_continuity(data_30m, '30m')
        
        # Check that time differences are 30 minutes
        time_diff = pd.Series(continuous.index).diff().dropna()
        assert all(td == timedelta(minutes=30) for td in time_diff)


class TestValidatorIntegration:
    """Integration tests for DataValidator with real-world scenarios."""
    
    @pytest.fixture
    def realistic_bad_data(self):
        """Create realistic data with multiple issues."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        np.random.seed(123)
        
        # Start with good data
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Add realistic issues
        # Flash crash
        data.iloc[100:102, data.columns.get_loc('low')] = data.iloc[99]['low'] * 0.5
        data.iloc[100:102, data.columns.get_loc('close')] = data.iloc[99]['close'] * 0.55
        
        # Missing data (exchange downtime)
        data.iloc[200:210, :] = np.nan
        
        # Volume spike (big trade)
        data.iloc[300, data.columns.get_loc('volume')] = data['volume'].mean() * 15
        
        # Wrong OHLC (data error)
        data.iloc[400, data.columns.get_loc('high')] = data.iloc[400]['low'] - 100
        
        return data
    
    def test_full_validation_pipeline(self, realistic_bad_data):
        """Test complete validation and cleaning pipeline."""
        validator = DataValidator()
        
        # Initial validation
        is_valid_before, issues_before = validator.validate_ohlcv(realistic_bad_data)
        print(f"\nInitial validation: Valid={is_valid_before}")
        print(f"Issues found: {len(issues_before)}")
        for issue in issues_before:
            print(f"  - {issue}")
        
        # Clean the data step by step
        cleaned_data = realistic_bad_data.copy()
        
        # Fix missing data
        cleaned_data = validator.fix_missing_data(cleaned_data)
        
        # Fix OHLC relationships
        cleaned_data = validator._fix_ohlc_relationships(cleaned_data)
        
        # Fix volume issues
        cleaned_data = validator._fix_volume_issues(cleaned_data)
        
        # Remove extreme outliers
        cleaned_data = validator.remove_outliers(cleaned_data, method='iqr')
        
        # Final validation
        is_valid_after, issues_after = validator.validate_ohlcv(cleaned_data)
        print(f"\nFinal validation: Valid={is_valid_after}")
        print(f"Remaining issues: {len(issues_after)}")
        
        # Generate report
        report = validator.generate_validation_report(cleaned_data)
        print(f"\nValidation Report Summary:")
        print(f"  Data shape: {report['data_shape']}")
        print(f"  Date range: {report['date_range']['start']} to {report['date_range']['end']}")
        print(f"  Recommendations: {report['recommendations']}")
        
        assert len(issues_after) < len(issues_before)
    
    def test_performance_with_large_dataset(self):
        """Test validator performance with large dataset."""
        # Create large dataset (1 year of 5-minute data)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='5min')
        large_data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        print(f"\nTesting with large dataset: {len(large_data)} rows")
        
        validator = DataValidator()
        
        # Time the validation
        import time
        start_time = time.time()
        is_valid, issues = validator.validate_ohlcv(large_data)
        end_time = time.time()
        
        print(f"Validation completed in {end_time - start_time:.2f} seconds")
        print(f"Valid: {is_valid}, Issues: {len(issues)}")
        
        assert end_time - start_time < 10  # Should complete within 10 seconds


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])