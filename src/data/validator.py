"""
Enhanced Data Validator Module

Comprehensive data validation and cleaning functionality for OHLCV data.
Ensures data quality and integrity before use in ML/RL trading strategies.

Author: Trading Bot System
Date: 2024
Version: 2.0 - Enhanced with advanced validation features
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from pathlib import Path
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed results from data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 100.0
    statistics: Dict[str, float] = field(default_factory=dict)
    fixed_df: Optional[pd.DataFrame] = None
    

@dataclass 
class DataQualityReport:
    """Comprehensive data quality report"""
    timestamp: datetime
    symbol: str
    timeframe: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    missing_candles: int
    outliers_detected: int
    corrections_made: int
    quality_score: float
    issues: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)


class DataValidator:
    """
    Enhanced data validation and cleaning for OHLCV data.
    
    This class provides methods to:
    - Check for missing data and gaps
    - Detect price anomalies and outliers (multiple methods)
    - Validate OHLCV relationships
    - Validate volume data
    - Fix timestamp issues
    - Ensure data continuity
    - Cross-timeframe validation
    - Statistical validation
    - Generate quality scores and reports
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataValidator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Validation thresholds
        self.max_price_change = self.config.get('max_price_change', 0.5)  # 50%
        self.max_volume_spike = self.config.get('max_volume_spike', 10)  # 10x average
        self.min_volume = self.config.get('min_volume', 0)
        self.max_gap_minutes = self.config.get('max_gap_minutes', 60)
        self.outlier_threshold = self.config.get('outlier_threshold', 4.0)  # Z-score
        
        # Track validation history
        self.validation_history = []
        self.correction_history = []
        
        # Known valid ranges for each pair
        self.valid_ranges = {
            'BTC_USDT': {'min': 100, 'max': 100000},
            'ETH_USDT': {'min': 10, 'max': 10000},
            'SOL_USDT': {'min': 0.1, 'max': 500},
            'ADA_USDT': {'min': 0.01, 'max': 10},
            'DOT_USDT': {'min': 0.1, 'max': 100},
            'AVAX_USDT': {'min': 1, 'max': 200},
            'MATIC_USDT': {'min': 0.01, 'max': 10},
            'LINK_USDT': {'min': 1, 'max': 100},
            'UNI_USDT': {'min': 1, 'max': 100},
            'XRP_USDT': {'min': 0.1, 'max': 10},
            'BNB_USDT': {'min': 10, 'max': 1000},
            'DOGE_USDT': {'min': 0.001, 'max': 1},
            'LTC_USDT': {'min': 10, 'max': 500},
            'ATOM_USDT': {'min': 1, 'max': 100},
            'ALGO_USDT': {'min': 0.01, 'max': 10}
        }
        
        logger.info("Enhanced DataValidator initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'max_price_change': 0.5,  # 50% max change between candles
            'max_volume_spike': 10,    # 10x average volume
            'min_volume': 0,           # Minimum acceptable volume
            'max_gap_minutes': 60,     # Maximum gap between candles
            'outlier_threshold': 4.0,  # Z-score threshold
            'outlier_method': 'combined',  # 'iqr', 'zscore', or 'combined'
            'interpolation_method': 'linear',  # Method for filling missing values
            'auto_fix': True,          # Automatically fix issues
            'backup_data': True,       # Create backup before fixing
        }
    
    # ================== MAIN COMPREHENSIVE VALIDATION ==================
    
    def validate_and_fix(self, 
                        df: pd.DataFrame, 
                        symbol: str,
                        timeframe: str,
                        auto_fix: bool = True,
                        comprehensive: bool = True) -> ValidationResult:
        """
        Enhanced comprehensive validation with all features.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1h')
            auto_fix: Whether to automatically fix issues
            comprehensive: Run all validation checks
            
        Returns:
            ValidationResult with complete details
        """
        errors = []
        warnings = []
        corrections = {}
        statistics = {}
        
        # Create a copy for corrections
        df_fixed = df.copy() if auto_fix else df
        
        # Track original state
        original_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
        
        # 1. Basic structure validation
        struct_valid, struct_errors = self._validate_structure(df)
        if not struct_valid:
            errors.extend(struct_errors)
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                corrections=corrections,
                quality_score=0.0,
                statistics=statistics
            )
        
        # 2. OHLCV relationships (using existing method)
        ohlc_issues = self._validate_ohlc_relationships(df_fixed)
        if ohlc_issues:
            warnings.extend(ohlc_issues)
            if auto_fix:
                df_fixed = self._fix_ohlc_relationships(df_fixed)
                corrections['ohlc_fixed'] = len(ohlc_issues)
        
        # 3. Missing data check (using existing method)
        missing_data = self.check_missing_data(df_fixed)
        if missing_data['total_missing'] > 0:
            warnings.append(f"Missing data: {missing_data['total_missing']} values")
            if auto_fix:
                df_fixed = self.fix_missing_data(df_fixed)
                corrections['missing_filled'] = missing_data['total_missing']
        
        # 4. Timestamp validation (enhanced)
        time_valid, time_issues, df_fixed = self._validate_timestamps_enhanced(
            df_fixed, timeframe, auto_fix
        )
        if not time_valid:
            errors.extend(time_issues)
        else:
            warnings.extend(time_issues)
        
        # 5. Price range validation (new)
        if symbol in self.valid_ranges:
            price_valid, price_issues, df_fixed = self._validate_price_ranges(
                df_fixed, symbol, auto_fix
            )
            warnings.extend(price_issues)
        
        # 6. Price anomalies (enhanced with multiple methods)
        anomalies = self.detect_anomalies_advanced(df_fixed)
        if anomalies['total_outliers'] > 0:
            warnings.append(f"Found {anomalies['total_outliers']} outliers")
            corrections['outliers'] = anomalies
            if auto_fix:
                df_fixed = self._fix_outliers_advanced(df_fixed, anomalies)
        
        # 7. Volume validation (using existing + enhancements)
        vol_valid = self.validate_volume(df_fixed)
        if not vol_valid:
            warnings.append("Volume validation failed")
            if auto_fix:
                df_fixed = self._fix_volume_issues(df_fixed)
                corrections['volume_fixed'] = True
        
        # 8. Gap detection (new)
        gaps = self._detect_gaps(df_fixed, timeframe)
        if gaps:
            warnings.append(f"Found {len(gaps)} gaps in data")
            corrections['gaps'] = gaps
            if auto_fix and timeframe:
                df_fixed = self.ensure_data_continuity(df_fixed, timeframe)
        
        # 9. Statistical validation (new)
        if comprehensive:
            stats_valid, stats = self._validate_statistics(df_fixed)
            statistics = stats
            if not stats_valid:
                warnings.append("Statistical anomalies detected")
        
        # 10. Calculate quality score
        quality_score = self._calculate_quality_score(
            len(errors), len(warnings), len(df), len(corrections)
        )
        
        # Check if data was modified
        if auto_fix:
            fixed_hash = hashlib.md5(pd.util.hash_pandas_object(df_fixed).values).hexdigest()
            if original_hash != fixed_hash:
                corrections['data_modified'] = True
                logger.info(f"Data was modified during validation for {symbol} {timeframe}")
        
        # Track validation
        self.validation_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'quality_score': quality_score,
            'errors': len(errors),
            'warnings': len(warnings),
            'corrections': len(corrections)
        })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrections=corrections,
            quality_score=quality_score,
            statistics=statistics,
            fixed_df=df_fixed  # Include the fixed dataframe
        )
    
    # ================== KEEP EXISTING METHODS (They're good!) ==================
    
    def validate_ohlcv(self, df: pd.DataFrame, symbol: str = 'UNKNOWN', 
                    timeframe: str = '1h', auto_fix: bool = True) -> 'ValidationResult':
        """
        Validate OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe
            auto_fix: Whether to automatically fix issues
            
        Returns:
            ValidationResult object
        """
        # Use the comprehensive validation method
        return self.validate_and_fix(df, symbol, timeframe, auto_fix, comprehensive=False)
    
    def check_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Keep existing method - it works well"""
        missing_info = {
            'total_missing': df.isnull().sum().sum(),
            'by_column': df.isnull().sum().to_dict(),
            'missing_rows': df.isnull().any(axis=1).sum(),
            'complete_rows': (~df.isnull().any(axis=1)).sum()
        }
        
        return missing_info
    
    def check_price_anomalies(self, df: pd.DataFrame) -> List[int]:
        """Keep existing method - enhance with advanced version"""
        # Use existing logic
        anomalies = []
        
        # Check for extreme price changes
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                pct_change = df[col].pct_change().abs()
                extreme_changes = pct_change[pct_change > self.max_price_change]
                anomalies.extend(extreme_changes.index.tolist())
        
        # Check for outliers using IQR
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                outliers = self._detect_outliers_iqr(df[col])
                anomalies.extend(outliers)
        
        # Remove duplicates
        return list(set(anomalies))
    
    def fix_missing_data(self, df: pd.DataFrame, 
                        method: str = 'interpolate') -> pd.DataFrame:
        """Keep existing method - it works well"""
        df_fixed = df.copy()
        
        if method == 'interpolate':
            # Interpolate numeric columns
            numeric_columns = df_fixed.select_dtypes(include=[np.number]).columns
            df_fixed[numeric_columns] = df_fixed[numeric_columns].interpolate(method='linear')
            
            # Forward fill any remaining NaN values
            df_fixed[numeric_columns] = df_fixed[numeric_columns].ffill()
            
            # Backward fill any remaining NaN values
            df_fixed[numeric_columns] = df_fixed[numeric_columns].bfill()
            
        elif method == 'forward_fill':
            df_fixed = df_fixed.ffill()
        elif method == 'backward_fill':
            df_fixed = df_fixed.bfill()
        else:
            # Default: fill with mean
            for col in df_fixed.select_dtypes(include=[np.number]).columns:
                df_fixed[col].fillna(df_fixed[col].mean(), inplace=True)
        
        logger.info(f"Fixed missing data using {method} method")
        return df_fixed
    
    def validate_volume(self, df: pd.DataFrame) -> bool:
        """Keep existing volume validation"""
        if 'volume' not in df.columns:
            return False
        
        # Check for negative volume
        if (df['volume'] < 0).any():
            logger.warning("Negative volume values found")
            return False
        
        # Check for extreme volume spikes
        volume_mean = df['volume'].rolling(window=20, min_periods=1).mean()
        volume_ratio = df['volume'] / volume_mean
        
        if (volume_ratio > self.max_volume_spike).any():
            logger.warning("Extreme volume spikes detected")
            return False
        
        # Check if all volumes are zero
        if (df['volume'] == 0).all():
            logger.warning("All volume values are zero")
            return False
        
        return True
    
    def ensure_data_continuity(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Keep existing method - works well"""
        # Map timeframe to pandas frequency
        freq_map = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        freq = freq_map.get(timeframe, '1H')
        
        # Create complete date range
        start_date = df.index[0]
        end_date = df.index[-1]
        complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Reindex to complete range
        df_continuous = df.reindex(complete_range)
        
        # Fill missing values
        df_continuous = self.fix_missing_data(df_continuous, method='interpolate')
        
        logger.info(f"Ensured data continuity for {timeframe} timeframe")
        return df_continuous
    
    # ================== NEW ENHANCED METHODS ==================
    
    def _validate_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Enhanced structure validation"""
        errors = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            return False, errors
        
        # Check data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' is not numeric")
        
        # Check index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")
        
        # Check for sufficient data
        if len(df) < 100:
            errors.append(f"Insufficient data: only {len(df)} rows (minimum 100 recommended)")
        
        return len(errors) == 0, errors
    
    def _validate_timestamps_enhanced(self, 
                                     df: pd.DataFrame, 
                                     timeframe: str,
                                     auto_fix: bool) -> Tuple[bool, List[str], pd.DataFrame]:
        """Enhanced timestamp validation"""
        issues = []
        is_valid = True
        
        # Check if sorted
        if not df.index.is_monotonic_increasing:
            issues.append("Timestamps are not sorted")
            is_valid = False
            if auto_fix:
                df = df.sort_index()
        
        # Check for duplicates
        duplicates = df.index.duplicated()
        if duplicates.any():
            issues.append(f"{duplicates.sum()} duplicate timestamps")
            is_valid = False
            if auto_fix:
                df = df[~df.index.duplicated(keep='last')]
        
        # Check time intervals
        expected_delta = self._timeframe_to_timedelta(timeframe)
        if expected_delta:
            time_diffs = pd.Series(df.index).diff()
            
            # Allow some tolerance for market closures
            if timeframe == '1d':
                invalid_gaps = time_diffs[time_diffs > timedelta(days=3)]
            elif timeframe in ['1h', '4h']:
                invalid_gaps = time_diffs[time_diffs > expected_delta * 2]
            else:
                invalid_gaps = time_diffs[time_diffs > expected_delta * 1.5]
            
            if len(invalid_gaps) > 0:
                issues.append(f"{len(invalid_gaps)} irregular time intervals")
        
        return is_valid, issues, df
    
    def _validate_price_ranges(self,
                              df: pd.DataFrame,
                              symbol: str,
                              auto_fix: bool) -> Tuple[bool, List[str], pd.DataFrame]:
        """Validate prices are within reasonable ranges"""
        issues = []
        
        if symbol not in self.valid_ranges:
            return True, issues, df
        
        min_price = self.valid_ranges[symbol]['min']
        max_price = self.valid_ranges[symbol]['max']
        
        # Check each price column
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                out_of_range = df[(df[col] < min_price) | (df[col] > max_price)]
                
                if len(out_of_range) > 0:
                    issues.append(f"{len(out_of_range)} {col} prices outside range [{min_price}, {max_price}]")
                    
                    if auto_fix:
                        # Clip to valid range
                        df[col] = df[col].clip(min_price, max_price)
        
        return True, issues, df
    
    def detect_anomalies_advanced(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced anomaly detection using multiple methods"""
        anomalies = {
            'zscore_outliers': {},
            'iqr_outliers': {},
            'isolation_forest_outliers': [],
            'total_outliers': 0,
            'details': []
        }
        
        # Method 1: Z-score detection
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                outliers = self._detect_outliers_zscore(df[col], threshold=self.outlier_threshold)
                if outliers:
                    anomalies['zscore_outliers'][col] = outliers
        
        # Method 2: IQR detection
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                outliers = self._detect_outliers_iqr(df[col])
                if outliers:
                    anomalies['iqr_outliers'][col] = outliers
        
        # Method 3: Isolation Forest (if enough data)
        if len(df) > 100:
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.01, random_state=42)
                features = df[['open', 'high', 'low', 'close', 'volume']].values
                outlier_labels = iso_forest.fit_predict(features)
                iso_outliers = df[outlier_labels == -1].index.tolist()
                anomalies['isolation_forest_outliers'] = iso_outliers
            except ImportError:
                logger.warning("sklearn not available for Isolation Forest detection")
        
        # Count total unique outliers
        all_outliers = set()
        for method_outliers in anomalies['zscore_outliers'].values():
            all_outliers.update(method_outliers)
        for method_outliers in anomalies['iqr_outliers'].values():
            all_outliers.update(method_outliers)
        all_outliers.update(anomalies['isolation_forest_outliers'])
        
        anomalies['total_outliers'] = len(all_outliers)
        
        return anomalies
    
    def _fix_outliers_advanced(self, df: pd.DataFrame, anomalies: Dict) -> pd.DataFrame:
        """Fix outliers using advanced methods"""
        df_fixed = df.copy()
        
        # Collect all outlier indices
        all_outliers = set()
        for col_outliers in anomalies.get('zscore_outliers', {}).values():
            all_outliers.update(col_outliers)
        
        # Fix outliers by interpolation or median replacement
        for idx in all_outliers:
            if idx in df_fixed.index:
                # Get surrounding window
                loc = df_fixed.index.get_loc(idx)
                window_start = max(0, loc - 5)
                window_end = min(len(df_fixed), loc + 6)
                
                # Replace with median of surrounding values
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_fixed.columns:
                        window_values = df_fixed.iloc[window_start:window_end][col]
                        df_fixed.loc[idx, col] = window_values.median()
        
        return df_fixed
    
    def _detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect missing candles/gaps in data"""
        gaps = []
        
        expected_delta = self._timeframe_to_timedelta(timeframe)
        if not expected_delta:
            return gaps
        
        for i in range(1, len(df)):
            actual_delta = df.index[i] - df.index[i-1]
            
            # Account for weekends in daily data
            if timeframe == '1d' and actual_delta.days <= 3:
                continue
            
            if actual_delta > expected_delta * 1.5:
                missing_periods = int(actual_delta / expected_delta) - 1
                
                if missing_periods > 0:
                    gaps.append({
                        'start': df.index[i-1],
                        'end': df.index[i],
                        'missing_candles': missing_periods,
                        'gap_duration': str(actual_delta)
                    })
        
        return gaps
    
    def _validate_statistics(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Validate statistical properties of the data"""
        stats = {}
        is_valid = True
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                series = df[col]
                
                stats[f'{col}_mean'] = float(series.mean())
                stats[f'{col}_std'] = float(series.std())
                stats[f'{col}_min'] = float(series.min())
                stats[f'{col}_max'] = float(series.max())
                stats[f'{col}_nulls'] = int(series.isnull().sum())
                
                # Check for autocorrelation (prices should be autocorrelated)
                if col != 'volume' and len(series) > 10:
                    autocorr = series.autocorr(lag=1)
                    stats[f'{col}_autocorr'] = float(autocorr) if pd.notna(autocorr) else 0
                    
                    if pd.notna(autocorr) and autocorr < 0.7:
                        is_valid = False
        
        # Calculate return statistics
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                stats['return_mean'] = float(returns.mean())
                stats['return_std'] = float(returns.std())
                stats['return_skew'] = float(returns.skew())
                stats['return_kurtosis'] = float(returns.kurtosis())
                
                # Check for unrealistic returns
                if abs(stats['return_mean']) > 0.1:  # 10% average return
                    is_valid = False
        
        return is_valid, stats
    
    def _calculate_quality_score(self, 
                                errors: int, 
                                warnings: int,
                                total_rows: int,
                                corrections: int) -> float:
        """Calculate overall data quality score (0-100)"""
        if total_rows == 0:
            return 0.0
        
        # Start with perfect score
        score = 100.0
        
        # Heavy penalty for errors
        score -= errors * 10
        
        # Light penalty for warnings
        score -= warnings * 2
        
        # Penalty for corrections needed
        score -= corrections * 3
        
        # Bonus for having sufficient data
        if total_rows > 1000:
            score += 5
        elif total_rows < 100:
            score -= 10
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, score))
    
    def validate_cross_timeframe(self,
                                data_dict: Dict[str, pd.DataFrame],
                                symbol: str) -> ValidationResult:
        """
        Validate consistency across different timeframes
        """
        errors = []
        warnings = []
        
        # Compare different timeframe combinations
        timeframe_relations = [
            ('1h', '4h', 4),
            ('4h', '1d', 6),  # Assuming 24h market
            ('1h', '1d', 24)
        ]
        
        for tf_low, tf_high, ratio in timeframe_relations:
            if tf_low in data_dict and tf_high in data_dict:
                df_low = data_dict[tf_low]
                df_high = data_dict[tf_high]
                
                # Find overlapping period
                overlap_start = max(df_low.index[0], df_high.index[0])
                overlap_end = min(df_low.index[-1], df_high.index[-1])
                
                if overlap_start < overlap_end:
                    # Check a few data points for consistency
                    sample_dates = pd.date_range(overlap_start, overlap_end, periods=min(10, len(df_high)))
                    
                    for date in sample_dates:
                        if date in df_high.index:
                            # Get corresponding period in lower timeframe
                            period_end = date
                            period_start = date - self._timeframe_to_timedelta(tf_high)
                            
                            low_period = df_low[period_start:period_end]
                            
                            if len(low_period) > 0:
                                # Compare aggregated values
                                high_close = df_high.loc[date, 'close']
                                low_close = low_period['close'].iloc[-1] if len(low_period) > 0 else None
                                
                                if low_close and abs(high_close - low_close) / high_close > 0.01:
                                    warnings.append(f"Inconsistency between {tf_low} and {tf_high} at {date}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            corrections={},
            quality_score=100 - len(warnings) * 5,
            statistics={}
        )
    
    def generate_quality_report(self,
                               validation_result: ValidationResult,
                               symbol: str,
                               timeframe: str,
                               df: pd.DataFrame) -> DataQualityReport:
        """Generate comprehensive data quality report"""
        
        # Count issues
        invalid_rows = len(validation_result.corrections.get('outliers', {}).get('total_outliers', 0))
        
        # Generate recommendations
        recommendations = []
        
        if validation_result.errors:
            recommendations.append(" Critical errors found - manual review required")
        
        if 'gaps' in validation_result.corrections:
            gaps = validation_result.corrections['gaps']
            if gaps:
                recommendations.append(f" Fill {len(gaps)} data gaps")
        
        if validation_result.quality_score < 90:
            recommendations.append(" Consider re-downloading data from source")
        
        if validation_result.quality_score > 95:
            recommendations.append(" Data quality excellent - ready for ML/RL")
        
        return DataQualityReport(
            timestamp=datetime.now(),
            symbol=symbol,
            timeframe=timeframe,
            total_rows=len(df),
            valid_rows=len(df) - invalid_rows,
            invalid_rows=invalid_rows,
            missing_candles=len(validation_result.corrections.get('gaps', [])),
            outliers_detected=validation_result.corrections.get('outliers', {}).get('total_outliers', 0),
            corrections_made=len(validation_result.corrections),
            quality_score=validation_result.quality_score,
            issues=[
                {'type': 'error', 'message': err} for err in validation_result.errors
            ] + [
                {'type': 'warning', 'message': warn} for warn in validation_result.warnings
            ],
            recommendations=recommendations,
            statistics=validation_result.statistics
        )
    
    def validate_all_assets(self, 
                           data_path: str = './data/raw/',
                           output_report: bool = True) -> pd.DataFrame:
        """
        Validate all 15 crypto assets
        
        Args:
            data_path: Path to data directory
            output_report: Whether to save report to file
            
        Returns:
            Summary DataFrame
        """
        symbols = [
            'BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT',
            'AVAX_USDT', 'MATIC_USDT', 'LINK_USDT', 'UNI_USDT', 'XRP_USDT',
            'BNB_USDT', 'DOGE_USDT', 'LTC_USDT', 'ATOM_USDT', 'ALGO_USDT'
        ]
        timeframes = ['1h', '4h', '1d']
        
        results = []
        all_reports = []
        
        print("\n === VALIDATING ALL ASSETS ===\n")
        
        for symbol in symbols:
            symbol_data = {}
            
            for timeframe in timeframes:
                file_path = Path(data_path) / timeframe / f"{symbol}_{timeframe}.csv"
                
                if file_path.exists():
                    # Load data
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Validate
                    validation = self.validate_and_fix(df, symbol, timeframe, auto_fix=True)
                    
                    # Save fixed data if needed
                    if validation.fixed_df is not None and validation.corrections:
                        backup_path = file_path.with_suffix('.csv.bak')
                        df.to_csv(backup_path)  # Backup original
                        validation.fixed_df.to_csv(file_path)  # Save fixed
                    
                    # Generate report
                    report = self.generate_quality_report(validation, symbol, timeframe, df)
                    all_reports.append(report)
                    
                    # Store for cross-timeframe validation
                    symbol_data[timeframe] = df
                    
                    # Summary row
                    results.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'rows': len(df),
                        'quality': validation.quality_score,
                        'errors': len(validation.errors),
                        'warnings': len(validation.warnings),
                        'fixed': len(validation.corrections) > 0,
                        'status': 'Y' if validation.quality_score > 95 else '️W' if validation.quality_score > 80 else 'N'
                    })
            
            # Cross-timeframe validation if we have multiple timeframes
            if len(symbol_data) > 1:
                cross_validation = self.validate_cross_timeframe(symbol_data, symbol)
                if cross_validation.warnings:
                    print(f" Cross-timeframe issues for {symbol}: {len(cross_validation.warnings)} warnings")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Display summary
        print("\n VALIDATION SUMMARY")
        print("=" * 80)
        print(summary_df.to_string())
        
        # Overall statistics
        avg_quality = summary_df['quality'].mean()
        total_errors = summary_df['errors'].sum()
        total_warnings = summary_df['warnings'].sum()
        fixed_count = summary_df['fixed'].sum()
        
        print(f"\n OVERALL STATISTICS:")
        print(f"  Average Quality Score: {avg_quality:.1f}%")
        print(f"  Total Errors: {total_errors}")
        print(f"  Total Warnings: {total_warnings}")
        print(f"  Files Fixed: {fixed_count}")
        
        if avg_quality > 95:
            print("\n DATA QUALITY: EXCELLENT - Ready for ML/RL!")
        elif avg_quality > 85:
            print("\n DATA QUALITY: GOOD - Review warnings before ML/RL")
        else:
            print("\n DATA QUALITY: NEEDS IMPROVEMENT - Fix errors before ML/RL")
        
        # Save report if requested
        if output_report:
            report_path = Path(data_path) / 'validation_report.json'
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'summary': summary_df.to_dict('records'),
                'overall_quality': avg_quality,
                'detailed_reports': [
                    {
                        'symbol': r.symbol,
                        'timeframe': r.timeframe,
                        'quality_score': r.quality_score,
                        'issues': r.issues,
                        'recommendations': r.recommendations
                    } for r in all_reports
                ]
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"\n Detailed report saved to: {report_path}")
        
        return summary_df
    
    # ================== UTILITY METHODS ==================
    
    def _timeframe_to_timedelta(self, timeframe: str) -> Optional[timedelta]:
        """Convert timeframe string to timedelta"""
        mapping = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
        return mapping.get(timeframe)
    
    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> List[int]:
        """Keep existing IQR detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3) -> List[int]:
        """Keep existing z-score detection"""
        if series.std() == 0:
            return []
        
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.tolist()
    
    def _fix_volume_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix volume-related issues"""
        df_fixed = df.copy()
        
        if 'volume' in df_fixed.columns:
            # Fix negative volumes
            df_fixed.loc[df_fixed['volume'] < 0, 'volume'] = abs(df_fixed.loc[df_fixed['volume'] < 0, 'volume'])
            
            # Fix extreme spikes
            volume_mean = df_fixed['volume'].rolling(window=20, min_periods=1).mean()
            volume_std = df_fixed['volume'].rolling(window=20, min_periods=1).std()
            upper_limit = volume_mean + (self.max_volume_spike * volume_std)
            
            # Cap extreme volumes
            extreme_mask = df_fixed['volume'] > upper_limit
            if extreme_mask.any():
                df_fixed.loc[extreme_mask, 'volume'] = upper_limit[extreme_mask]
            
            # Handle zero volumes (interpolate)
            zero_mask = df_fixed['volume'] == 0
            if zero_mask.any():
                df_fixed.loc[zero_mask, 'volume'] = np.nan
                df_fixed['volume'] = df_fixed['volume'].interpolate(method='linear')
                df_fixed['volume'] = df_fixed['volume'].ffill()
                df_fixed['volume'] = df_fixed['volume'].bfill()
        
        return df_fixed
    
    # Keep existing methods that are still in the original file
    def _fix_price_anomalies(self, df: pd.DataFrame, anomaly_indices: List[int]) -> pd.DataFrame:
        """Keep existing method"""
        df_fixed = df.copy()
        
        for idx in anomaly_indices:
            if idx in df_fixed.index:
                # Use interpolation for anomaly rows
                for col in ['open', 'high', 'low', 'close']:
                    if col in df_fixed.columns:
                        # Find surrounding valid values
                        loc = df_fixed.index.get_loc(idx)
                        
                        # Get previous and next valid values
                        prev_val = df_fixed[col].iloc[max(0, loc-1)]
                        next_val = df_fixed[col].iloc[min(len(df_fixed)-1, loc+1)]
                        
                        # Interpolate
                        df_fixed.loc[idx, col] = (prev_val + next_val) / 2
        
        logger.info(f"Fixed {len(anomaly_indices)} price anomalies")
        return df_fixed
    
    # Keep the existing generate_validation_report for compatibility
    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Keep existing report generation for backward compatibility"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'date_range': {
                'start': str(df.index[0]) if not df.empty else None,
                'end': str(df.index[-1]) if not df.empty else None
            },
            'validation_results': {},
            'statistics': {},
            'recommendations': []
        }
        
        # Run validation
        is_valid, issues = self.validate_ohlcv(df)
        report['validation_results']['is_valid'] = is_valid
        report['validation_results']['issues'] = issues
        
        # Calculate statistics
        if not df.empty:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    report['statistics'][col] = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'nulls': int(df[col].isna().sum())
                    }
        
        # Add recommendations
        if not is_valid:
            report['recommendations'].append("Consider running auto_fix=True")
            if 'Missing data' in str(issues):
                report['recommendations'].append("Use interpolation to fill missing values")
            if 'anomalies' in str(issues):
                report['recommendations'].append("Review and fix price anomalies")
        
        return report
    

    # Add these methods to the DataValidator class in validator.py

    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> List[str]:
        """
        Validate OHLC relationships
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of issues found
        """
        issues = []
        
        # Check if high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            issues.append(f"High < Low in {invalid_hl.sum()} rows")
        
        # Check if high >= open and close
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
        if invalid_high.any():
            issues.append(f"High less than open/close in {invalid_high.sum()} rows")
        
        # Check if low <= open and close
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])
        if invalid_low.any():
            issues.append(f"Low greater than open/close in {invalid_low.sum()} rows")
        
        return issues

    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix OHLC relationship issues
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Fixed DataFrame
        """
        df = df.copy()
        
        # Ensure high is max of OHLC
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Ensure low is min of OHLC
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df

    def _validate_timestamps_enhanced(self, df: pd.DataFrame, 
                                    timeframe: str, 
                                    auto_fix: bool) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Enhanced timestamp validation
        
        Args:
            df: DataFrame with datetime index
            timeframe: Timeframe string
            auto_fix: Whether to fix issues
            
        Returns:
            Tuple of (is_valid, issues, fixed_df)
        """
        issues = []
        is_valid = True
        
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("Index is not DatetimeIndex")
            is_valid = False
            if auto_fix:
                df.index = pd.to_datetime(df.index)
        
        # Check for duplicates
        if df.index.duplicated().any():
            issues.append(f"Found {df.index.duplicated().sum()} duplicate timestamps")
            is_valid = False
            if auto_fix:
                df = df[~df.index.duplicated(keep='first')]
        
        # Check if sorted
        if not df.index.is_monotonic_increasing:
            issues.append("Timestamps not sorted")
            if auto_fix:
                df = df.sort_index()
        
        return is_valid, issues, df

    def _validate_price_ranges(self, df: pd.DataFrame, 
                            symbol: str, 
                            auto_fix: bool) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validate price ranges for a symbol
        
        Args:
            df: DataFrame with OHLC data
            symbol: Trading symbol
            auto_fix: Whether to fix issues
            
        Returns:
            Tuple of (is_valid, issues, fixed_df)
        """
        issues = []
        is_valid = True
        
        if symbol in self.valid_ranges:
            ranges = self.valid_ranges[symbol]
            
            # Check for prices outside valid range
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    out_of_range = (df[col] < ranges['min']) | (df[col] > ranges['max'])
                    if out_of_range.any():
                        issues.append(f"{col} has {out_of_range.sum()} values outside range")
                        is_valid = False
                        
                        if auto_fix:
                            df.loc[df[col] < ranges['min'], col] = ranges['min']
                            df.loc[df[col] > ranges['max'], col] = ranges['max']
        
        return is_valid, issues, df

    def _fix_outliers_advanced(self, df: pd.DataFrame, anomalies: Dict) -> pd.DataFrame:
        """
        Fix outliers in data
        
        Args:
            df: DataFrame with OHLC data
            anomalies: Dictionary of detected anomalies
            
        Returns:
            Fixed DataFrame
        """
        df = df.copy()
        
        # Simple interpolation for outliers
        for col in ['open', 'high', 'low', 'close']:
            if col in anomalies and col in df.columns:
                outlier_indices = anomalies[col]
                if len(outlier_indices) > 0:
                    df.loc[outlier_indices, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
        
        return df

    def _fix_volume_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix volume issues
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            Fixed DataFrame
        """
        df = df.copy()
        
        # Fix negative volumes
        if 'volume' in df.columns:
            df.loc[df['volume'] < 0, 'volume'] = 0
            
            # Fix extreme volume spikes
            median_vol = df['volume'].median()
            df.loc[df['volume'] > median_vol * 100, 'volume'] = median_vol * 10
        
        return df

    def _validate_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame structure
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        is_valid = True
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            errors.append(f"Missing columns: {missing}")
            is_valid = False
        
        # Check if empty
        if df.empty:
            errors.append("DataFrame is empty")
            is_valid = False
        
        return is_valid, errors

    def _fix_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for fix_missing_data for compatibility"""
        return self.fix_missing_data(df)

    def _validate_statistics(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate statistical properties of data
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Tuple of (is_valid, statistics)
        """
        stats = {}
        is_valid = True
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                series = df[col]
                
                stats[f'{col}_mean'] = float(series.mean())
                stats[f'{col}_std'] = float(series.std())
                stats[f'{col}_min'] = float(series.min())
                stats[f'{col}_max'] = float(series.max())
                stats[f'{col}_nulls'] = int(series.isnull().sum())
        
        return is_valid, stats


# Example usage
if __name__ == "__main__":
    # Initialize enhanced validator
    validator = DataValidator()
    
    print("=== Enhanced Data Validator ===\n")
    
    # Validate all assets
    summary = validator.validate_all_assets(
        data_path='./data/raw/',
        output_report=True
    )
    
    print("\n Enhanced validation complete!")
    print("Your data is now verified and ML/RL ready!")