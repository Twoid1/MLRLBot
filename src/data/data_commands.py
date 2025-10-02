"""
Data Commands Module
Integration layer for data management CLI commands
Coordinates DataManager, Validator, KrakenConnector, and Database
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from tqdm import tqdm

# Import all components
from .data_manager import DataManager
from .validator import DataValidator, ValidationResult
from .database import DatabaseManager
from .kraken_connector import KrakenConnector

# Setup logging
logger = logging.getLogger(__name__)


class DataCommands:
    """
    Integration layer for data management operations
    Handles: --update and --validate commands
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize data commands handler
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize components
        self.data_manager = DataManager(config_path)
        self.validator = DataValidator()
        self.database = DatabaseManager(db_type='sqlite')
        
        # Initialize Kraken connector (paper mode for updates)
        self.kraken = KrakenConnector(
            mode='paper',  # Use paper mode for data fetching
            data_path=str(self.data_manager.raw_path),
            update_existing_data=True
        )
        
        # Create logs directory
        self.logs_path = Path('./logs')
        self.logs_path.mkdir(exist_ok=True)
        
        logger.info("DataCommands initialized successfully")
    
    # ==================== UPDATE COMMAND ====================
    
    def update_all_data(self, 
                       symbols: Optional[List[str]] = None,
                       timeframes: Optional[List[str]] = None,
                       verbose: bool = True) -> Dict[str, any]:
        """
        Update all data files with latest data from Kraken
        
        Args:
            symbols: List of symbols to update (None = all)
            timeframes: List of timeframes to update (None = all)
            verbose: Print progress information
            
        Returns:
            Dictionary with update results
        """
        print("\n" + "="*60)
        print("UPDATING DATA FROM KRAKEN")
        print("="*60 + "\n")
        
        # Get symbols and timeframes
        if symbols is None:
            symbols = self.data_manager.available_symbols
        if timeframes is None:
            timeframes = self.data_manager.available_timeframes
        
        print(f"Symbols to update: {len(symbols)}")
        print(f"Timeframes to update: {len(timeframes)}")
        print(f"Total operations: {len(symbols) * len(timeframes)}\n")
        
        # Track results
        results = {
            'updated': [],
            'failed': [],
            'already_current': [],
            'total_candles_added': 0,
            'start_time': datetime.now()
        }
        
        # Update each symbol/timeframe combination
        total_ops = len(symbols) * len(timeframes)
        current_op = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                current_op += 1
                
                if verbose:
                    print(f"[{current_op}/{total_ops}] Updating {symbol} {timeframe}...", end=" ")
                
                try:
                    # Update single pair
                    candles_added = self._update_single_pair(symbol, timeframe)
                    
                    if candles_added > 0:
                        results['updated'].append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'candles_added': candles_added
                        })
                        results['total_candles_added'] += candles_added
                        
                        if verbose:
                            print(f" Added {candles_added} candles")
                            
                    else:
                        results['already_current'].append(f"{symbol}_{timeframe}")
                        if verbose:
                            print(" Already current")
                    
                except Exception as e:
                    results['failed'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': str(e)
                    })
                    if verbose:
                        print(f" Failed: {str(e)}")
                    logger.error(f"Failed to update {symbol} {timeframe}: {e}")
        
        # Calculate duration
        results['duration'] = (datetime.now() - results['start_time']).total_seconds()
        
        # Print summary
        self._print_update_summary(results)
        
        # Save update log
        self._save_update_log(results)
        
        return results
    
    def _update_single_pair(self, symbol: str, timeframe: str) -> int:
        """
        Update data for a single symbol/timeframe pair
        
        Args:
            symbol: Trading symbol (e.g., 'BTC_USDT')
            timeframe: Timeframe (e.g., '1h')
            
        Returns:
            Number of new candles added
        """
        # Standardize symbol format for file loading
        file_symbol = symbol.replace('/', '_')
        
        # Load existing data
        existing_df = self.data_manager.load_existing_data(symbol, timeframe)
        
        if existing_df.empty:
            logger.warning(f"No existing data found for {symbol} {timeframe}")
            return 0
        
        # Get last timestamp
        last_timestamp = existing_df.index[-1]
        
        # Check if update is needed
        current_time = datetime.now()
        time_diff = (current_time - last_timestamp).total_seconds() / 3600  # hours
        
        # Get expected update interval
        interval_hours = self._timeframe_to_hours(timeframe)
        
        # If data is recent enough, skip
        if time_diff < interval_hours * 2:
            return 0
        
        # Fetch new data from Kraken
        new_candles = self.kraken._fill_gap_for_pair(file_symbol, timeframe)
        
        # Sync to database if candles were added
        if new_candles > 0:
            updated_df = self.data_manager.load_existing_data(symbol, timeframe, use_cache=False)
            self.database.store_ohlcv(updated_df, symbol, timeframe)
        
        return new_candles
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """Convert timeframe string to hours"""
        mapping = {
            '1m': 1/60, '5m': 5/60, '15m': 15/60, '30m': 30/60,
            '1h': 1, '2h': 2, '4h': 4, '6h': 6, '12h': 12,
            '1d': 24, '1w': 168
        }
        return mapping.get(timeframe, 1)
    
    def _print_update_summary(self, results: Dict) -> None:
        """Print update summary"""
        print("\n" + "="*60)
        print("UPDATE SUMMARY")
        print("="*60)
        print(f" Updated: {len(results['updated'])} files")
        print(f" Already current: {len(results['already_current'])} files")
        print(f" Failed: {len(results['failed'])} files")
        print(f" Total candles added: {results['total_candles_added']}")
        print(f"  Duration: {results['duration']:.2f} seconds")
        
        if results['failed']:
            print("\nFailed updates:")
            for fail in results['failed']:
                print(f"  - {fail['symbol']} {fail['timeframe']}: {fail['error']}")
        
        print("="*60 + "\n")
    
    def _save_update_log(self, results: Dict) -> None:
        """Save update log to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.logs_path / f'data_update_{timestamp}.json'
        
        # Prepare serializable results
        log_data = {
            'timestamp': timestamp,
            'updated_count': len(results['updated']),
            'failed_count': len(results['failed']),
            'already_current_count': len(results['already_current']),
            'total_candles_added': results['total_candles_added'],
            'duration_seconds': results['duration'],
            'updated_files': results['updated'],
            'failed_files': results['failed']
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Update log saved to: {log_file}\n")
    
    # ==================== VALIDATE COMMAND ====================
    
    def validate_all_data(self, 
                         symbols: Optional[List[str]] = None,
                         timeframes: Optional[List[str]] = None,
                         auto_fix: bool = False,
                         verbose: bool = True) -> Dict[str, any]:
        """
        Validate all existing data files with DETAILED error locations
        
        Args:
            symbols: List of symbols to validate (None = all)
            timeframes: List of timeframes to validate (None = all)
            auto_fix: Whether to automatically fix issues (False = report only)
            verbose: Print detailed information
            
        Returns:
            Dictionary with validation results
        """
        print("\n" + "="*60)
        print("VALIDATING DATA")
        print("="*60 + "\n")
        
        # Get symbols and timeframes
        if symbols is None:
            symbols = self.data_manager.available_symbols
        if timeframes is None:
            timeframes = self.data_manager.available_timeframes
        
        print(f"Symbols to validate: {len(symbols)}")
        print(f"Timeframes to validate: {len(timeframes)}")
        print(f"Auto-fix: {'ENABLED' if auto_fix else 'DISABLED'}")
        print(f"Total files to check: {len(symbols) * len(timeframes)}\n")
        
        # Track results
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'files_with_warnings': 0,
            'files_with_errors': 0,
            'missing_files': 0,
            'detailed_results': [],
            'start_time': datetime.now()
        }
        
        # Validate each symbol/timeframe
        for symbol in tqdm(symbols, desc="Validating symbols"):
            for timeframe in timeframes:
                result = self._validate_single_file(symbol, timeframe, auto_fix, verbose)
                
                validation_results['total_files'] += 1
                validation_results['detailed_results'].append(result)
                
                if result['status'] == 'missing':
                    validation_results['missing_files'] += 1
                elif result['status'] == 'valid':
                    validation_results['valid_files'] += 1
                elif result['status'] == 'warnings':
                    validation_results['files_with_warnings'] += 1
                elif result['status'] == 'errors':
                    validation_results['files_with_errors'] += 1
        
        # Calculate duration
        validation_results['duration'] = (datetime.now() - validation_results['start_time']).total_seconds()
        
        # Print summary
        self._print_validation_summary(validation_results)
        
        # Save validation report
        self._save_validation_report(validation_results)
        
        # If errors found, show DETAILED info
        if validation_results['files_with_errors'] > 0:
            self._print_detailed_error_locations(validation_results)
        
        return validation_results
    
    def _validate_single_file(self, symbol: str, timeframe: str, 
                              auto_fix: bool, verbose: bool) -> Dict:
        """
        Validate a single data file with DETAILED error tracking
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            auto_fix: Whether to fix issues automatically
            verbose: Print details
            
        Returns:
            Dictionary with validation result including exact error locations
        """
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'valid',
            'errors': [],
            'warnings': [],
            'quality_score': 100.0,
            'statistics': {},
            'detailed_errors': []  # NEW: Store exact locations
        }
        
        try:
            # Load data
            df = self.data_manager.load_existing_data(symbol, timeframe)
            
            if df.empty:
                result['status'] = 'missing'
                result['errors'].append('File not found or empty')
                return result
            
            # Run validation
            validation_result = self.validator.validate_and_fix(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                auto_fix=auto_fix,
                comprehensive=True
            )
            
            # Store basic results
            result['errors'] = validation_result.errors
            result['warnings'] = validation_result.warnings
            result['quality_score'] = validation_result.quality_score
            result['statistics'] = validation_result.statistics
            
            # NEW: Extract DETAILED error information
            result['detailed_errors'] = self._extract_detailed_errors(
                df, validation_result, symbol, timeframe
            )
            
            # Determine status
            if validation_result.errors:
                result['status'] = 'errors'
            elif validation_result.warnings:
                result['status'] = 'warnings'
            else:
                result['status'] = 'valid'
            
            # If auto-fix is enabled and data was fixed, save it
            if auto_fix and validation_result.fixed_df is not None:
                if validation_result.corrections:
                    self.data_manager._save_data(validation_result.fixed_df, symbol, timeframe)
                    result['auto_fixed'] = True
                    if verbose:
                        print(f"  Auto-fixed {symbol} {timeframe}")
            
        except Exception as e:
            result['status'] = 'errors'
            result['errors'].append(f"Validation failed: {str(e)}")
            logger.error(f"Error validating {symbol} {timeframe}: {e}")
        
        return result
    
    def _extract_detailed_errors(self, df: pd.DataFrame, 
                                 validation_result: ValidationResult,
                                 symbol: str, timeframe: str) -> List[Dict]:
        """
        Extract EXACT locations of errors in the dataframe
        
        Args:
            df: DataFrame being validated
            validation_result: Validation result object
            symbol: Symbol name
            timeframe: Timeframe
            
        Returns:
            List of detailed error dictionaries with exact locations
        """
        detailed_errors = []
        
        # Check OHLC relationships
        detailed_errors.extend(self._check_ohlc_errors(df))
        
        # Check for duplicates
        detailed_errors.extend(self._check_duplicate_errors(df))
        
        # Check for missing values
        detailed_errors.extend(self._check_missing_value_errors(df))
        
        # Check for outliers
        detailed_errors.extend(self._check_outlier_errors(df))
        
        # Check for timestamp gaps
        detailed_errors.extend(self._check_timestamp_gaps(df, timeframe))
        
        return detailed_errors
    
    def _check_ohlc_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Check for OHLC relationship errors with exact row numbers"""
        errors = []
        
        # High should be >= open, close, low
        high_errors = df[
            (df['high'] < df['open']) | 
            (df['high'] < df['close']) | 
            (df['high'] < df['low'])
        ]
        
        for idx, row in high_errors.iterrows():
            errors.append({
                'type': 'OHLC_VIOLATION',
                'severity': 'ERROR',
                'row_number': df.index.get_loc(idx) + 2,  # +2 for header and 0-index
                'timestamp': str(idx),
                'issue': 'High is less than Open/Close/Low',
                'values': {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                },
                'suggestion': f"High should be max of all values. Expected >= {max(row['open'], row['close'], row['low']):.2f}"
            })
        
        # Low should be <= open, close, high
        low_errors = df[
            (df['low'] > df['open']) | 
            (df['low'] > df['close']) | 
            (df['low'] > df['high'])
        ]
        
        for idx, row in low_errors.iterrows():
            errors.append({
                'type': 'OHLC_VIOLATION',
                'severity': 'ERROR',
                'row_number': df.index.get_loc(idx) + 2,
                'timestamp': str(idx),
                'issue': 'Low is greater than Open/Close/High',
                'values': {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                },
                'suggestion': f"Low should be min of all values. Expected <= {min(row['open'], row['close'], row['high']):.2f}"
            })
        
        return errors
    
    def _check_duplicate_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Check for duplicate timestamps with exact locations"""
        errors = []
        
        duplicates = df.index[df.index.duplicated(keep=False)]
        
        if len(duplicates) > 0:
            # Group duplicates
            dup_groups = {}
            for idx in duplicates:
                if idx not in dup_groups:
                    dup_groups[idx] = []
                dup_groups[idx].append(df.index.get_loc(idx) + 2)
            
            for timestamp, row_numbers in dup_groups.items():
                errors.append({
                    'type': 'DUPLICATE_TIMESTAMP',
                    'severity': 'ERROR',
                    'row_numbers': row_numbers,
                    'timestamp': str(timestamp),
                    'issue': f'Duplicate timestamp found in {len(row_numbers)} rows',
                    'suggestion': f'Remove duplicate rows, keep only one. Rows: {row_numbers}'
                })
        
        return errors
    
    def _check_missing_value_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Check for NaN/missing values with exact locations"""
        errors = []
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            missing_mask = df[col].isna()
            if missing_mask.any():
                missing_indices = df[missing_mask].index
                for idx in missing_indices:
                    errors.append({
                        'type': 'MISSING_VALUE',
                        'severity': 'ERROR',
                        'row_number': df.index.get_loc(idx) + 2,
                        'timestamp': str(idx),
                        'issue': f'Missing value in {col} column',
                        'column': col,
                        'suggestion': 'Fill with interpolated value or remove row'
                    })
        
        return errors
    
    def _check_outlier_errors(self, df: pd.DataFrame) -> List[Dict]:
        """Check for outliers with exact locations"""
        errors = []
        
        # Check for negative values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            negative_mask = df[col] < 0
            if negative_mask.any():
                negative_indices = df[negative_mask].index
                for idx in negative_indices:
                    errors.append({
                        'type': 'NEGATIVE_VALUE',
                        'severity': 'ERROR',
                        'row_number': df.index.get_loc(idx) + 2,
                        'timestamp': str(idx),
                        'issue': f'Negative value in {col} column',
                        'column': col,
                        'value': float(df.loc[idx, col]),
                        'suggestion': 'Remove row or correct value'
                    })
        
        # Check for extreme price changes (>50% in one candle)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.5]
            
            for idx in extreme_changes.index:
                if pd.notna(price_changes[idx]):
                    errors.append({
                        'type': 'EXTREME_PRICE_CHANGE',
                        'severity': 'WARNING',
                        'row_number': df.index.get_loc(idx) + 2,
                        'timestamp': str(idx),
                        'issue': f'Extreme price change: {price_changes[idx]*100:.1f}%',
                        'previous_close': float(df['close'].iloc[df.index.get_loc(idx) - 1]),
                        'current_close': float(df.loc[idx, 'close']),
                        'change_percent': float(price_changes[idx] * 100),
                        'suggestion': 'Verify if this is legitimate or data error'
                    })
        
        return errors
    
    def _check_timestamp_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Check for timestamp gaps with exact locations"""
        errors = []
        
        # Calculate expected time delta
        time_deltas = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '1w': timedelta(weeks=1)
        }
        
        expected_delta = time_deltas.get(timeframe)
        if expected_delta is None:
            return errors
        
        # Check for gaps
        for i in range(1, len(df)):
            actual_delta = df.index[i] - df.index[i-1]
            if actual_delta > expected_delta * 1.5:  # Allow 50% tolerance
                errors.append({
                    'type': 'TIMESTAMP_GAP',
                    'severity': 'WARNING',
                    'row_number': df.index.get_loc(df.index[i]) + 2,
                    'timestamp_before': str(df.index[i-1]),
                    'timestamp_after': str(df.index[i]),
                    'issue': f'Gap of {actual_delta} detected (expected {expected_delta})',
                    'gap_duration': str(actual_delta),
                    'expected_duration': str(expected_delta),
                    'missing_candles': int((actual_delta / expected_delta) - 1),
                    'suggestion': 'Run --update to fill missing data'
                })
        
        return errors
    
    def _print_validation_summary(self, results: Dict) -> None:
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f" Total files checked: {results['total_files']}")
        print(f" Valid files: {results['valid_files']}")
        print(f" Files with warnings: {results['files_with_warnings']}")
        print(f" Files with errors: {results['files_with_errors']}")
        print(f" Missing files: {results['missing_files']}")
        print(f"  Duration: {results['duration']:.2f} seconds")
        
        # Calculate health percentage
        if results['total_files'] > 0:
            health_pct = (results['valid_files'] / results['total_files']) * 100
            print(f"\n Data Health: {health_pct:.1f}%")
        
        print("="*60 + "\n")
    
    def _print_detailed_error_locations(self, results: Dict) -> None:
        """Print DETAILED error information with exact locations"""
        print("\n" + "="*70)
        print(" DETAILED ERROR REPORT - EXACT LOCATIONS FOR MANUAL FIXING")
        print("="*70 + "\n")
        
        error_count = 0
        for result in results['detailed_results']:
            if result['status'] == 'errors' and result.get('detailed_errors'):
                error_count += 1
                
                # Get file path
                file_symbol = result['symbol'].replace('/', '_')
                file_path = self.data_manager.raw_path / result['timeframe'] / f"{file_symbol}_{result['timeframe']}.csv"
                
                print(f"\n{'='*70}")
                print(f"File #{error_count}: {result['symbol']} {result['timeframe']}")
                print(f"{'='*70}")
                print(f" Location: {file_path}")
                print(f" Quality Score: {result['quality_score']:.1f}/100")
                print(f" Total Errors: {len([e for e in result['detailed_errors'] if e['severity'] == 'ERROR'])}")
                print(f"  Total Warnings: {len([e for e in result['detailed_errors'] if e['severity'] == 'WARNING'])}")
                print()
                
                # Group errors by type
                errors_by_type = {}
                for error in result['detailed_errors']:
                    error_type = error['type']
                    if error_type not in errors_by_type:
                        errors_by_type[error_type] = []
                    errors_by_type[error_type].append(error)
                
                # Print each error type
                for error_type, error_list in errors_by_type.items():
                    print(f"\n   {error_type} ({len(error_list)} occurrences)")
                    print("  " + "-"*66)
                    
                    for i, error in enumerate(error_list[:10], 1):  # Show first 10 of each type
                        if error['type'] == 'OHLC_VIOLATION':
                            print(f"\n  [{i}] Row {error['row_number']} | {error['timestamp']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Values: O={error['values']['open']:.2f}, "
                                  f"H={error['values']['high']:.2f}, "
                                  f"L={error['values']['low']:.2f}, "
                                  f"C={error['values']['close']:.2f}")
                            print(f"      Fix: {error['suggestion']}")
                        
                        elif error['type'] == 'DUPLICATE_TIMESTAMP':
                            print(f"\n  [{i}] Rows {error['row_numbers']} | {error['timestamp']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Fix: {error['suggestion']}")
                        
                        elif error['type'] == 'MISSING_VALUE':
                            print(f"\n  [{i}] Row {error['row_number']} | {error['timestamp']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Column: {error['column']}")
                            print(f"      Fix: {error['suggestion']}")
                        
                        elif error['type'] == 'NEGATIVE_VALUE':
                            print(f"\n  [{i}] Row {error['row_number']} | {error['timestamp']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Value: {error['value']}")
                            print(f"      Fix: {error['suggestion']}")
                        
                        elif error['type'] == 'EXTREME_PRICE_CHANGE':
                            print(f"\n  [{i}] Row {error['row_number']} | {error['timestamp']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Previous: ${error['previous_close']:.2f}")
                            print(f"      Current: ${error['current_close']:.2f}")
                            print(f"      Change: {error['change_percent']:.1f}%")
                            print(f"      Fix: {error['suggestion']}")
                        
                        elif error['type'] == 'TIMESTAMP_GAP':
                            print(f"\n  [{i}] After Row {error['row_number']}")
                            print(f"      Issue: {error['issue']}")
                            print(f"      Gap: {error['timestamp_before']} → {error['timestamp_after']}")
                            print(f"      Missing: ~{error['missing_candles']} candles")
                            print(f"      Fix: {error['suggestion']}")
                    
                    if len(error_list) > 10:
                        print(f"\n  ... and {len(error_list) - 10} more {error_type} errors")
                
                print("\n" + "  " + "-"*66)
                print(f"   HOW TO FIX:")
                print(f"     1. Open: {file_path}")
                print(f"     2. Navigate to the row numbers listed above")
                print(f"     3. Fix the specific issues described")
                print(f"     4. Save the file")
                print(f"     5. Re-run: python main.py data --validate --symbols {result['symbol']} --timeframes {result['timeframe']}")
        
        print("\n" + "="*70)
        print(f" SUMMARY: {error_count} files need manual fixes")
        print("="*70 + "\n")
    
    def _save_validation_report(self, results: Dict) -> None:
        """Save validation report with detailed errors to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report with all details
        report_file = self.logs_path / f'validation_report_{timestamp}.json'
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_files': results['total_files'],
                'valid_files': results['valid_files'],
                'files_with_warnings': results['files_with_warnings'],
                'files_with_errors': results['files_with_errors'],
                'missing_files': results['missing_files'],
                'duration_seconds': results['duration']
            },
            'detailed_results': results['detailed_results']
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed validation report saved to: {report_file}")
        
        # Save human-readable text report
        txt_report = self.logs_path / f'validation_report_{timestamp}.txt'
        with open(txt_report, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DATA VALIDATION REPORT - DETAILED ERROR LOCATIONS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Total files checked: {results['total_files']}\n")
            f.write(f"Valid files: {results['valid_files']}\n")
            f.write(f"Files with warnings: {results['files_with_warnings']}\n")
            f.write(f"Files with errors: {results['files_with_errors']}\n")
            f.write(f"Missing files: {results['missing_files']}\n\n")
            
            # Write detailed errors
            if results['files_with_errors'] > 0:
                f.write("\nFILES WITH ERRORS - DETAILED LOCATIONS\n")
                f.write("="*70 + "\n")
                
                for result in results['detailed_results']:
                    if result['status'] == 'errors' and result.get('detailed_errors'):
                        file_symbol = result['symbol'].replace('/', '_')
                        file_path = self.data_manager.raw_path / result['timeframe'] / f"{file_symbol}_{result['timeframe']}.csv"
                        
                        f.write(f"\n{result['symbol']} {result['timeframe']}\n")
                        f.write(f"File: {file_path}\n")
                        f.write(f"Quality Score: {result['quality_score']:.1f}/100\n")
                        f.write("-"*70 + "\n")
                        
                        for error in result['detailed_errors']:
                            f.write(f"\n  Type: {error['type']}\n")
                            f.write(f"  Severity: {error['severity']}\n")
                            if 'row_number' in error:
                                f.write(f"  Row: {error['row_number']}\n")
                            if 'row_numbers' in error:
                                f.write(f"  Rows: {error['row_numbers']}\n")
                            f.write(f"  Issue: {error['issue']}\n")
                            if 'suggestion' in error:
                                f.write(f"  Fix: {error['suggestion']}\n")
                        
                        f.write("\n")
        
        print(f"Text report saved to: {txt_report}\n")


# Convenience function for CLI usage
def run_data_command(command: str, **kwargs) -> Dict:
    """
    Run a data command
    
    Args:
        command: Command to run ('update' or 'validate')
        **kwargs: Additional arguments
        
    Returns:
        Results dictionary
    """
    dc = DataCommands()
    
    if command == 'update':
        return dc.update_all_data(**kwargs)
    elif command == 'validate':
        return dc.validate_all_data(**kwargs)
    else:
        raise ValueError(f"Unknown command: {command}")


# Testing
if __name__ == "__main__":
    print("=== Data Commands Module Test ===\n")
    
    # Initialize
    dc = DataCommands()
    
    # Test validation
    print("Running validation test...\n")
    results = dc.validate_all_data(
        symbols=['BTC/USDT', 'ETH/USDT'],
        timeframes=['1h', '1d'],
        verbose=True
    )
    
    print("\nData Commands Module Ready!")