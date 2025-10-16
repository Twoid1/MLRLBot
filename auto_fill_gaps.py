#!/usr/bin/env python3
"""
synthetic_gap_filler.py - Fill Data Gaps with Synthetic Data

Fills missing OHLCV data gaps using intelligent interpolation when
Kraken doesn't have the historical data available.

Filling Methods:
1. LINEAR: Linear interpolation (smooth transition)
2. FORWARD: Carry forward last known values
3. PATTERN: Copy pattern from previous similar period
4. SMART: Adaptive method based on gap size

Usage:
    python synthetic_gap_filler.py                    # Use SMART method
    python synthetic_gap_filler.py --method linear    # Use linear interpolation
    python synthetic_gap_filler.py --dry-run          # Preview without saving
    python synthetic_gap_filler.py --mark-synthetic   # Add marker column
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timeframe configurations
TIMEFRAME_CONFIGS = {
    '1m': {'interval': 1, 'max_gap_candles': 60},      # Max 1 hour gap
    '5m': {'interval': 5, 'max_gap_candles': 72},      # Max 5 hours gap
    '15m': {'interval': 15, 'max_gap_candles': 48},    # Max 12 hours gap
    '30m': {'interval': 30, 'max_gap_candles': 48},    # Max 24 hours gap
    '1h': {'interval': 60, 'max_gap_candles': 72},     # Max 3 days gap
    '4h': {'interval': 240, 'max_gap_candles': 42},    # Max 7 days gap
    '1d': {'interval': 1440, 'max_gap_candles': 30},   # Max 30 days gap
}


class SyntheticGapFiller:
    """Fill data gaps with synthetic OHLCV data"""
    
    def __init__(self, 
                 data_dir: str = 'data/raw',
                 method: str = 'smart',
                 dry_run: bool = False,
                 mark_synthetic: bool = True):
        """
        Initialize gap filler
        
        Args:
            data_dir: Directory containing CSV files
            method: Filling method (smart, linear, forward, pattern)
            dry_run: Preview only, don't save changes
            mark_synthetic: Add 'is_synthetic' column to mark filled data
        """
        self.data_dir = Path(data_dir)
        self.method = method.lower()
        self.dry_run = dry_run
        self.mark_synthetic = mark_synthetic
        
        self.stats = {
            'files_processed': 0,
            'gaps_found': 0,
            'gaps_filled': 0,
            'candles_added': 0,
            'skipped_large_gaps': 0
        }
        
        # Validate method
        valid_methods = ['smart', 'linear', 'forward', 'pattern']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from: {valid_methods}")
    
    def process_all(self, filter_symbols: Optional[List[str]] = None):
        """Process all CSV files and fill gaps"""
        
        print("\n" + "=" * 80)
        print(" SYNTHETIC GAP FILLER")
        print("=" * 80)
        print(f"Method: {self.method.upper()}")
        print(f"Mark synthetic data: {self.mark_synthetic}")
        
        if self.dry_run:
            print("Mode: DRY RUN (no changes will be saved)")
        else:
            print("Mode: LIVE (files will be modified)")
        print()
        
        # Find CSV files
        csv_files = self._find_csv_files()
        
        if not csv_files:
            print(f" No CSV files found in {self.data_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files\n")
        
        # Process each file
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] {csv_file.name}")
            print("-" * 80)
            
            symbol, timeframe = self._parse_filename(csv_file)
            
            if not symbol or not timeframe:
                print(" Skipping - invalid filename format")
                continue
            
            if filter_symbols and symbol not in filter_symbols:
                print(" Skipping - not in filter list")
                continue
            
            self.stats['files_processed'] += 1
            self._process_file(csv_file, symbol, timeframe)
            print()
        
        self._print_summary()
    
    def _find_csv_files(self) -> List[Path]:
        """Find all CSV files"""
        csv_files = []
        
        # Root level
        csv_files.extend(list(self.data_dir.glob('*.csv')))
        
        # Subdirectories
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(list(subdir.glob('*.csv')))
        
        return sorted(csv_files)
    
    def _parse_filename(self, csv_file: Path) -> tuple:
        """Parse filename to extract symbol and timeframe"""
        parts = csv_file.stem.split('_')
        
        if len(parts) < 2:
            return None, None
        
        if len(parts) >= 3:
            symbol = f"{parts[0]}/{parts[1]}"
            timeframe = parts[2]
        else:
            symbol = f"{parts[0]}/{parts[1]}"
            timeframe = csv_file.parent.name if csv_file.parent.name != 'raw' else '1h'
        
        return symbol, timeframe
    
    def _process_file(self, csv_file: Path, symbol: str, timeframe: str):
        """Process a single file"""
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            
            if len(df) < 2:
                print(" File too small")
                return
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            original_count = len(df)
            print(f" Loaded {original_count} candles")
            print(f"   Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Detect gaps
            gaps = self._detect_gaps(df, timeframe)
            
            if not gaps:
                print(" No gaps found")
                return
            
            total_missing = sum(g['count'] for g in gaps)
            self.stats['gaps_found'] += len(gaps)
            
            print(f"  Found {len(gaps)} gaps ({total_missing} missing candles)")
            
            # Show gap details
            for i, gap in enumerate(gaps[:5], 1):  # Show first 5
                print(f"   Gap {i}: {gap['start']} -> {gap['end']} ({gap['count']} candles)")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps) - 5} more gaps")
            
            # Fill gaps
            df_filled = self._fill_all_gaps(df, gaps, timeframe)
            
            if df_filled is not None:
                added = len(df_filled) - original_count
                self.stats['candles_added'] += added
                self.stats['gaps_filled'] += len(gaps)
                
                if not self.dry_run:
                    df_filled.to_csv(csv_file, index=False)
                    print(f" Saved {len(df_filled)} candles (added {added})")
                else:
                    print(f" Would add {added} candles (dry run)")
            
        except Exception as e:
            print(f" Error: {e}")
            logger.exception("Error processing file")
    
    def _detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict]:
        """Detect gaps in the dataframe"""
        
        config = TIMEFRAME_CONFIGS.get(timeframe)
        if not config:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return []
        
        expected_delta = timedelta(minutes=config['interval'])
        max_gap_candles = config['max_gap_candles']
        
        gaps = []
        
        for i in range(len(df) - 1):
            current_time = df.loc[i, 'timestamp']
            next_time = df.loc[i + 1, 'timestamp']
            actual_delta = next_time - current_time
            
            if actual_delta > expected_delta * 1.5:
                missing_periods = int(actual_delta / expected_delta) - 1
                
                # Check if gap is too large
                if missing_periods > max_gap_candles:
                    logger.warning(f"Gap too large ({missing_periods} candles), skipping")
                    self.stats['skipped_large_gaps'] += 1
                    continue
                
                gaps.append({
                    'start': current_time,
                    'end': next_time,
                    'count': missing_periods,
                    'start_idx': i
                })
        
        return gaps
    
    def _fill_all_gaps(self, df: pd.DataFrame, gaps: List[Dict], timeframe: str) -> pd.DataFrame:
        """Fill all gaps in the dataframe"""
        
        # Add synthetic marker column if requested
        if self.mark_synthetic and 'is_synthetic' not in df.columns:
            df['is_synthetic'] = False
        
        all_synthetic_rows = []
        
        for gap in gaps:
            synthetic_data = self._fill_single_gap(df, gap, timeframe)
            if synthetic_data is not None:
                all_synthetic_rows.append(synthetic_data)
        
        if all_synthetic_rows:
            # Combine original + synthetic data
            synthetic_df = pd.concat(all_synthetic_rows, ignore_index=True)
            df_combined = pd.concat([df, synthetic_df], ignore_index=True)
            
            # Sort and clean
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='first')
            
            return df_combined
        
        return df
    
    def _fill_single_gap(self, df: pd.DataFrame, gap: Dict, timeframe: str) -> Optional[pd.DataFrame]:
        """Fill a single gap using selected method"""
        
        if self.method == 'smart':
            # Choose method based on gap size
            if gap['count'] <= 5:
                return self._fill_linear(df, gap, timeframe)
            elif gap['count'] <= 20:
                return self._fill_pattern(df, gap, timeframe)
            else:
                return self._fill_forward(df, gap, timeframe)
        
        elif self.method == 'linear':
            return self._fill_linear(df, gap, timeframe)
        
        elif self.method == 'forward':
            return self._fill_forward(df, gap, timeframe)
        
        elif self.method == 'pattern':
            return self._fill_pattern(df, gap, timeframe)
    
    def _fill_linear(self, df: pd.DataFrame, gap: Dict, timeframe: str) -> pd.DataFrame:
        """Fill gap with linear interpolation"""
        
        config = TIMEFRAME_CONFIGS[timeframe]
        interval = timedelta(minutes=config['interval'])
        
        # Get boundary values
        start_row = df[df['timestamp'] == gap['start']].iloc[0]
        end_row = df[df['timestamp'] == gap['end']].iloc[0]
        
        # Generate timestamps
        timestamps = []
        current = gap['start'] + interval
        while current < gap['end']:
            timestamps.append(current)
            current += interval
        
        if not timestamps:
            return None
        
        # Linear interpolation for OHLC
        synthetic_rows = []
        n = len(timestamps)
        
        for i, ts in enumerate(timestamps):
            ratio = (i + 1) / (n + 1)
            
            # Interpolate close price
            close = start_row['close'] + (end_row['close'] - start_row['close']) * ratio
            
            # Generate OHLC around close with small variance
            variance = abs(close * 0.002)  # 0.2% variance
            open_price = close + np.random.uniform(-variance, variance)
            high = max(open_price, close) + np.random.uniform(0, variance)
            low = min(open_price, close) - np.random.uniform(0, variance)
            
            # Interpolate volume
            volume = start_row['volume'] + (end_row['volume'] - start_row['volume']) * ratio
            volume = max(volume * np.random.uniform(0.8, 1.2), 0)  # Add variance
            
            row = {
                'timestamp': ts,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': round(volume, 2)
            }
            
            if self.mark_synthetic:
                row['is_synthetic'] = True
            
            synthetic_rows.append(row)
        
        return pd.DataFrame(synthetic_rows)
    
    def _fill_forward(self, df: pd.DataFrame, gap: Dict, timeframe: str) -> pd.DataFrame:
        """Fill gap by carrying forward last known values"""
        
        config = TIMEFRAME_CONFIGS[timeframe]
        interval = timedelta(minutes=config['interval'])
        
        # Get last known values
        last_row = df[df['timestamp'] == gap['start']].iloc[0]
        
        # Generate timestamps
        timestamps = []
        current = gap['start'] + interval
        while current < gap['end']:
            timestamps.append(current)
            current += interval
        
        if not timestamps:
            return None
        
        # Create synthetic rows
        synthetic_rows = []
        for ts in timestamps:
            # Use last known OHLC with tiny random variations
            variance = abs(last_row['close'] * 0.001)  # 0.1% variance
            
            row = {
                'timestamp': ts,
                'open': round(last_row['close'] + np.random.uniform(-variance, variance), 2),
                'high': round(last_row['close'] + np.random.uniform(0, variance * 2), 2),
                'low': round(last_row['close'] - np.random.uniform(0, variance * 2), 2),
                'close': round(last_row['close'] + np.random.uniform(-variance, variance), 2),
                'volume': round(last_row['volume'] * np.random.uniform(0.9, 1.1), 2)
            }
            
            if self.mark_synthetic:
                row['is_synthetic'] = True
            
            synthetic_rows.append(row)
        
        return pd.DataFrame(synthetic_rows)
    
    def _fill_pattern(self, df: pd.DataFrame, gap: Dict, timeframe: str) -> pd.DataFrame:
        """Fill gap by copying pattern from previous period"""
        
        config = TIMEFRAME_CONFIGS[timeframe]
        interval = timedelta(minutes=config['interval'])
        
        # Generate timestamps
        timestamps = []
        current = gap['start'] + interval
        while current < gap['end']:
            timestamps.append(current)
            current += interval
        
        if not timestamps:
            return None
        
        gap_size = len(timestamps)
        
        # Get previous pattern (same duration before gap)
        start_idx = gap['start_idx']
        pattern_start_idx = max(0, start_idx - gap_size)
        pattern = df.iloc[pattern_start_idx:start_idx]
        
        if len(pattern) < gap_size:
            # Not enough history, use forward fill
            return self._fill_forward(df, gap, timeframe)
        
        # Get the last known price to scale pattern
        last_close = df.iloc[start_idx]['close']
        pattern_last_close = pattern.iloc[-1]['close']
        scale_factor = last_close / pattern_last_close if pattern_last_close != 0 else 1.0
        
        # Create synthetic rows by scaling pattern
        synthetic_rows = []
        for i, ts in enumerate(timestamps):
            if i < len(pattern):
                pattern_row = pattern.iloc[i]
                
                row = {
                    'timestamp': ts,
                    'open': round(pattern_row['open'] * scale_factor, 2),
                    'high': round(pattern_row['high'] * scale_factor, 2),
                    'low': round(pattern_row['low'] * scale_factor, 2),
                    'close': round(pattern_row['close'] * scale_factor, 2),
                    'volume': round(pattern_row['volume'] * np.random.uniform(0.8, 1.2), 2)
                }
                
                if self.mark_synthetic:
                    row['is_synthetic'] = True
                
                synthetic_rows.append(row)
        
        return pd.DataFrame(synthetic_rows) if synthetic_rows else None
    
    def _print_summary(self):
        """Print final summary"""
        
        print("\n" + "=" * 80)
        print(" SUMMARY")
        print("=" * 80)
        print(f"Files processed:       {self.stats['files_processed']}")
        print(f"Gaps found:            {self.stats['gaps_found']}")
        print(f"Gaps filled:           {self.stats['gaps_filled']}")
        print(f"Candles added:         {self.stats['candles_added']}")
        
        if self.stats['skipped_large_gaps'] > 0:
            print(f"Large gaps skipped:    {self.stats['skipped_large_gaps']}")
        
        print("=" * 80)
        
        if not self.dry_run and self.stats['gaps_filled'] > 0:
            print("\n All gaps filled with synthetic data!")
            if self.mark_synthetic:
                print(" Synthetic data marked with 'is_synthetic' column")
        elif self.dry_run:
            print("\n Dry run complete - no files were modified")
        else:
            print("\n No gaps needed filling")
        
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fill data gaps with synthetic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Filling Methods:
  linear   - Smooth linear interpolation (best for small gaps)
  forward  - Carry forward last values (conservative)
  pattern  - Copy previous period pattern (good for medium gaps)
  smart    - Adaptive method based on gap size (recommended)

Examples:
  python synthetic_gap_filler.py --dry-run
  python synthetic_gap_filler.py --method linear
  python synthetic_gap_filler.py --symbols BTC/USDT --mark-synthetic
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['smart', 'linear', 'forward', 'pattern'],
        default='smart',
        help='Filling method (default: smart)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without saving changes'
    )
    parser.add_argument(
        '--mark-synthetic',
        action='store_true',
        default=True,
        help='Add is_synthetic column (default: True)'
    )
    parser.add_argument(
        '--no-mark',
        action='store_true',
        help='Do not mark synthetic data'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Only process specific symbols',
        metavar='SYMBOL'
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Data directory (default: data/raw)',
        metavar='DIR'
    )
    
    args = parser.parse_args()
    
    # Handle mark_synthetic flag
    mark_synthetic = args.mark_synthetic and not args.no_mark
    
    try:
        filler = SyntheticGapFiller(
            data_dir=args.data_dir,
            method=args.method,
            dry_run=args.dry_run,
            mark_synthetic=mark_synthetic
        )
        
        filler.process_all(filter_symbols=args.symbols)
        
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Fatal error: {e}")
        logger.exception("Fatal error")
        sys.exit(1)


if __name__ == '__main__':
    main()