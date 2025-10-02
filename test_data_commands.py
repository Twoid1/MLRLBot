#!/usr/bin/env python3
"""
fill_data_gaps.py - Fixed for Subdirectory Structure

Automatically detects and fills gaps in historical OHLCV data files.
Fixed: Now correctly searches subdirectories (1h/, 4h/, 1d/, etc.)

Usage:
    python fill_data_gaps.py --scan              # Scan for gaps
    python fill_data_gaps.py --fill              # Fill all gaps
    python fill_data_gaps.py --fill --dry-run    # Preview what would be filled
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.kraken_connector import KrakenConnector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Timeframe configurations
TIMEFRAME_CONFIGS = {
    '1m': {'interval': 1, 'kraken_interval': 1},
    '5m': {'interval': 5, 'kraken_interval': 5},
    '15m': {'interval': 15, 'kraken_interval': 15},
    '30m': {'interval': 30, 'kraken_interval': 30},
    '1h': {'interval': 60, 'kraken_interval': 60},
    '4h': {'interval': 240, 'kraken_interval': 240},
    '1d': {'interval': 1440, 'kraken_interval': 1440},
}


class DataGapFiller:
    """Automatically detect and fill gaps in OHLCV data"""
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.kraken = None
        self._initialize_kraken()
        
    def _initialize_kraken(self):
        """Initialize Kraken connector with proper error handling"""
        try:
            self.kraken = KrakenConnector()
            if hasattr(self.kraken, 'api') and self.kraken.api is not None:
                logger.info(" Kraken connector initialized successfully")
            else:
                logger.warning(" Kraken connector initialized but API may not be available")
        except Exception as e:
            logger.error(f" Failed to initialize Kraken connector: {e}")
            self.kraken = None
    
    def detect_gaps(self, 
                    symbols: Optional[List[str]] = None,
                    timeframes: Optional[List[str]] = None) -> Dict:
        """
        Detect gaps in all data files
        
        Args:
            symbols: List of symbols to check (None = all)
            timeframes: List of timeframes to check (None = all)
            
        Returns:
            Dictionary of gaps found
        """
        logger.info("=" * 70)
        logger.info(" SCANNING FOR DATA GAPS")
        logger.info("=" * 70)
        
        all_gaps = {}
        files_checked = 0
        gaps_found = 0
        
        # Search for CSV files in:
        # 1. Root directory (data/raw/*.csv)
        # 2. Subdirectories (data/raw/1h/*.csv, data/raw/1d/*.csv, etc.)
        csv_files = []
        
        # Root level
        csv_files.extend(list(self.data_dir.glob('*.csv')))
        
        # Subdirectories (one level deep)
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                csv_files.extend(list(subdir.glob('*.csv')))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_dir} or its subdirectories")
            logger.info(f"Expected structure: {self.data_dir}/1h/BTC_USDT_1h.csv")
            return all_gaps
        
        logger.info(f"Found {len(csv_files)} CSV files to check")
        
        for csv_file in csv_files:
            # Parse filename: BTC_USDT_1h.csv or BTC/USDT format
            parts = csv_file.stem.split('_')
            if len(parts) < 2:
                logger.warning(f"Skipping file with unexpected format: {csv_file.name}")
                continue
            
            # Handle different naming conventions
            if len(parts) >= 3:
                # Format: BTC_USDT_1h.csv
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = parts[2]
            else:
                # Format: BTC_USDT.csv (try to infer timeframe from parent dir)
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = csv_file.parent.name if csv_file.parent.name != 'raw' else '1h'
            
            # Filter by symbols/timeframes if specified
            if symbols and symbol not in symbols:
                continue
            if timeframes and timeframe not in timeframes:
                continue
            
            files_checked += 1
            
            # Check for gaps
            gaps = self._find_gaps_in_file(csv_file, symbol, timeframe)
            
            if gaps:
                all_gaps[csv_file] = gaps
                gaps_found += len(gaps)
                total_missing = sum(g['count'] for g in gaps)
                logger.info(f" {csv_file.name}: {len(gaps)} gaps ({total_missing} missing candles)")
            else:
                logger.info(f" {csv_file.name}: No gaps")
        
        logger.info("=" * 70)
        logger.info(f" SCAN SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Files checked: {files_checked}")
        logger.info(f"Files with gaps: {len(all_gaps)}")
        logger.info(f"Total gaps found: {gaps_found}")
        if all_gaps:
            total_missing = sum(sum(g['count'] for g in gaps) for gaps in all_gaps.values())
            logger.info(f"Total missing candles: {total_missing}")
        logger.info("=" * 70)
        
        return all_gaps
    
    def _find_gaps_in_file(self, 
                          file_path: Path, 
                          symbol: str, 
                          timeframe: str) -> List[Dict]:
        """Find gaps in a single file"""
        try:
            # Load data
            df = pd.read_csv(file_path)
            if len(df) < 2:
                return []
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Get expected interval
            config = TIMEFRAME_CONFIGS.get(timeframe)
            if not config:
                logger.warning(f"Unknown timeframe: {timeframe}")
                return []
            
            expected_delta = timedelta(minutes=config['interval'])
            
            # Find gaps
            gaps = []
            for i in range(len(df) - 1):
                current_time = df.loc[i, 'timestamp']
                next_time = df.loc[i + 1, 'timestamp']
                
                actual_delta = next_time - current_time
                
                # Gap detected if actual > 1.5x expected
                if actual_delta > expected_delta * 1.5:
                    missing_periods = int(actual_delta / expected_delta) - 1
                    
                    gaps.append({
                        'start': current_time,
                        'end': next_time,
                        'count': missing_periods,
                        'duration': str(actual_delta)
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")
            return []
    
    def fill_gaps(self, 
                  gaps: Dict, 
                  dry_run: bool = False) -> Dict:
        """
        Fill detected gaps by fetching data from Kraken
        
        Args:
            gaps: Dictionary of gaps from detect_gaps()
            dry_run: If True, don't actually fill gaps
            
        Returns:
            Dictionary of fill results
        """
        if not self.kraken:
            logger.error(" Cannot fill gaps: Kraken connector not initialized")
            return {}
        
        logger.info("=" * 70)
        logger.info(" FILLING DATA GAPS")
        logger.info("=" * 70)
        
        if dry_run:
            logger.info(" DRY RUN MODE - No files will be modified")
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        for file_path, file_gaps in gaps.items():
            if not file_gaps:
                continue
            
            # Parse filename
            parts = file_path.stem.split('_')
            if len(parts) < 2:
                continue
            
            # Handle different naming conventions
            if len(parts) >= 3:
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = parts[2]
            else:
                symbol = f"{parts[0]}/{parts[1]}"
                timeframe = file_path.parent.name if file_path.parent.name != 'raw' else '1h'
            
            logger.info(f"\n Processing: {file_path}")
            logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}")
            logger.info(f"   Gaps to fill: {len(file_gaps)}")
            
            try:
                if not dry_run:
                    filled_count = self._fill_file_gaps(
                        file_path, 
                        symbol, 
                        timeframe, 
                        file_gaps
                    )
                    
                    results['success'].append({
                        'file': file_path.name,
                        'gaps_filled': len(file_gaps),
                        'candles_added': filled_count
                    })
                    
                    logger.info(f"    Filled {len(file_gaps)} gaps ({filled_count} candles)")
                else:
                    total_candles = sum(g['count'] for g in file_gaps)
                    logger.info(f"   Would fill {len(file_gaps)} gaps ({total_candles} candles)")
                    results['skipped'].append(file_path.name)
                
            except Exception as e:
                logger.error(f"    Failed: {e}")
                results['failed'].append({
                    'file': file_path.name,
                    'error': str(e)
                })
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info(" FILL SUMMARY")
        logger.info("=" * 70)
        logger.info(f" Success: {len(results['success'])}")
        logger.info(f" Failed: {len(results['failed'])}")
        if dry_run:
            logger.info(f" Skipped (dry run): {len(results['skipped'])}")
        logger.info("=" * 70)
        
        return results
    
    def _fill_file_gaps(self, 
                       file_path: Path, 
                       symbol: str, 
                       timeframe: str, 
                       gaps: List[Dict]) -> int:
        """Fill gaps in a single file"""
        # Load existing data
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get Kraken pair name
        kraken_pair = self._get_kraken_pair(symbol)
        kraken_interval = TIMEFRAME_CONFIGS[timeframe]['kraken_interval']
        
        total_added = 0
        
        for gap in gaps:
            try:
                # Convert timestamps to unix
                since = int(gap['start'].timestamp())
                until = int(gap['end'].timestamp())
                
                logger.info(f"   Fetching: {gap['start']} to {gap['end']}")
                
                # Fetch data from Kraken
                new_data = self._fetch_gap_data(
                    kraken_pair, 
                    kraken_interval, 
                    since,
                    until
                )
                
                if new_data is not None and len(new_data) > 0:
                    # Append new data
                    df = pd.concat([df, new_data], ignore_index=True)
                    total_added += len(new_data)
                    
                    # Rate limiting
                    time.sleep(0.5)
                else:
                    logger.warning(f"   No data returned for gap")
                    
            except Exception as e:
                logger.error(f"   Error fetching gap data: {e}")
                continue
        
        if total_added > 0:
            # Remove duplicates, sort, and save
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.to_csv(file_path, index=False)
        
        return total_added
    
    def _fetch_gap_data(self, 
                       pair: str, 
                       interval: int, 
                       since: int,
                       until: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Kraken for a specific time range"""
        try:
            # Use the kraken connector's fetch method
            df = self.kraken.fetch_ohlc(
                pair=pair,
                interval=interval,
                since=since
            )
            
            if df is None or len(df) == 0:
                return None
            
            # Filter to our time range
            df = df[df['timestamp'] <= pd.Timestamp.fromtimestamp(until)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching gap data: {e}")
            return None
    
    def _get_kraken_pair(self, symbol: str) -> str:
        """Convert symbol to Kraken pair format"""
        symbol_map = {
            'BTC/USDT': 'XBTUSDT',
            'ETH/USDT': 'ETHUSDT',
            'SOL/USDT': 'SOLUSDT',
            'ADA/USDT': 'ADAUSDT',
            'DOT/USDT': 'DOTUSDT',
            'MATIC/USDT': 'MATICUSDT',
            'AVAX/USDT': 'AVAXUSDT',
            'LINK/USDT': 'LINKUSDT',
            'ATOM/USDT': 'ATOMUSDT',
            'UNI/USDT': 'UNIUSDT',
            'XRP/USDT': 'XRPUSDT',
            'LTC/USDT': 'LTCUSDT',
            'DOGE/USDT': 'DOGEUSDT',
            'SHIB/USDT': 'SHIBUSDT',
            'TRX/USDT': 'TRXUSDT',
        }
        return symbol_map.get(symbol, symbol.replace('/', ''))


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fill gaps in historical data')
    parser.add_argument('--scan', action='store_true', help='Scan for gaps only')
    parser.add_argument('--fill', action='store_true', help='Fill detected gaps')
    parser.add_argument('--dry-run', action='store_true', help='Preview fills without making changes')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--timeframes', nargs='+', help='Specific timeframes to process')
    
    args = parser.parse_args()
    
    # Initialize filler
    filler = DataGapFiller()
    
    # Detect gaps
    gaps = filler.detect_gaps(
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    if not gaps:
        logger.info("\n No gaps found! All data is continuous.")
        return
    
    # If scan only, we're done
    if args.scan and not args.fill:
        return
    
    # Ask for confirmation if not dry run
    if args.fill and not args.dry_run:
        print(f"\nFound {len(gaps)} files with gaps.")
        response = input("Do you want to fill these gaps? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            logger.info("Gap filling cancelled by user")
            return
    
    # Fill gaps
    if args.fill:
        results = filler.fill_gaps(gaps, dry_run=args.dry_run)
        
        if not args.dry_run and results['success']:
            logger.info("\n TIP: Run with --scan to verify all gaps are filled")


if __name__ == '__main__':
    main()