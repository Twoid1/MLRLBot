"""
Binance Historical Data Connector
Fetches historical OHLCV data from Binance API for training

Features:
- Clean, gap-free historical data
- Multiple timeframes support
- Rate limiting and retry logic
- Automatic data validation
- CSV storage compatible with existing system
- Symbol mapping for Kraken compatibility

Used for: Training data only (Live trading still uses Kraken)
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BinanceDataConfig:
    """Configuration for Binance data fetching"""
    # Symbols to fetch (Binance format)
    symbols: List[str] = None
    
    # Timeframes (Binance format)
    timeframes: List[str] = None
    
    # Date range
    start_date: str = '2020-01-01'
    end_date: str = None  # None = today
    
    # Storage
    data_path: str = './data/raw/'
    
    # API settings
    base_url: str = 'https://api.binance.com'
    rate_limit_delay: float = 0.5  # Seconds between requests
    max_retries: int = 3
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 
                'ADAUSDT', 'DOTUSDT', 'AVAXUSDT'
            ]
        
        if self.timeframes is None:
            self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        if self.end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')


class BinanceHistoricalData:
    """
    Binance Historical Data Connector
    Fetches clean OHLCV data for training
    """
    
    # Binance to Kraken symbol mapping
    SYMBOL_MAPPING = {
        'BTCUSDT': 'BTC_USDT',
        'ETHUSDT': 'ETH_USDT',
        'SOLUSDT': 'SOL_USDT',
        'ADAUSDT': 'ADA_USDT',
        'DOTUSDT': 'DOT_USDT',
        'AVAXUSDT': 'AVAX_USDT',
        'MATICUSDT': 'MATIC_USDT',
        'LINKUSDT': 'LINK_USDT',
        'UNIUSDT': 'UNI_USDT',
        'XRPUSDT': 'XRP_USDT',
        'BNBUSDT': 'BNB_USDT',
        'DOGEUSDT': 'DOGE_USDT',
        'LTCUSDT': 'LTC_USDT',
        'ATOMUSDT': 'ATOM_USDT',
        'ALGOUSDT': 'ALGO_USDT'
    }
    
    # Binance timeframe format
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w'
    }
    
    def __init__(self, config: Optional[BinanceDataConfig] = None):
        """Initialize Binance data connector"""
        self.config = config or BinanceDataConfig()
        
        # Create data directories
        self.data_path = Path(self.config.data_path)
        for tf in self.config.timeframes:
            (self.data_path / tf).mkdir(parents=True, exist_ok=True)
        
        logger.info("="*80)
        logger.info("BINANCE HISTORICAL DATA CONNECTOR")
        logger.info("="*80)
        logger.info(f"Symbols: {len(self.config.symbols)}")
        logger.info(f"Timeframes: {self.config.timeframes}")
        logger.info(f"Date Range: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"Data Path: {self.data_path}")
        logger.info("="*80)
    
    def fetch_all_data(self, update_existing: bool = False):
        """
        Fetch all historical data for all symbols and timeframes
        
        Args:
            update_existing: If True, update existing files with new data
        """
        logger.info("\n Starting historical data fetch from Binance...")
        
        total_downloads = len(self.config.symbols) * len(self.config.timeframes)
        completed = 0
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                try:
                    logger.info(f"\n[{completed+1}/{total_downloads}] Fetching {symbol} {timeframe}...")
                    
                    # Check if file exists
                    kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
                    file_path = self.data_path / timeframe / f"{kraken_symbol}_{timeframe}.csv"
                    
                    if file_path.exists() and not update_existing:
                        logger.info(f"   File exists, skipping (use update_existing=True to refresh)")
                        completed += 1
                        continue
                    
                    # Fetch data
                    df = self.fetch_klines(symbol, timeframe)
                    
                    if df is not None and len(df) > 0:
                        # Save to CSV
                        self._save_data(df, symbol, timeframe)
                        logger.info(f"   Saved {len(df)} candles")
                    else:
                        logger.warning(f"   No data received")
                    
                    completed += 1
                    
                    # Rate limiting
                    time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"   Error: {e}")
                    completed += 1
        
        logger.info("\n" + "="*80)
        logger.info(f" DOWNLOAD COMPLETE: {completed}/{total_downloads} successful")
        logger.info("="*80)
    
    def fetch_klines(self, symbol: str, timeframe: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Binance
        
        Args:
            symbol: Binance symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '5m', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
        
        # Convert start date to timestamp
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        
        # FIXED: Handle end timestamp correctly
        if end_date == datetime.now().strftime('%Y-%m-%d'):
            # Use current time for today (gets most recent data)
            end_ts = int(datetime.now().timestamp() * 1000)
            logger.debug(f"   Using current timestamp: {datetime.now()}")
        else:
            # Use end of day for past dates
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) - timedelta(seconds=1)
            end_ts = int(end_dt.timestamp() * 1000)
            logger.debug(f"   Using end of day: {end_dt}")
        
        # Binance returns max 1000 candles per request
        all_klines = []
        current_ts = start_ts
        
        interval = self.TIMEFRAME_MAPPING.get(timeframe, timeframe)
        
        while current_ts < end_ts:
            try:
                # Build request
                url = f"{self.config.base_url}/api/v3/klines"
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }
                
                # Make request with retries
                response = self._make_request(url, params)
                
                if response is None:
                    break
                
                klines = response.json()
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                
                # Update timestamp for next batch
                current_ts = klines[-1][0] + 1  # Last timestamp + 1ms
                
                # Rate limiting
                time.sleep(self.config.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching klines: {e}")
                break
        
        if not all_klines:
            return None
        
        # Convert to DataFrame
        df = self._parse_klines(all_klines)
        
        return df
    
    def _make_request(self, url: str, params: Dict) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt+1}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    return None
    
    def _parse_klines(self, klines: List) -> pd.DataFrame:
        """
        Parse Binance klines to DataFrame
        
        Binance kline format:
        [
            Open time,
            Open,
            High,
            Low,
            Close,
            Volume,
            Close time,
            Quote asset volume,
            Number of trades,
            Taker buy base asset volume,
            Taker buy quote asset volume,
            Ignore
        ]
        """
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Select and convert columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set index
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to CSV with Kraken-compatible naming"""
        # Convert to Kraken symbol format
        kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
        
        # Create file path
        file_path = self.data_path / timeframe / f"{kraken_symbol}_{timeframe}.csv"
        
        # Save to CSV
        df.to_csv(file_path)
        
        logger.info(f"   Saved to: {file_path}")
    
    def update_existing_data(self):
        """Update existing data files with new candles"""
        logger.info("\n Updating existing data with new candles...")
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                try:
                    kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
                    file_path = self.data_path / timeframe / f"{kraken_symbol}_{timeframe}.csv"
                    
                    if not file_path.exists():
                        logger.info(f"   {symbol} {timeframe}: File doesn't exist, skipping")
                        continue
                    
                    # Load existing data
                    existing_df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    
                    # Get last timestamp
                    last_timestamp = existing_df.index[-1]
                    start_date = (last_timestamp + timedelta(minutes=1)).strftime('%Y-%m-%d')
                    
                    logger.info(f"   {symbol} {timeframe}: Fetching from {start_date}...")
                    
                    # Fetch new data
                    new_df = self.fetch_klines(symbol, timeframe, start_date=start_date)
                    
                    if new_df is not None and len(new_df) > 0:
                        # Combine and remove duplicates
                        combined_df = pd.concat([existing_df, new_df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        combined_df.sort_index(inplace=True)
                        
                        # Save
                        combined_df.to_csv(file_path)
                        
                        new_candles = len(combined_df) - len(existing_df)
                        logger.info(f"   Added {new_candles} new candles")
                    else:
                        logger.info(f"   Already up to date")
                    
                    time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"   Error updating {symbol} {timeframe}: {e}")
        
        logger.info("\n Update complete")
    
    def validate_data(self):
        """Validate all downloaded data"""
        logger.info("\n Validating data quality...")
        
        issues = []
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
                file_path = self.data_path / timeframe / f"{kraken_symbol}_{timeframe}.csv"
                
                if not file_path.exists():
                    issues.append(f"Missing: {symbol} {timeframe}")
                    continue
                
                try:
                    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    
                    # Check for NaN values
                    nan_count = df.isna().sum().sum()
                    if nan_count > 0:
                        issues.append(f"{symbol} {timeframe}: {nan_count} NaN values")
                    
                    # Check for gaps
                    time_diff = df.index.to_series().diff()
                    expected_diff = self._get_expected_timedelta(timeframe)
                    gaps = (time_diff > expected_diff * 1.5).sum()
                    
                    if gaps > 0:
                        issues.append(f"{symbol} {timeframe}: {gaps} time gaps")
                    
                    # Check data range
                    start = df.index[0].strftime('%Y-%m-%d')
                    end = df.index[-1].strftime('%Y-%m-%d')
                    logger.info(f"   {symbol} {timeframe}: {len(df)} candles ({start} to {end})")
                    
                except Exception as e:
                    issues.append(f"{symbol} {timeframe}: Error - {e}")
        
        if issues:
            logger.warning("\n  Issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("\n All data validated successfully!")
    
    def _get_expected_timedelta(self, timeframe: str) -> timedelta:
        """Get expected time difference between candles"""
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
        return mapping.get(timeframe, timedelta(minutes=5))
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        summary = {}
        
        for symbol in self.config.symbols:
            kraken_symbol = self.SYMBOL_MAPPING.get(symbol, symbol)
            summary[kraken_symbol] = {}
            
            for timeframe in self.config.timeframes:
                file_path = self.data_path / timeframe / f"{kraken_symbol}_{timeframe}.csv"
                
                if file_path.exists():
                    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                    summary[kraken_symbol][timeframe] = {
                        'candles': len(df),
                        'start': df.index[0].strftime('%Y-%m-%d'),
                        'end': df.index[-1].strftime('%Y-%m-%d'),
                        'size_mb': file_path.stat().st_size / 1024 / 1024
                    }
                else:
                    summary[kraken_symbol][timeframe] = None
        
        return summary


# ==================== CLI INTERFACE ====================

def main():
    """Command-line interface for data fetching"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Historical Data Fetcher')
    parser.add_argument('--fetch', action='store_true', help='Fetch all historical data')
    parser.add_argument('--update', action='store_true', help='Update existing data')
    parser.add_argument('--validate', action='store_true', help='Validate data quality')
    parser.add_argument('--summary', action='store_true', help='Show data summary')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD, default: today)')
    parser.add_argument('--symbols', nargs='+', help='Symbols to fetch (default: all)')
    parser.add_argument('--timeframes', nargs='+', help='Timeframes to fetch (default: all)')
    
    args = parser.parse_args()
    
    # Create config
    config = BinanceDataConfig(
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if args.symbols:
        config.symbols = args.symbols
    
    if args.timeframes:
        config.timeframes = args.timeframes
    
    # Initialize connector
    connector = BinanceHistoricalData(config)
    
    # Execute commands
    if args.fetch:
        connector.fetch_all_data()
        connector.validate_data()
    
    elif args.update:
        connector.update_existing_data()
        connector.validate_data()
    
    elif args.validate:
        connector.validate_data()
    
    elif args.summary:
        summary = connector.get_data_summary()
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(json.dumps(summary, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()