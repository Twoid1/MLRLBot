#!/usr/bin/env python3
"""
Simple Kraken Data Updater
Just updates your CSV files with latest data from Kraken
Can fetch historical data from a specified start date
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

# Configuration
SYMBOLS = [
    'BTC_USDT',
    'DOT_USDT', 'ETH_USDT',
    'SOL_USDT'
]

TIMEFRAMES = ['15m', '1h', '4h', '1d']

PAIR_MAPPING = {
    # Kraken uses USD pairs, not USDT
    # Format: Your filename -> Kraken API pair name
    'ADA_USDT': 'ADAUSD',
    'ALGO_USDT': 'ALGOUSD', 
    'AVAX_USDT': 'AVAXUSD',
    'BTC_USDT': 'XXBTZUSD',  # BTC has X prefix
    'DOGE_USDT': 'XDGUSD',   # DOGE uses XDG
    'DOT_USDT': 'DOTUSD',
    'ETH_USDT': 'XETHZUSD',  # ETH has X prefix
    'LINK_USDT': 'LINKUSD',
    'LTC_USDT': 'XLTCZUSD',  # LTC has X prefix
    'MATIC_USDT': 'MATICUSD',
    'SHIB_USDT': 'SHIBUSD',
    'SOL_USDT': 'SOLUSD',
    'TRX_USDT': 'TRXUSD',
    'UNI_USDT': 'UNIUSD',
    'XRP_USDT': 'XXRPZUSD'   # XRP has X prefix
}

TIMEFRAME_MINUTES = {
    '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
}

DATA_DIR = 'data/raw'

# Historical data start date - change this to fetch from earlier
HISTORICAL_START = datetime(2021, 1, 1)  # January 1, 2021


def fetch_kraken_data_trades(pair, interval_minutes, start_date, end_date):
    """
    Fetch data using Trades endpoint and aggregate into OHLCV candles
    More reliable for historical data than OHLC endpoint
    """
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())
    bucket_seconds = interval_minutes * 60
    
    def bucketize(timestamp):
        """Round timestamp down to nearest interval bucket"""
        return (timestamp // bucket_seconds) * bucket_seconds
    
    buckets = {}
    since = str(int((start_ts - 60) * 1_000_000_000))  # Nanoseconds
    
    try:
        while True:
            response = requests.get(
                'https://api.kraken.com/0/public/Trades',
                params={'pair': pair, 'since': since},
                timeout=30
            )
            
            data = response.json()
            
            if data.get('error'):
                error_msg = ', '.join(data['error'])
                print(f"    API Error: {error_msg}")
                return None
            
            result = data.get('result', {})
            
            # Find the trades key (not 'last')
            trades_key = None
            for key in result.keys():
                if key != 'last':
                    trades_key = key
                    break
            
            if not trades_key:
                break
            
            trades = result[trades_key]
            
            if not trades:
                break
            
            # Process trades: [price, volume, time, side, ordertype, misc]
            for trade in trades:
                price = float(trade[0])
                volume = float(trade[1])
                trade_time = float(trade[2])
                ts = int(trade_time)
                
                # Only process trades within our date range
                if ts < start_ts or ts >= end_ts:
                    continue
                
                # Get the interval bucket for this trade
                bucket_start = bucketize(ts)
                
                if bucket_start not in buckets:
                    buckets[bucket_start] = {
                        'open': price,
                        'high': price,
                        'low': price,
                        'close': price,
                        'volume': volume
                    }
                else:
                    bucket = buckets[bucket_start]
                    bucket['high'] = max(bucket['high'], price)
                    bucket['low'] = min(bucket['low'], price)
                    bucket['close'] = price
                    bucket['volume'] += volume
            
            # Get next cursor
            new_since = result.get('last')
            if not new_since or new_since == since:
                break
            
            since = new_since
            
            # Check if we've covered our target range
            if buckets and max(buckets.keys()) >= ((end_ts - 1) // bucket_seconds) * bucket_seconds:
                break
            
            time.sleep(1)  # Rate limit
        
        # Convert buckets to DataFrame
        if not buckets:
            return None
        
        rows = []
        for bucket_start in sorted(buckets.keys()):
            if start_ts <= bucket_start < end_ts:
                bucket = buckets[bucket_start]
                rows.append({
                    'timestamp': pd.Timestamp(bucket_start, unit='s'),
                    'open': bucket['open'],
                    'high': bucket['high'],
                    'low': bucket['low'],
                    'close': bucket['close'],
                    'volume': bucket['volume']
                })
        
        if not rows:
            return None
        
        df = pd.DataFrame(rows)
        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        print(f"    Exception: {e}")
        return None


def fetch_kraken_data(pair, interval, since=None):
    """Fetch OHLC data from Kraken (for updates only, not historical)"""
    params = {'pair': pair, 'interval': interval}
    if since:
        params['since'] = since
    
    try:
        response = requests.get('https://api.kraken.com/0/public/OHLC', params=params, timeout=30)
        data = response.json()
        
        # Check for API errors
        if data.get('error'):
            error_msg = ', '.join(data['error'])
            print(f"    API Error: {error_msg}")
            return None
        
        # Get the result - Kraken may return data under a different key than requested
        result = data.get('result', {})
        if not result or len(result) <= 1:  # Only 'last' field or empty
            print(f"    No data returned for {pair}")
            return None
        
        # Find the actual pair key (Kraken might return it under a different name)
        pair_key = None
        for key in result.keys():
            if key != 'last':
                pair_key = key
                break
        
        if not pair_key:
            print(f"    No pair data found")
            return None
        
        pair_data = result[pair_key]
        
        df = pd.DataFrame(
            pair_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Keep only the columns in your format: timestamp, open, high, low, close, volume
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df
        
    except Exception as e:
        print(f"    Exception: {e}")
        return None


def fetch_all_historical_data(pair, interval, start_date):
    """
    Fetch all historical data from start_date to now using Trades endpoint
    Breaks into chunks to avoid timeouts
    """
    print(f"    Fetching from {start_date.date()} using Trades endpoint...")
    
    all_data = []
    current_start = start_date
    now = datetime.now()
    
    # Fetch in 30-day chunks to avoid timeouts
    chunk_days = 30
    
    while current_start < now:
        chunk_end = min(current_start + timedelta(days=chunk_days), now)
        
        print(f"    Chunk: {current_start.date()} to {chunk_end.date()}")
        
        chunk_data = fetch_kraken_data_trades(pair, interval, current_start, chunk_end)
        
        if chunk_data is not None and len(chunk_data) > 0:
            all_data.append(chunk_data)
            print(f"      Got {len(chunk_data)} rows")
        else:
            print(f"      No data for this chunk")
        
        current_start = chunk_end
        time.sleep(2)  # Pause between chunks
    
    if not all_data:
        return None
    
    # Combine all chunks
    combined = pd.concat(all_data)
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    return combined


def update_file(symbol, timeframe):
    """Update a single CSV file"""
    filepath = os.path.join(DATA_DIR, timeframe, f'{symbol}_{timeframe}.csv')
    kraken_pair = PAIR_MAPPING[symbol]
    interval = TIMEFRAME_MINUTES[timeframe]
    
    # Read existing data
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'])
        existing.set_index('timestamp', inplace=True)
        since = int(existing.index[-1].timestamp())
        print(f"{symbol} {timeframe}: Updating from {existing.index[-1]}")
        
        # Fetch only new data
        new_data = fetch_kraken_data(kraken_pair, interval, since)
        
        if new_data is None:
            print(f"   Failed to fetch data")
            return False
        
        # Combine and save
        new_data = new_data[~new_data.index.isin(existing.index)]
        if len(new_data) == 0:
            print(f"   Already up to date")
            return True
        
        combined = pd.concat([existing, new_data]).sort_index()
        print(f"   Added {len(new_data)} new rows")
        
    else:
        # No existing file - fetch all historical data from HISTORICAL_START
        print(f"{symbol} {timeframe}: Fetching full history from {HISTORICAL_START.date()}")
        
        combined = fetch_all_historical_data(kraken_pair, interval, HISTORICAL_START)
        
        if combined is None or len(combined) == 0:
            print(f"   Failed to fetch historical data")
            return False
        
        print(f"   Saved {len(combined)} rows ({combined.index[0].date()} to {combined.index[-1].date()})")
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    combined.to_csv(filepath)
    
    time.sleep(1)  # Rate limit
    return True


def main():
    """Update all files"""
    print("=" * 60)
    print("Kraken Data Updater")
    print("=" * 60)
    print(f"Historical data start: {HISTORICAL_START.date()}")
    print(f"Note: For new files, will fetch from {HISTORICAL_START.date()} to current")
    print(f"      For existing files, will only fetch missing data")
    print("=" * 60)
    
    total = len(SYMBOLS) * len(TIMEFRAMES)
    success = 0
    
    for i, symbol in enumerate(SYMBOLS, 1):
        for j, timeframe in enumerate(TIMEFRAMES, 1):
            current = (i-1) * len(TIMEFRAMES) + j
            print(f"\n[{current}/{total}] {symbol} {timeframe}")
            if update_file(symbol, timeframe):
                success += 1
    
    print("\n" + "=" * 60)
    print(f"Complete! {success}/{total} successful")
    print("=" * 60)


if __name__ == '__main__':
    main()