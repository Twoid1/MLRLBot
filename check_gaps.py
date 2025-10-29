#!/usr/bin/env python3
"""
Check for gaps in historical crypto data files
"""

import os
import pandas as pd
from datetime import timedelta

# Configuration
SYMBOLS = [
    'ADA_USDT', 'ALGO_USDT', 'AVAX_USDT', 'BTC_USDT', 'DOGE_USDT',
    'DOT_USDT', 'ETH_USDT', 'LINK_USDT', 'LTC_USDT', 'MATIC_USDT',
    'SHIB_USDT', 'SOL_USDT', 'TRX_USDT', 'UNI_USDT', 'XRP_USDT'
]

TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d', '1w']

TIMEFRAME_DELTAS = {
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    '1d': timedelta(days=1),
    '1w': timedelta(weeks=1)
}

DATA_DIR = 'data/raw'


def check_gaps(filepath, timeframe):
    """Check for gaps in a single file"""
    if not os.path.exists(filepath):
        return {'status': 'missing', 'gaps': []}
    
    try:
        df = pd.read_csv(filepath)
        if len(df) == 0:
            return {'status': 'empty', 'gaps': []}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        expected_delta = TIMEFRAME_DELTAS[timeframe]
        gaps = []
        
        for i in range(len(df) - 1):
            current_time = df['timestamp'].iloc[i]
            next_time = df['timestamp'].iloc[i + 1]
            expected_next = current_time + expected_delta
            
            if next_time > expected_next:
                # Found a gap
                gap_start = current_time
                gap_end = next_time
                missing_periods = int((next_time - expected_next) / expected_delta) + 1
                gaps.append({
                    'from': gap_start,
                    'to': gap_end,
                    'missing': missing_periods
                })
        
        return {
            'status': 'ok' if len(gaps) == 0 else 'gaps',
            'total_rows': len(df),
            'start_date': df['timestamp'].iloc[0],
            'end_date': df['timestamp'].iloc[-1],
            'gaps': gaps
        }
        
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'gaps': []}


def main():
    """Check all files for gaps"""
    import sys
    brief = '--brief' in sys.argv or '-b' in sys.argv
    
    print("=" * 80)
    print("CHECKING FOR DATA GAPS")
    print("=" * 80)
    
    all_results = {}
    total_files = 0
    missing_files = 0
    files_with_gaps = 0
    total_gaps = 0
    
    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            filepath = os.path.join(DATA_DIR, timeframe, f'{symbol}_{timeframe}.csv')
            result = check_gaps(filepath, timeframe)
            
            total_files += 1
            
            if result['status'] == 'missing':
                missing_files += 1
            elif result['status'] == 'gaps':
                files_with_gaps += 1
                total_gaps += len(result['gaps'])
            
            if symbol not in all_results:
                all_results[symbol] = {}
            all_results[symbol][timeframe] = result
    
    # Print summary
    print(f"\nTotal files checked: {total_files}")
    print(f"Missing files: {missing_files}")
    print(f"Files with gaps: {files_with_gaps}")
    print(f"Total gaps found: {total_gaps}")
    
    if brief:
        # Brief mode - just show summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        if missing_files == 0 and files_with_gaps == 0:
            print(" ALL FILES ARE COMPLETE - NO GAPS FOUND!")
        else:
            print(f" Issues found:")
            if missing_files > 0:
                print(f"  - {missing_files} files are missing")
            if files_with_gaps > 0:
                print(f"  - {files_with_gaps} files have gaps ({total_gaps} total gaps)")
            print(f"\nRun with no arguments for detailed view")
            print(f"Run update_kraken_data.py to fix missing data and fill gaps")
        
        print("=" * 80)
        return
    
    # Print details for files with issues
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for symbol in SYMBOLS:
        has_issues = False
        symbol_output = []
        
        for timeframe in TIMEFRAMES:
            result = all_results[symbol][timeframe]
            
            if result['status'] == 'missing':
                has_issues = True
                symbol_output.append(f"  {timeframe:6}  FILE MISSING")
            
            elif result['status'] == 'empty':
                has_issues = True
                symbol_output.append(f"  {timeframe:6}  FILE EMPTY")
            
            elif result['status'] == 'error':
                has_issues = True
                symbol_output.append(f"  {timeframe:6}  ERROR: {result['error']}")
            
            elif result['status'] == 'gaps':
                has_issues = True
                gap_count = len(result['gaps'])
                total_missing = sum(g['missing'] for g in result['gaps'])
                symbol_output.append(f"  {timeframe:6}  {gap_count} gaps ({total_missing} missing periods)")
                
                # Show gap details
                for gap in result['gaps']:
                    symbol_output.append(f"           Gap: {gap['from']} -> {gap['to']} ({gap['missing']} missing)")
            
            else:
                # No issues
                symbol_output.append(f"  {timeframe:6}  OK ({result['total_rows']} rows, {result['start_date']} to {result['end_date']})")
        
        # Print symbol section
        if has_issues:
            print(f"\n{symbol}")
            print("-" * 80)
            for line in symbol_output:
                print(line)
    
    # Print files without issues
    print("\n" + "=" * 80)
    print("FILES WITHOUT ISSUES")
    print("=" * 80)
    
    clean_count = 0
    for symbol in SYMBOLS:
        all_clean = True
        for timeframe in TIMEFRAMES:
            if all_results[symbol][timeframe]['status'] != 'ok':
                all_clean = False
                break
        
        if all_clean:
            print(f" {symbol} - All timeframes clean")
            clean_count += 1
    
    if clean_count == 0:
        print("None - all symbols have some issues")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if missing_files == 0 and files_with_gaps == 0:
        print(" ALL FILES ARE COMPLETE - NO GAPS FOUND!")
    else:
        print(f" Issues found:")
        if missing_files > 0:
            print(f"  - {missing_files} files are missing")
        if files_with_gaps > 0:
            print(f"  - {files_with_gaps} files have gaps ({total_gaps} total gaps)")
        print(f"\nRun update_kraken_data.py to fix missing data and fill gaps")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
