"""
test_with_real_data.py

Test Kraken Connector with your actual historical data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kraken_connector import MultiAssetKrakenConnector
from datetime import datetime
import pandas as pd
from colorama import init, Fore, Style

init()

def test_with_real_data():
    """Test with your actual data files"""
    
    print(f"\n{Fore.CYAN} TESTING WITH REAL DATA{Style.RESET_ALL}")
    print("="*60)
    
    # Initialize connector with your real data path
    connector = MultiAssetKrakenConnector(
        mode='paper',
        data_path='./data/raw/',  # Your actual data path
        update_existing_data=False  # Don't update during test
    )
    
    print("\n Checking your 15 crypto pairs...")
    
    results = {}
    for pair in connector.ALL_PAIRS:
        print(f"\n{Fore.YELLOW}{pair}:{Style.RESET_ALL}")
        
        for timeframe in ['1h', '4h', '1d']:
            df = connector.load_existing_data(pair, timeframe)
            
            if not df.empty:
                last_date = df.index[-1]
                days_old = (datetime.now() - last_date).days
                
                status = "Y" if days_old < 7 else "N"
                print(f"  {timeframe}: {status} {len(df)} rows, "
                      f"last: {last_date.strftime('%Y-%m-%d')} "
                      f"({days_old} days old)")
                
                results[f"{pair}_{timeframe}"] = {
                    'rows': len(df),
                    'last_date': last_date,
                    'days_old': days_old
                }
            else:
                print(f"  {timeframe}:  No data found")
                results[f"{pair}_{timeframe}"] = None
    
    print(f"\n Testing gap detection...")
    
    status_df = connector.check_data_status()
    gaps = status_df[status_df['status'] == 'NEEDS_UPDATE']
    
    if not gaps.empty:
        print(f"\n{Fore.YELLOW}Found {len(gaps)} gaps that need updating:{Style.RESET_ALL}")
        for _, gap in gaps.iterrows():
            print(f"  • {gap['symbol']} {gap['timeframe']}: "
                  f"{gap['candles_missing']} candles missing "
                  f"({gap['gap_hours']:.1f} hours)")
    else:
        print(f"{Fore.GREEN}All data is up to date!{Style.RESET_ALL}")
    
    print(f"\n Testing paper trading with real prices...")
    
    # Get current prices
    prices = connector.get_all_current_prices()
    
    # Test a trade with real price
    btc_price = connector.get_current_price('BTC_USDT')
    if btc_price:
        print(f"\nCurrent BTC price: ${btc_price:,.2f}")
        
        # Simulate a small purchase
        from src.data.kraken_connector import KrakenOrder
        order = KrakenOrder(
            pair='BTC_USDT',
            type='buy',
            ordertype='market',
            volume=0.001
        )
        
        result = connector.place_order(order)
        if result['success']:
            print(f"{Fore.GREEN} Test order successful!{Style.RESET_ALL}")
            print(f"   Order ID: {result['order_id']}")
            print(f"   Execution price: ${result['execution_price']:,.2f}")
            print(f"   Cost: ${result['execution_price'] * 0.001:,.2f}")
    
    print(f"\n Summary:")
    
    # Count data availability
    total_files = len(connector.ALL_PAIRS) * 3  # 3 timeframes
    available_files = sum(1 for v in results.values() if v is not None)
    current_files = sum(1 for v in results.values() 
                       if v and v['days_old'] < 7)
    
    print(f"\n  Total expected files: {total_files}")
    print(f"  Available files: {available_files} "
          f"({available_files/total_files*100:.1f}%)")
    print(f"  Current files (<7 days): {current_files} "
          f"({current_files/total_files*100:.1f}%)")
    
    if available_files == total_files:
        print(f"\n{Fore.GREEN} All data files present!{Style.RESET_ALL}")
    else:
        missing = total_files - available_files
        print(f"\n{Fore.YELLOW} {missing} data files missing{Style.RESET_ALL}")
    
    if current_files < available_files:
        print(f"{Fore.YELLOW} Run connector.fill_data_gaps() to update old data{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Test complete!{Style.RESET_ALL}")

if __name__ == "__main__":
    test_with_real_data()