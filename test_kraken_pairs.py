#!/usr/bin/env python3
"""
Test which Kraken pairs actually work
Run this BEFORE running the updater
"""

import requests
import time

# All possible pair formats to test
test_pairs = {
    'ADA': ['ADAUSD', 'ADAUSDT', 'ADAEUR'],
    'ALGO': ['ALGOUSD', 'ALGOUSDT', 'ALGOEUR'],
    'AVAX': ['AVAXUSD', 'AVAXUSDT', 'AVAXEUR'],
    'BTC': ['XBTUSD', 'XXBTZUSD', 'BTCUSD'],
    'DOGE': ['DOGEUSD', 'XDGUSD', 'DOGEUSDT'],
    'DOT': ['DOTUSD', 'DOTUSDT', 'DOTEUR'],
    'ETH': ['ETHUSD', 'XETHZUSD', 'ETHUSDT'],
    'LINK': ['LINKUSD', 'LINKUSDT', 'LINKEUR'],
    'LTC': ['LTCUSD', 'XLTCZUSD', 'LTCUSDT'],
    'MATIC': ['MATICUSD', 'MATICUSDT', 'MATICEUR'],
    'SHIB': ['SHIBUSD', 'SHIBUSDT', 'SHIBEUR'],
    'SOL': ['SOLUSD', 'SOLUSDT', 'SOLEUR'],
    'TRX': ['TRXUSD', 'TRXUSDT', 'TRXEUR'],
    'UNI': ['UNIUSD', 'UNIUSDT', 'UNIEUR'],
    'XRP': ['XRPUSD', 'XXRPZUSD', 'XRPUSDT']
}

print("=" * 80)
print("Testing Kraken Pairs")
print("=" * 80)
print("\nThis will test which coin pairs are available on Kraken...")
print("Please wait, testing each pair...\n")

working_pairs = {}
failed_coins = []

for symbol, pairs in test_pairs.items():
    print(f"\nTesting {symbol}...")
    found = False
    
    for pair in pairs:
        try:
            response = requests.get(
                'https://api.kraken.com/0/public/OHLC',
                params={'pair': pair, 'interval': 1440},
                timeout=10
            )
            data = response.json()
            
            # Check if successful
            if not data.get('error'):
                result = data.get('result', {})
                if len(result) > 1:  # More than just 'last'
                    # Get the actual pair name Kraken returns
                    actual_pair = [k for k in result.keys() if k != 'last'][0]
                    print(f"   {pair} works (Kraken returns: {actual_pair})")
                    working_pairs[symbol] = actual_pair
                    found = True
                    break
            
            time.sleep(0.5)  # Rate limit
            
        except Exception as e:
            continue
    
    if not found:
        print(f"   No working pair found for {symbol}")
        failed_coins.append(symbol)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if working_pairs:
    print(f"\n {len(working_pairs)} coins available on Kraken:\n")
    print("PAIR_MAPPING = {")
    for symbol, pair in working_pairs.items():
        print(f"    '{symbol}_USDT': '{pair}',  # Note: Kraken USD, not USDT")
    print("}")
    
    print("\n\nSYMBOLS = [")
    for symbol in working_pairs.keys():
        print(f"    '{symbol}_USDT',")
    print("]")

if failed_coins:
    print(f"\n\n {len(failed_coins)} coins NOT available on Kraken:")
    for symbol in failed_coins:
        print(f"  - {symbol}")
    print("\nYou'll need to remove these from your SYMBOLS list")
    print("or use a different exchange (like Binance) for these coins.")

print("\n" + "=" * 80)
print("\nNext steps:")
print("1. Copy the PAIR_MAPPING and SYMBOLS from above")
print("2. Update your update_kraken_data.py script")
print("3. Run the updater")
print("=" * 80)
