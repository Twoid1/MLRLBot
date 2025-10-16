#!/usr/bin/env python3
"""
Diagnose Data Loading Issues
Helps identify why data files aren't loading in the training system
"""

import pandas as pd
from pathlib import Path
import sys

print("=" * 80)
print("DATA LOADING DIAGNOSTICS")
print("=" * 80)
print()

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Configuration
assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'ADA_USDT', 'DOT_USDT']
timeframes = ['1h', '4h', '1d']
data_dir = Path('data/raw')

print("[1/4] Checking directory structure...")
print("-" * 80)

if not data_dir.exists():
    print(f" Data directory doesn't exist: {data_dir}")
    print("\nCreate it with:")
    print(f"  mkdir -p {data_dir}")
else:
    print(f" Data directory exists: {data_dir}")

for tf in timeframes:
    tf_dir = data_dir / tf
    if tf_dir.exists():
        print(f" {tf}/ directory exists")
    else:
        print(f" {tf}/ directory missing")

print()
print("[2/4] Checking for data files...")
print("-" * 80)

files_found = 0
files_missing = 0

for asset in assets:
    print(f"\n{asset}:")
    for tf in timeframes:
        # Try different file patterns
        patterns = [
            data_dir / tf / f"{asset}_{tf}.csv",
            data_dir / f"{asset}_{tf}.csv",
            data_dir / tf / f"{asset.replace('_', '')}_{tf}.csv",
        ]
        
        found = False
        for filepath in patterns:
            if filepath.exists():
                print(f"   {tf}: {filepath}")
                files_found += 1
                found = True
                break
        
        if not found:
            print(f"   {tf}: No file found")
            print(f"     Expected: {data_dir / tf / f'{asset}_{tf}.csv'}")
            files_missing += 1

print()
print(f"Files found: {files_found}/{len(assets) * len(timeframes)}")
print(f"Files missing: {files_missing}")

print()
print("[3/4] Testing DataManager...")
print("-" * 80)

try:
    from src.data.data_manager import DataManager
    
    dm = DataManager()
    print(" DataManager initialized successfully")
    
    # Try to load one file
    print("\nTesting load_existing_data()...")
    for asset in assets:
        for tf in timeframes:
            try:
                df = dm.load_existing_data(asset, tf)
                if not df.empty:
                    print(f" {asset} {tf}: Loaded {len(df)} rows")
                    print(f"   Columns: {df.columns.tolist()}")
                    print(f"   Index type: {type(df.index)}")
                    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
                    break
                else:
                    print(f" {asset} {tf}: Empty DataFrame returned")
            except Exception as e:
                print(f" {asset} {tf}: Error - {e}")
        break  # Just test first asset
    
except ImportError as e:
    print(f" Cannot import DataManager: {e}")
except Exception as e:
    print(f" Error initializing DataManager: {e}")
    import traceback
    traceback.print_exc()

print()
print("[4/4] Inspecting a data file directly...")
print("-" * 80)

# Find any CSV file
csv_files = list(data_dir.rglob("*.csv"))
if csv_files:
    test_file = csv_files[0]
    print(f"Found file: {test_file}")
    print()
    
    try:
        # Try reading with pandas
        df = pd.read_csv(test_file)
        print(f" File readable with pandas")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
        print()
        print("First 3 rows:")
        print(df.head(3))
        print()
        
        # Check for required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns.str.lower()]
        
        if missing:
            print(f"  Missing required columns: {missing}")
        else:
            print(" All required OHLCV columns present")
        
        # Check for timestamp
        timestamp_cols = ['timestamp', 'date', 'time', 'datetime']
        has_timestamp = any(col in df.columns.str.lower() for col in timestamp_cols)
        
        if has_timestamp:
            print(" Timestamp column found")
        elif df.columns[0] in ['Unnamed: 0', '0']:
            print("  Possible timestamp in index (column 0)")
        else:
            print("  No obvious timestamp column")
            print(f"   First column: {df.columns[0]}")
        
    except Exception as e:
        print(f" Error reading file: {e}")
        import traceback
        traceback.print_exc()
else:
    print(" No CSV files found in data directory")

print()
print("=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)
print()

if files_missing > 0:
    print(" ISSUE: Missing data files")
    print()
    print("SOLUTION: Create sample data for testing")
    print("  python create_sample_data.py")
    print()
    print("This will create realistic OHLCV data for all assets and timeframes.")
    print()
    
elif files_found > 0:
    print(" Data files exist")
    print()
    print("If training still fails, the issue might be:")
    print("  1. File format (CSV structure)")
    print("  2. Column names (must have: open, high, low, close, volume)")
    print("  3. Timestamp/index format")
    print()
    print("Check the file inspection above for details.")
    print()

print("=" * 80)
print("RECOMMENDED ACTIONS")
print("=" * 80)
print()

if files_missing > 0:
    print("1. Create sample data:")
    print("   python create_sample_data.py")
    print()
    print("2. Or download real data and place in:")
    print("   data/raw/1h/BTC_USDT_1h.csv")
    print("   data/raw/4h/BTC_USDT_4h.csv")
    print("   ... (etc for all assets)")
    print()
else:
    print("1. If files exist but won't load, check format:")
    print("   - Must be CSV with OHLCV columns")
    print("   - Must have datetime index or timestamp column")
    print("   - Column names should be lowercase: open, high, low, close, volume")
    print()
    print("2. Try creating fresh sample data:")
    print("   python create_sample_data.py")
    print()

print("3. Then retry training:")
print("   python main.py train --both")
print()
print("=" * 80)