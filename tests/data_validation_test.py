"""
Data Quality Validation Test
Checks your actual training data for anomalies that could cause bugs

This will find:
- Broken prices (NaN, Inf, zeros, negatives)
- Extreme price movements (1000x jumps)
- Volume anomalies
- Timestamp issues
- Data that could make positions look impossible
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))


def check_price_data_quality(df, asset_name="Unknown"):
    """
    Check OHLCV data for quality issues
    
    Returns: (passed, issues_found)
    """
    print(f"\n{'='*80}")
    print(f" CHECKING {asset_name} DATA QUALITY")
    print(f"{'='*80}")
    
    issues = []
    warnings_found = []
    
    print(f"\nDataset Info:")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check 1: Missing data
    print(f"\n1. Checking for missing data...")
    missing = df.isnull().sum()
    if missing.any():
        print(f"   FAIL: Found missing data!")
        for col, count in missing[missing > 0].items():
            print(f"     {col}: {count} missing values ({count/len(df)*100:.2f}%)")
            issues.append(f"Missing {count} values in {col}")
    else:
        print(f"   PASS: No missing data")
    
    # Check 2: NaN or Inf values
    print(f"\n2. Checking for NaN/Inf values...")
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            continue
        
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        
        if nan_count > 0:
            print(f"   FAIL: {col} has {nan_count} NaN values")
            issues.append(f"{col} has {nan_count} NaN values")
        
        if inf_count > 0:
            print(f"   FAIL: {col} has {inf_count} Inf values")
            issues.append(f"{col} has {inf_count} Inf values")
    
    if not issues:
        print(f"   PASS: No NaN/Inf values")
    
    # Check 3: Zero or negative prices
    print(f"\n3. Checking for invalid prices (≤0)...")
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            continue
        
        zero_count = (df[col] <= 0).sum()
        if zero_count > 0:
            print(f"   FAIL: {col} has {zero_count} zero/negative prices")
            print(f"     Indices: {df[df[col] <= 0].index.tolist()[:5]}")
            issues.append(f"{col} has invalid prices")
    
    if not any("invalid prices" in i for i in issues):
        print(f"   PASS: All prices positive")
    
    # Check 4: Extreme price jumps
    print(f"\n4. Checking for extreme price movements...")
    for col in ['close']:
        if col not in df.columns:
            continue
        
        returns = df[col].pct_change()
        
        # Find extreme moves (>100% in one hour)
        extreme_up = returns > 1.0  # 100%+ gain
        extreme_down = returns < -0.5  # 50%+ drop
        
        if extreme_up.any():
            count = extreme_up.sum()
            max_jump = returns[extreme_up].max()
            print(f"    WARNING: {count} extreme upward moves (>{max_jump:.1%})")
            affected_dates = df[extreme_up].index.tolist()[:3]
            print(f"     Examples: {affected_dates}")
            warnings_found.append(f"{count} extreme price jumps")
        
        if extreme_down.any():
            count = extreme_down.sum()
            max_drop = returns[extreme_down].min()
            print(f"    WARNING: {count} extreme downward moves (<{max_drop:.1%})")
            affected_dates = df[extreme_down].index.tolist()[:3]
            print(f"     Examples: {affected_dates}")
            warnings_found.append(f"{count} extreme price drops")
    
    if not warnings_found:
        print(f"   PASS: No extreme price movements")
    
    # Check 5: OHLC relationship validity
    print(f"\n5. Checking OHLC relationships...")
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High should be >= Open, Close, Low
        invalid_high = ((df['high'] < df['open']) | 
                       (df['high'] < df['close']) | 
                       (df['high'] < df['low'])).sum()
        
        # Low should be <= Open, Close, High
        invalid_low = ((df['low'] > df['open']) | 
                      (df['low'] > df['close']) | 
                      (df['low'] > df['high'])).sum()
        
        if invalid_high > 0:
            print(f"   FAIL: {invalid_high} rows where high < other prices")
            issues.append(f"Invalid OHLC: {invalid_high} rows")
        
        if invalid_low > 0:
            print(f"   FAIL: {invalid_low} rows where low > other prices")
            issues.append(f"Invalid OHLC: {invalid_low} rows")
        
        if invalid_high == 0 and invalid_low == 0:
            print(f"   PASS: OHLC relationships valid")
    
    # Check 6: Price scale anomalies
    print(f"\n6. Checking for price scale anomalies...")
    close_prices = df['close'].dropna()
    
    min_price = close_prices.min()
    max_price = close_prices.max()
    median_price = close_prices.median()
    
    print(f"  Price range: ${min_price:.8f} to ${max_price:.8f}")
    print(f"  Median: ${median_price:.8f}")
    
    # Check for suspicious patterns
    if min_price < 0.0001:
        print(f"    WARNING: Very small prices detected (${min_price:.8f})")
        print(f"     This could cause numerical precision issues")
        warnings_found.append("Very small prices")
    
    if max_price / min_price > 1000:
        print(f"    WARNING: Price range is {max_price/min_price:.0f}x")
        print(f"     From ${min_price:.8f} to ${max_price:.8f}")
        warnings_found.append("Extreme price range")
    
    # Check 7: The specific SOL issue from your report
    print(f"\n7. Checking for the SOL $3.53 anomaly...")
    
    # Look for prices around $3.53 that could cause issues
    prices_near_353 = df[(df['close'] > 3.50) & (df['close'] < 3.56)]
    
    if len(prices_near_353) > 0:
        print(f"  Found {len(prices_near_353)} rows with prices near $3.53")
        
        # Check if these prices have any anomalies
        sample_prices = prices_near_353['close'].head(10)
        print(f"  Sample prices: {sample_prices.tolist()}")
        
        # Check for precision issues
        for idx in prices_near_353.index[:5]:
            price = df.loc[idx, 'close']
            # Try to calculate position size
            capital = 9500
            expected_coins = capital / price
            
            print(f"  At ${price:.8f}: Can buy {expected_coins:.2f} coins with $9,500")
            
            if expected_coins > 10000:
                print(f"      WARNING: This would give {expected_coins:.2f} coins!")
                print(f"    If price has precision issues, this could explain your bug")
                warnings_found.append(f"Price {price} gives {expected_coins:.0f} coins")
    
    # Check 8: Volume sanity
    print(f"\n8. Checking volume data...")
    if 'volume' in df.columns:
        volume = df['volume'].dropna()
        
        if (volume == 0).any():
            zero_vol_count = (volume == 0).sum()
            print(f"    WARNING: {zero_vol_count} periods with zero volume")
            warnings_found.append(f"{zero_vol_count} zero volume periods")
        
        if (volume < 0).any():
            print(f"   FAIL: Negative volume detected")
            issues.append("Negative volume")
        else:
            print(f"   PASS: Volume data looks good")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY FOR {asset_name}")
    print(f"{'='*80}")
    print(f"Critical Issues: {len(issues)}")
    print(f"Warnings: {len(warnings_found)}")
    
    if issues:
        print(f"\n CRITICAL ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    
    if warnings_found:
        print(f"\n  WARNINGS:")
        for i, warning in enumerate(warnings_found, 1):
            print(f"  {i}. {warning}")
    
    if not issues and not warnings_found:
        print(f"\n DATA QUALITY: EXCELLENT")
    elif not issues:
        print(f"\n  DATA QUALITY: ACCEPTABLE (but has warnings)")
    else:
        print(f"\n DATA QUALITY: POOR (critical issues found)")
    
    return len(issues) == 0, issues, warnings_found


def load_and_check_training_data():
    """
    Load your actual training data and check it
    """
    print(f"\n{'='*80}")
    print(f" LOADING YOUR ACTUAL TRAINING DATA")
    print(f"{'='*80}")
    
    # Try to find your data files
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print(f"\n  Data directory not found: {data_dir}")
        print(f"Please update the path to your actual data directory")
        return
    
    # Look for SOL data specifically
    sol_files = list(data_dir.glob("*SOL*"))
    
    if not sol_files:
        print(f"\n  No SOL data files found in {data_dir}")
        print(f"Looking for any OHLCV files...")
        all_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))
        
        if all_files:
            print(f"\nFound {len(all_files)} data files:")
            for f in all_files[:10]:
                print(f"  - {f.name}")
        else:
            print(f"\n No data files found!")
            print(f"Please check your data directory path")
            return
    else:
        print(f"\nFound {len(sol_files)} SOL data files:")
        for f in sol_files:
            print(f"  - {f.name}")
    
    # Load and check each file
    results = {}
    
    for file_path in sol_files[:5]:  # Check first 5 files
        try:
            print(f"\nLoading {file_path.name}...")
            
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                continue
            
            passed, issues, warnings = check_price_data_quality(df, file_path.stem)
            results[file_path.name] = {
                'passed': passed,
                'issues': issues,
                'warnings': warnings
            }
            
        except Exception as e:
            print(f" Error loading {file_path.name}: {e}")
    
    # Overall summary
    if results:
        print(f"\n{'='*80}")
        print(f" OVERALL DATA QUALITY SUMMARY")
        print(f"{'='*80}")
        
        for filename, result in results.items():
            status = "Y" if result['passed'] else "X"
            issues_count = len(result['issues'])
            warnings_count = len(result['warnings'])
            
            print(f"\n{status} {filename}")
            if issues_count > 0:
                print(f"   {issues_count} critical issues")
            if warnings_count > 0:
                print(f"   {warnings_count} warnings")


def check_specific_sol_date():
    """
    Check the specific date from your report: 2020-09-09 14:00:00
    """
    print(f"\n{'='*80}")
    print(f" CHECKING SPECIFIC DATE FROM YOUR REPORT")
    print(f"{'='*80}")
    print(f"\nYour report shows:")
    print(f"  Timestamp: 2020-09-09 14:00:00")
    print(f"  Price: $3.53")
    print(f"  Position: 18,928 SOL")
    print(f"  This should be impossible with $10k balance!\n")
    
    # Try to load SOL data
    data_dir = Path("data/processed")
    sol_files = list(data_dir.glob("*SOL*1h*")) + list(data_dir.glob("*SOL*hourly*"))
    
    if not sol_files:
        print(f"  Could not find SOL hourly data")
        print(f"Please provide the path to your SOL training data")
        return
    
    try:
        file_path = sol_files[0]
        print(f"Loading: {file_path.name}")
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        else:
            df = pd.read_parquet(file_path)
        
        # Find the specific date
        target_date = pd.Timestamp('2020-09-09 14:00:00')
        
        if target_date in df.index:
            print(f"\n Found the exact timestamp!")
            
            # Get data around this date
            idx = df.index.get_loc(target_date)
            context_df = df.iloc[max(0, idx-5):min(len(df), idx+6)]
            
            print(f"\nData around 2020-09-09 14:00:00:")
            print(context_df[['open', 'high', 'low', 'close', 'volume']])
            
            # Check this specific price
            price = df.loc[target_date, 'close']
            print(f"\nClose price at 2020-09-09 14:00:00: ${price:.8f}")
            
            # Calculate what position size you could buy
            capital = 9500  # 95% of $10k
            max_coins = capital / price
            
            print(f"With $9,500, you can buy: {max_coins:.2f} SOL")
            print(f"Your report showed: 18,928 SOL")
            print(f"That's {18928 / max_coins:.1f}x more than possible!")
            
            # Check if price has precision issues
            print(f"\nPrice precision check:")
            print(f"  Price as float64: {price}")
            print(f"  Price as string: {str(price)}")
            print(f"  Price * 1e8: {price * 1e8}")
            
            # Check surrounding prices
            print(f"\nSurrounding prices:")
            for i in range(max(0, idx-2), min(len(df), idx+3)):
                date = df.index[i]
                p = df.iloc[i]['close']
                coins = 9500 / p
                marker = " ← TARGET" if i == idx else ""
                print(f"  {date}: ${p:.8f} → {coins:.2f} coins{marker}")
        
        else:
            print(f"\n  Exact timestamp not found")
            print(f"Available date range: {df.index[0]} to {df.index[-1]}")
            
            # Find closest date
            closest_idx = df.index.get_indexer([target_date], method='nearest')[0]
            closest_date = df.index[closest_idx]
            print(f"Closest date: {closest_date}")
            
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all data validation tests"""
    print(f"\n{'='*80}")
    print(f" DATA QUALITY VALIDATION TEST SUITE")
    print(f"{'='*80}")
    print(f"\nThis will check your actual training data for issues that")
    print(f"could cause the impossible trades in your report.")
    print(f"{'='*80}")
    
    # Test 1: Load and check all training data
    load_and_check_training_data()
    
    # Test 2: Check the specific problematic date
    check_specific_sol_date()
    
    print(f"\n{'='*80}")
    print(f" DATA VALIDATION COMPLETE")
    print(f"{'='*80}")
    
    print(f"\nNext steps:")
    print(f"  1. Review any critical issues found above")
    print(f"  2. If no issues found in data, check feature calculation")
    print(f"  3. Check if features_df has any extreme values")
    print(f"  4. Verify that ML predictions aren't causing issues")


if __name__ == "__main__":
    main()