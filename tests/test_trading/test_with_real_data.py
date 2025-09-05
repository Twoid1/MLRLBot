#!/usr/bin/env python
"""
Test Trading Modules with Real Data
This script tests the trading modules using your actual data from data/raw/
Place this in the project root or tests/ directory
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import trading modules
try:
    from src.trading.risk_manager import RiskManager, RiskConfig, RiskLevel
    from src.trading.position_sizer import PositionSizer
    from src.trading.portfolio import Portfolio
    from src.trading.executor import OrderExecutor, MarketConditions
    print(" All trading modules imported successfully")
except ImportError as e:
    print(f" Error importing trading modules: {e}")
    sys.exit(1)


class RealDataTester:
    """Test trading modules with real market data"""
    
    def __init__(self, data_path: str = "data/raw"):
        """Initialize with path to data directory"""
        self.data_path = Path(data_path)
        self.available_data = {}
        self.loaded_data = {}
        
        # Scan for available data
        self._scan_data_directory()
    
    def _scan_data_directory(self):
        """Scan data directory for available files"""
        print(f"\n Scanning {self.data_path} for data files...")
        
        if not self.data_path.exists():
            print(f" Data directory not found: {self.data_path}")
            return
        
        # Look for CSV files in timeframe subdirectories
        timeframes = ['1h', '4h', '1d', '1w', '30m', '15m', '5m']
        
        for tf in timeframes:
            tf_path = self.data_path / tf
            if tf_path.exists():
                csv_files = list(tf_path.glob("*.csv"))
                if csv_files:
                    self.available_data[tf] = csv_files
                    print(f"   Found {len(csv_files)} files in {tf}/")
        
        # Also check root directory
        root_csv = list(self.data_path.glob("*.csv"))
        if root_csv:
            self.available_data['root'] = root_csv
            print(f"   Found {len(root_csv)} files in root")
        
        if not self.available_data:
            print(" No data files found!")
        else:
            print(f" Found data for {len(self.available_data)} timeframes")
    
    def load_data(self, timeframe: str = '1h', symbol_filter: str = 'BTC'):
        """Load data for specific timeframe and symbol"""
        if timeframe not in self.available_data:
            print(f" No data found for timeframe: {timeframe}")
            return None
        
        # Find file matching symbol filter
        files = self.available_data[timeframe]
        matching_files = [f for f in files if symbol_filter.upper() in f.stem.upper()]
        
        if not matching_files:
            print(f" No files found matching: {symbol_filter}")
            return None
        
        # Load first matching file
        file_path = matching_files[0]
        print(f"\n Loading: {file_path.name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Try to identify and set datetime index
            date_columns = ['date', 'Date', 'datetime', 'timestamp', 'time', 'Time']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
            
            # If no date column found, try to parse index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    print("    Could not parse datetime index")
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Check for required OHLCV columns
            required = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                print(f"    Missing columns: {missing}")
                # Try alternative names
                alt_names = {
                    'open': ['Open', 'o'],
                    'high': ['High', 'h'],
                    'low': ['Low', 'l'],
                    'close': ['Close', 'c'],
                    'volume': ['Volume', 'vol', 'v']
                }
                
                for std_name, alternatives in alt_names.items():
                    if std_name not in df.columns:
                        for alt in alternatives:
                            if alt in df.columns or alt.lower() in df.columns:
                                df[std_name] = df[alt] if alt in df.columns else df[alt.lower()]
                                break
            
            print(f"    Loaded {len(df)} rows")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f" Error loading file: {e}")
            return None
    
    def test_with_data(self, df: pd.DataFrame, symbol: str = "BTC/USDT"):
        """Run tests using loaded data"""
        print(f"\n{'='*60}")
        print(f"TESTING WITH REAL DATA: {symbol}")
        print(f"{'='*60}")
        
        # Get recent data for testing
        test_data = df.tail(100)
        
        # Calculate basic metrics
        current_price = test_data['close'].iloc[-1]
        returns = test_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        print(f"\n Data Statistics:")
        print(f"   Current price: ${current_price:,.2f}")
        print(f"   Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        print(f"   Daily range: ${test_data['high'].iloc[-1] - test_data['low'].iloc[-1]:,.2f}")
        
        # Initialize trading modules
        initial_capital = 10000
        
        print(f"\n Initializing Trading System...")
        risk_manager = RiskManager(initial_capital)
        position_sizer = PositionSizer(initial_capital)
        portfolio = Portfolio(initial_capital)
        executor = OrderExecutor(mode='simulation')
        
        # Test 1: Position Sizing
        print(f"\n POSITION SIZING TEST")
        
        # Calculate position using different methods
        kelly_result = position_sizer.kelly_criterion_basic(
            win_probability=0.55,
            win_amount=returns[returns > 0].mean() * current_price if len(returns[returns > 0]) > 0 else 100,
            loss_amount=abs(returns[returns < 0].mean()) * current_price if len(returns[returns < 0]) > 0 else 100,
            kelly_fraction=0.25
        )
        
        risk_based_size = risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=0.7,
            current_price=current_price,
            volatility=volatility
        )
        
        print(f"   Kelly position value: ${kelly_result.position_value:.2f}")
        print(f"   Kelly position size: {kelly_result.position_value/current_price:.6f} units")
        print(f"   Risk-based size: {risk_based_size:.6f} units")
        print(f"   Risk-based value: ${risk_based_size * current_price:.2f}")
        
        # Test 2: Risk Calculations
        print(f"\n RISK MANAGEMENT TEST")
        
        # Calculate ATR for stop loss
        high_low = test_data['high'] - test_data['low']
        high_close = abs(test_data['high'] - test_data['close'].shift())
        low_close = abs(test_data['low'] - test_data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        stop_loss = risk_manager.calculate_stop_loss(current_price, 'long', atr)
        take_profit = risk_manager.calculate_take_profit(current_price, 'long', atr)
        
        print(f"   ATR (14): ${atr:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f} ({((stop_loss/current_price)-1)*100:.2f}%)")
        print(f"   Take Profit: ${take_profit:.2f} ({((take_profit/current_price)-1)*100:.2f}%)")
        
        # Test 3: Simulated Trade
        print(f"\n SIMULATED TRADE TEST")
        
        # Set market conditions
        spread = current_price * 0.0001  # 0.01% spread
        market = MarketConditions(
            bid=current_price - spread/2,
            ask=current_price + spread/2,
            last_price=current_price,
            spread=spread,
            volume=float(test_data['volume'].iloc[-1]),
            volatility=volatility,
            liquidity=0.9
        )
        executor.update_market_conditions(symbol, market)
        
        # Create and execute order
        position_size = min(risk_based_size, kelly_result.position_value/current_price)
        order = executor.create_order(
            symbol=symbol,
            side='buy',
            quantity=position_size,
            order_type='market'
        )
        
        result = executor.submit_order(order)
        
        if result.success:
            print(f"    Order executed successfully")
            print(f"   Execution price: ${result.execution_price:.2f}")
            print(f"   Slippage: {result.slippage:.4%}")
            print(f"   Commission: ${result.commission:.2f}")
            
            # Add to portfolio
            portfolio.open_position(
                symbol=symbol,
                quantity=result.actual_quantity,
                price=result.execution_price,
                position_type='long',
                stop_loss=stop_loss,
                take_profit=take_profit,
                fees=result.commission
            )
            
            # Simulate price movements
            print(f"\n BACKTESTING WITH HISTORICAL DATA")
            
            # Use last 10 prices to simulate P&L
            simulation_prices = test_data['close'].tail(10).values
            
            for i, price in enumerate(simulation_prices):
                portfolio.update_position_price(symbol, price)
                position = portfolio.positions.get(symbol)
                
                if position:
                    pnl = position.unrealized_pnl
                    pnl_pct = position.unrealized_return * 100
                    print(f"   Day {i+1}: ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            
            # Final portfolio status
            summary = portfolio.get_summary()
            print(f"\n PORTFOLIO SUMMARY:")
            print(f"   Total Value: ${summary['value']['total']:.2f}")
            print(f"   Unrealized P&L: ${summary['pnl']['unrealized']:.2f}")
            print(f"   Total Return: {summary['pnl']['total_return']:.2%}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test with all available data"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST WITH ALL AVAILABLE DATA")
        print("="*60)
        
        if not self.available_data:
            print(" No data available for testing")
            return
        
        # Test with different timeframes
        test_configs = [
            ('1h', 'BTC'),
            ('4h', 'ETH'),
            ('1d', 'BTC')
        ]
        
        for timeframe, symbol_filter in test_configs:
            if timeframe in self.available_data:
                df = self.load_data(timeframe, symbol_filter)
                if df is not None and not df.empty:
                    self.test_with_data(df, f"{symbol_filter}/USDT")
                else:
                    print(f" Skipping {timeframe} {symbol_filter} - no data")
            else:
                print(f" No data for timeframe: {timeframe}")


def main():
    """Main entry point"""
    print("="*70)
    print("REAL DATA TESTING TOOL")
    print("Testing trading modules with your actual market data")
    print("="*70)
    
    # Check for data directory
    data_dirs = [
        "data/raw",
        "../data/raw",
        "../../data/raw",
        "./raw"
    ]
    
    tester = None
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f" Found data directory: {data_dir}")
            tester = RealDataTester(data_dir)
            break
    
    if tester is None:
        print("\n Could not find data directory!")
        print("Please ensure your data is in one of these locations:")
        for dir in data_dirs:
            print(f"   - {dir}/")
        return
    
    # Check if we have any data
    if not tester.available_data:
        print("\n No data files found in the data directory!")
        print("Expected structure:")
        print("   data/raw/")
        print("       1h/")
        print("           BTCUSDT_1h.csv")
        print("           ETHUSDT_1h.csv")
        print("       4h/")
        print("           BTCUSDT_4h.csv")
        print("           ...")
        return
    
    # Show available data
    print("\n Available Data:")
    for timeframe, files in tester.available_data.items():
        print(f"\n   {timeframe}:")
        for f in files[:3]:  # Show first 3 files
            print(f"      - {f.name}")
        if len(files) > 3:
            print(f"      ... and {len(files)-3} more files")
    
    # Run tests
    print("\n" + "="*70)
    print("Starting Tests...")
    print("="*70)
    
    # Try to load and test with BTC 1h data first
    if '1h' in tester.available_data:
        df = tester.load_data('1h', 'BTC')
        if df is not None:
            tester.test_with_data(df, 'BTC/USDT')
    
    # Ask if user wants to run comprehensive test
    print("\n" + "="*70)
    try:
        response = input("Run comprehensive test with all data? (y/n): ")
        if response.lower() == 'y':
            tester.run_comprehensive_test()
    except:
        # If running non-interactively, skip
        pass
    
    print("\n" + "="*70)
    print(" Real data testing complete!")
    print("="*70)


if __name__ == "__main__":
    main()