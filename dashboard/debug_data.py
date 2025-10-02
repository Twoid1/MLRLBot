import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def create_sample_data():
    """Create sample data files for testing"""
    data_path = './data/raw'
    os.makedirs(data_path, exist_ok=True)
    
    # Create sample data for BTC_USDT
    assets = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    for asset in assets:
        for tf in timeframes:
            filename = f"{asset}_{tf}.csv"
            filepath = os.path.join(data_path, filename)
            
            if not os.path.exists(filepath):
                print(f"Creating sample data: {filename}")
                
                # Generate sample OHLCV data
                periods = 1000
                dates = pd.date_range(end=datetime.now(), periods=periods, freq=tf)
                
                # Random walk for price
                base_price = {'BTC_USDT': 40000, 'ETH_USDT': 2500, 'SOL_USDT': 100}[asset]
                returns = np.random.randn(periods) * 0.01
                close = base_price * (1 + returns).cumprod()
                
                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': close * (1 + np.random.randn(periods) * 0.002),
                    'high': close * (1 + np.abs(np.random.randn(periods) * 0.005)),
                    'low': close * (1 - np.abs(np.random.randn(periods) * 0.005)),
                    'close': close,
                    'volume': np.random.uniform(1000, 10000, periods)
                })
                
                df.set_index('timestamp', inplace=True)
                df.to_csv(filepath)
                print(f" Created {filename}")
            else:
                print(f" File exists: {filename}")

if __name__ == "__main__":
    create_sample_data()
    print("\nData check complete!")