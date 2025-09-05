"""
Data Manager Module - FIXED VERSION

Fixed issues:
1. Better timestamp parsing for various formats
2. Handle parquet dependency gracefully
3. Fixed update_data logic
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Central data management system for loading, storing, and serving OHLCV data.
    
    Fixed version with better timestamp handling and dependency management.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the DataManager with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.base_path = Path(self.config['data']['base_path'])
        self.raw_path = Path(self.config['data']['raw_data_path'])
        self.processed_path = Path(self.config['data']['processed_data_path'])
        self.features_path = Path(self.config['data']['features_path'])
        
        # Create directories if they don't exist
        for path in [self.base_path, self.raw_path, self.processed_path, self.features_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded data
        self.data_cache = {}
        
        # Available assets and timeframes from config
        self.available_symbols = self.config['assets']['symbols']
        self.available_timeframes = self.config['timeframes']
        
        # Standard column names
        self.ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"DataManager initialized with {len(self.available_symbols)} symbols and {len(self.available_timeframes)} timeframes")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default configuration if config file is not found."""
        return {
            'data': {
                'base_path': './data',
                'raw_data_path': './data/raw',
                'processed_data_path': './data/processed',
                'features_path': './data/features'
            },
            'assets': {
                'symbols': [
                    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
                    'LINK/USDT', 'LTC/USDT', 'ATOM/USDT', 'UNI/USDT', 'ALGO/USDT'
                ]
            },
            'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d', '1w']
        }
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key for data storage."""
        return f"{symbol.replace('/', '_')}_{timeframe}"
    
    def load_existing_data(self, symbol: str, timeframe: str, 
                          use_cache: bool = True) -> pd.DataFrame:
        """
        Load OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = self._get_cache_key(symbol, timeframe)
        
        # Check cache first
        if use_cache and cache_key in self.data_cache:
            logger.debug(f"Loading {symbol} {timeframe} from cache")
            return self.data_cache[cache_key].copy()
        
        # Try different file patterns and naming conventions
        file_patterns = [
            f"{symbol.replace('/', '_')}_{timeframe}",
            f"{symbol.replace('/', '')}_{timeframe}",
            f"{symbol.split('/')[0]}_{symbol.split('/')[1]}_{timeframe}",
            f"{symbol.split('/')[0]}USDT_{timeframe}"  # For BTCUSDT format
        ]
        
        df = None
        for pattern in file_patterns:
            # Try CSV first
            csv_path = self.raw_path / timeframe / f"{pattern}.csv"
            if csv_path.exists():
                df = self._load_csv(csv_path)
                if df is not None:
                    break
            
            # Try parquet
            parquet_path = self.raw_path / timeframe / f"{pattern}.parquet"
            if parquet_path.exists():
                df = self._load_parquet(parquet_path)
                if df is not None:
                    break
            
            # Try without timeframe subfolder
            csv_path = self.raw_path / f"{pattern}.csv"
            if csv_path.exists():
                df = self._load_csv(csv_path)
                if df is not None:
                    break
        
        if df is None:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Ensure datetime index
        df = self._ensure_datetime_index(df)
        
        # Sort by index
        df = df.sort_index()
        
        # Cache the data
        if use_cache:
            self.data_cache[cache_key] = df.copy()
        
        logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
        return df
    
    def _load_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load data from CSV file with better timestamp parsing."""
        try:
            # Try reading with timestamp column
            df = pd.read_csv(filepath)
            
            # Check if there's a timestamp column
            timestamp_cols = ['timestamp', 'date', 'datetime', 'time', 'Date', 'Timestamp']
            timestamp_col = None
            for col in timestamp_cols:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                # Parse the timestamp column properly
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], 
                                                   format='%Y-%m-%d %H:%M:%S',
                                                   errors='coerce')
                # If parsing failed, try other formats
                if df[timestamp_col].isna().all():
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            return df
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}")
            return None
    
    def _load_parquet(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load data from Parquet file."""
        try:
            df = pd.read_parquet(filepath)
            return df
        except ImportError:
            logger.warning("Parquet support not available. Install pyarrow or fastparquet.")
            return None
        except Exception as e:
            logger.error(f"Error loading Parquet {filepath}: {e}")
            return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase."""
        # Common column name mappings
        column_mappings = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume',
            'o': 'open', 'h': 'high', 'l': 'low', 
            'c': 'close', 'v': 'volume',
            'date': 'timestamp', 'Date': 'timestamp', 
            'time': 'timestamp', 'Time': 'timestamp',
            'datetime': 'timestamp', 'Datetime': 'timestamp'
        }
        
        # Rename columns
        df = df.rename(columns=column_mappings)
        df.columns = df.columns.str.lower()
        
        # Ensure we have all OHLCV columns
        required_columns = self.ohlcv_columns
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, will be filled with NaN")
                df[col] = np.nan
        
        return df
    
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has a datetime index."""
        if 'timestamp' in df.columns:
            # Make sure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.set_index('timestamp')
        elif 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.set_index('date')
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Could not convert index to datetime, using integer index")
        
        return df
    
    def get_multi_timeframe_data(self, symbol: str, 
                                 timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes for a single symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes (uses all available if None)
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = self.available_timeframes
        
        data_dict = {}
        for tf in timeframes:
            df = self.load_existing_data(symbol, tf)
            if not df.empty:
                data_dict[tf] = df
            else:
                logger.warning(f"No data for {symbol} {tf}")
        
        return data_dict
    
    def get_multi_asset_data(self, symbols: Optional[List[str]] = None, 
                            timeframe: str = '1h') -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple assets for a single timeframe.
        
        Args:
            symbols: List of symbols (uses all available if None)
            timeframe: Timeframe to load
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.available_symbols
        
        data_dict = {}
        for symbol in symbols:
            df = self.load_existing_data(symbol, timeframe)
            if not df.empty:
                data_dict[symbol] = df
            else:
                logger.warning(f"No data for {symbol} {timeframe}")
        
        return data_dict
    
    def resample_data(self, df: pd.DataFrame, source_tf: str, 
                     target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data from one timeframe to another.
        
        Args:
            df: Source DataFrame with OHLCV data
            source_tf: Source timeframe (e.g., '5m')
            target_tf: Target timeframe (e.g., '1h')
            
        Returns:
            Resampled DataFrame
        """
        # Convert timeframe strings to pandas offset aliases
        tf_mapping = {
            '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
        
        target_offset = tf_mapping.get(target_tf, target_tf)
        
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = df['open'].resample(target_offset).first()
        resampled['high'] = df['high'].resample(target_offset).max()
        resampled['low'] = df['low'].resample(target_offset).min()
        resampled['close'] = df['close'].resample(target_offset).last()
        resampled['volume'] = df['volume'].resample(target_offset).sum()
        
        # Remove NaN rows
        resampled = resampled.dropna()
        
        logger.info(f"Resampled from {source_tf} to {target_tf}: {len(df)} -> {len(resampled)} rows")
        return resampled
    
    def update_data(self, symbol: str, timeframe: str, new_data: pd.DataFrame) -> None:
        """
        Update existing data with new data.
        
        Fixed to properly append new data instead of replacing.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            new_data: New OHLCV data to append
        """
        # Load existing data
        existing_data = self.load_existing_data(symbol, timeframe, use_cache=False)
        
        if existing_data.empty:
            # No existing data, save new data
            updated_data = new_data
        else:
            # Ensure both have datetime index
            if not isinstance(existing_data.index, pd.DatetimeIndex):
                existing_data = self._ensure_datetime_index(existing_data)
            if not isinstance(new_data.index, pd.DatetimeIndex):
                new_data = self._ensure_datetime_index(new_data)
            
            # Combine and remove duplicates
            updated_data = pd.concat([existing_data, new_data])
            updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
            updated_data = updated_data.sort_index()
        
        # Save updated data
        self._save_data(updated_data, symbol, timeframe)
        
        # Update cache
        cache_key = self._get_cache_key(symbol, timeframe)
        self.data_cache[cache_key] = updated_data.copy()
        
        logger.info(f"Updated {symbol} {timeframe}: {len(updated_data)} total rows")
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Save DataFrame to file."""
        filepath = self.processed_path / timeframe / f"{symbol.replace('/', '_')}_{timeframe}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def get_latest_data(self, symbol: str, timeframe: str, n_bars: int = 100) -> pd.DataFrame:
        """
        Get the latest n bars of data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            n_bars: Number of bars to retrieve
            
        Returns:
            DataFrame with latest n bars
        """
        df = self.load_existing_data(symbol, timeframe)
        if df.empty:
            return df
        
        return df.tail(n_bars)
    
    def create_training_dataset(self, symbols: Optional[List[str]] = None,
                               timeframes: Optional[List[str]] = None,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create a combined training dataset from multiple symbols and timeframes.
        
        Args:
            symbols: List of symbols to include
            timeframes: List of timeframes to include
            start_date: Start date for data (format: 'YYYY-MM-DD')
            end_date: End date for data (format: 'YYYY-MM-DD')
            
        Returns:
            Combined DataFrame with symbol and timeframe columns
        """
        if symbols is None:
            symbols = self.available_symbols[:5]  # Use first 5 symbols by default
        if timeframes is None:
            timeframes = ['1h']  # Default to 1h timeframe
        
        all_data = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                df = self.load_existing_data(symbol, timeframe)
                
                if df.empty:
                    continue
                
                # Filter by date if specified
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                
                # Add symbol and timeframe columns
                df['symbol'] = symbol
                df['timeframe'] = timeframe
                
                all_data.append(df)
        
        if not all_data:
            logger.warning("No data found for training dataset")
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.sort_index()
        
        logger.info(f"Created training dataset: {len(combined_df)} rows from {len(symbols)} symbols and {len(timeframes)} timeframes")
        return combined_df
    
    def split_data(self, df: pd.DataFrame, 
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:
            logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
    
    def get_data_info(self) -> dict:
        """
        Get information about available data.
        
        Returns:
            Dictionary with data statistics
        """
        info = {
            'symbols': self.available_symbols,
            'timeframes': self.available_timeframes,
            'data_stats': {}
        }
        
        for symbol in self.available_symbols[:3]:  # Check first 3 symbols
            symbol_stats = {}
            for tf in self.available_timeframes:
                df = self.load_existing_data(symbol, tf)
                if not df.empty:
                    symbol_stats[tf] = {
                        'rows': len(df),
                        'start_date': str(df.index[0]),
                        'end_date': str(df.index[-1]),
                        'missing_values': df.isnull().sum().to_dict()
                    }
            if symbol_stats:
                info['data_stats'][symbol] = symbol_stats
        
        return info
    
    def align_multi_asset_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Align multiple asset DataFrames to have the same timestamps.
        
        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            
        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}
        
        # Find common date range
        all_indices = [df.index for df in data_dict.values() if not df.empty]
        if not all_indices:
            return data_dict
        
        common_start = max(idx[0] for idx in all_indices)
        common_end = min(idx[-1] for idx in all_indices)
        
        # Align all DataFrames
        aligned_dict = {}
        for symbol, df in data_dict.items():
            aligned_df = df.loc[common_start:common_end]
            aligned_dict[symbol] = aligned_df
        
        logger.info(f"Aligned {len(aligned_dict)} assets to common date range: {common_start} to {common_end}")
        return aligned_dict
    
    def calculate_returns(self, df: pd.DataFrame, periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate returns for different periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods to calculate returns for
            
        Returns:
            DataFrame with return columns added
        """
        returns_df = df.copy()
        
        for period in periods:
            returns_df[f'return_{period}'] = df['close'].pct_change(period)
            returns_df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        return returns_df
    
    def save_processed_data(self, df: pd.DataFrame, name: str) -> None:
        """
        Save processed data to file.
        
        Fixed to handle missing parquet support gracefully.
        
        Args:
            df: DataFrame to save
            name: Name for the saved file
        """
        # Try parquet first, fall back to CSV
        try:
            filepath = self.processed_path / f"{name}.parquet"
            df.to_parquet(filepath)
            logger.info(f"Saved processed data to {filepath}")
        except (ImportError, Exception) as e:
            # Fall back to CSV if parquet fails
            filepath = self.processed_path / f"{name}.csv"
            df.to_csv(filepath)
            logger.info(f"Saved processed data to {filepath} (CSV format due to: {e})")
    
    def load_processed_data(self, name: str) -> pd.DataFrame:
        """
        Load processed data from file.
        
        Fixed to handle both parquet and CSV formats.
        
        Args:
            name: Name of the saved file
            
        Returns:
            Loaded DataFrame
        """
        # Try parquet first
        filepath = self.processed_path / f"{name}.parquet"
        if filepath.exists():
            try:
                return pd.read_parquet(filepath)
            except:
                pass
        
        # Try CSV
        filepath = self.processed_path / f"{name}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            logger.warning(f"Processed data file not found: {name}")
            return pd.DataFrame()
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()
        logger.info("Data cache cleared")


# Example usage and testing
if __name__ == "__main__":
    # Initialize DataManager
    dm = DataManager()
    
    # Get data info
    print("\n=== Data Information ===")
    info = dm.get_data_info()
    print(f"Available symbols: {info['symbols'][:5]}...")
    print(f"Available timeframes: {info['timeframes']}")
    
    # Load single asset data
    print("\n=== Loading BTC/USDT 1h data ===")
    btc_data = dm.load_existing_data('BTC/USDT', '1h')
    if not btc_data.empty:
        print(f"Loaded {len(btc_data)} rows")
        print(f"Date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        print(f"Columns: {btc_data.columns.tolist()}")
        print(f"Sample data:\n{btc_data.head()}")
    
    print("\n=== DataManager test complete ===")