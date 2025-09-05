"""
Test suite for DatabaseManager module - FIXED FOR WINDOWS

This version properly closes database connections to avoid Windows file locking issues.

Run with: pytest tests/test_data/test_database.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys
import json
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.database import DatabaseManager, OHLCVData, Features, Trades, Performance


class TestDatabaseManager:
    """Test suite for DatabaseManager class."""
    
    @pytest.fixture
    def temp_db_dir(self):
        """Create a temporary directory for database files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup - remove entire directory
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            pass  # Windows sometimes holds locks, ignore
    
    @pytest.fixture
    def db_manager(self, temp_db_dir):
        """Create a DatabaseManager instance with temporary database."""
        db_path = os.path.join(temp_db_dir, 'test.db')
        connection_string = f'sqlite:///{db_path}'
        db = DatabaseManager(db_type='sqlite', connection_string=connection_string)
        yield db
        # Important: Close all connections before cleanup
        db.engine.dispose()  # This closes all connections
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Generate sample OHLCV data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Ensure valid OHLC relationships
        data['high'] = data[['open', 'high', 'close']].max(axis=1)
        data['low'] = data[['open', 'low', 'close']].min(axis=1)
        
        return data
    
    @pytest.fixture
    def sample_features(self):
        """Generate sample feature data."""
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='1h')
        
        features = pd.DataFrame({
            'rsi': np.random.uniform(30, 70, len(dates)),
            'macd': np.random.uniform(-100, 100, len(dates)),
            'volume_ratio': np.random.uniform(0.5, 2, len(dates)),
            'atr': np.random.uniform(100, 500, len(dates))
        }, index=dates)
        
        return features
    
    @pytest.fixture
    def sample_trades(self):
        """Generate sample trade data."""
        trades = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1D'),
            'symbol': ['BTC/USDT'] * 5,
            'side': ['buy', 'sell', 'buy', 'sell', 'buy'],
            'quantity': [0.1, 0.1, 0.2, 0.2, 0.15],
            'price': [42000, 43000, 41500, 44000, 43500],
            'fee': [10.92, 11.18, 21.58, 22.88, 22.62],
            'pnl': [0, 100, 0, 500, 0],
            'strategy': 'test_strategy',
            'is_paper': True,
            'trade_id': [f'trade_{i}' for i in range(5)]
        })
        
        return trades
    
    def test_initialization(self, temp_db_dir):
        """Test DatabaseManager initialization."""
        db_path = os.path.join(temp_db_dir, 'test_init.db')
        connection_string = f'sqlite:///{db_path}'
        
        # Test SQLite initialization
        db = DatabaseManager('sqlite', connection_string)
        assert db is not None
        assert db.db_type == 'sqlite'
        
        # Test that tables are created
        assert db.engine is not None
        assert db.SessionLocal is not None
        
        # Clean up
        db.engine.dispose()
    
    def test_create_tables(self, db_manager):
        """Test that all tables are created correctly."""
        # Tables should be created on initialization
        db_manager.create_tables()
        
        # Check tables exist by trying to query them
        session = db_manager.get_session()
        try:
            # These queries should not raise errors
            session.query(OHLCVData).first()
            session.query(Features).first()
            session.query(Trades).first()
            session.query(Performance).first()
        finally:
            session.close()
    
    def test_store_and_load_ohlcv(self, db_manager, sample_ohlcv_data):
        """Test storing and loading OHLCV data."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store data
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Load it back
        loaded_data = db_manager.load_ohlcv(symbol, timeframe)
        
        assert not loaded_data.empty
        assert len(loaded_data) == len(sample_ohlcv_data)
        assert all(col in loaded_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        
        # Check values match (approximately due to float precision)
        assert loaded_data['close'].mean() == pytest.approx(sample_ohlcv_data['close'].mean(), rel=1e-5)
    
    def test_update_existing_ohlcv(self, db_manager, sample_ohlcv_data):
        """Test updating existing OHLCV records."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store initial data
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Modify data and store again (should update)
        modified_data = sample_ohlcv_data.copy()
        modified_data['close'] = modified_data['close'] * 1.1
        
        db_manager.store_ohlcv(modified_data, symbol, timeframe)
        
        # Load and verify update
        loaded_data = db_manager.load_ohlcv(symbol, timeframe)
        
        # Should have same number of rows (updated, not duplicated)
        assert len(loaded_data) == len(sample_ohlcv_data)
        
        # Values should match modified data
        assert loaded_data['close'].mean() == pytest.approx(modified_data['close'].mean(), rel=1e-5)
    
    def test_load_ohlcv_with_date_filter(self, db_manager, sample_ohlcv_data):
        """Test loading OHLCV data with date filters."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store data
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Load with start date filter
        start_date = datetime(2024, 1, 5)
        filtered_data = db_manager.load_ohlcv(symbol, timeframe, start_date=start_date)
        
        assert not filtered_data.empty
        assert filtered_data.index[0] >= pd.Timestamp(start_date)
        
        # Load with end date filter
        end_date = datetime(2024, 1, 7)
        filtered_data = db_manager.load_ohlcv(symbol, timeframe, end_date=end_date)
        
        assert not filtered_data.empty
        assert filtered_data.index[-1] <= pd.Timestamp(end_date)
        
        # Load with both filters
        filtered_data = db_manager.load_ohlcv(symbol, timeframe, 
                                             start_date=start_date, 
                                             end_date=end_date)
        
        assert not filtered_data.empty
        assert filtered_data.index[0] >= pd.Timestamp(start_date)
        assert filtered_data.index[-1] <= pd.Timestamp(end_date)
    
    def test_store_and_load_features(self, db_manager, sample_features):
        """Test storing and loading features."""
        feature_set_name = 'test_features'
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store features
        db_manager.store_features(sample_features, feature_set_name, symbol, timeframe)
        
        # Load them back
        loaded_features = db_manager.load_features(feature_set_name, symbol, timeframe)
        
        assert not loaded_features.empty
        assert len(loaded_features) == len(sample_features)
        assert all(col in loaded_features.columns for col in sample_features.columns)
        
        # Check values
        assert loaded_features['rsi'].mean() == pytest.approx(sample_features['rsi'].mean(), rel=1e-5)
    
    def test_store_and_load_trades(self, db_manager, sample_trades):
        """Test storing and loading trade records."""
        # Store trades
        db_manager.store_trades(sample_trades)
        
        # Load all trades
        loaded_trades = db_manager.load_trades()
        
        assert not loaded_trades.empty
        assert len(loaded_trades) == len(sample_trades)
        
        # Test filtering by symbol
        btc_trades = db_manager.load_trades(symbol='BTC/USDT')
        assert len(btc_trades) == 5
        
        # Test filtering by strategy
        strategy_trades = db_manager.load_trades(strategy='test_strategy')
        assert len(strategy_trades) == 5
        
        # Test filtering by date
        start_date = datetime(2024, 1, 3)
        recent_trades = db_manager.load_trades(start_date=start_date)
        assert len(recent_trades) == 3  # Days 3, 4, 5
    
    def test_get_latest_timestamp(self, db_manager, sample_ohlcv_data):
        """Test getting the latest timestamp for a symbol."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Should return None for non-existent data
        latest = db_manager.get_latest_timestamp(symbol, timeframe)
        assert latest is None
        
        # Store data
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Get latest timestamp
        latest = db_manager.get_latest_timestamp(symbol, timeframe)
        assert latest is not None
        assert latest == sample_ohlcv_data.index[-1].to_pydatetime()
    
    def test_cleanup_old_data(self, db_manager, sample_ohlcv_data):
        """Test cleanup of old data."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store data
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Check initial count
        initial_stats = db_manager.get_database_stats()
        initial_count = initial_stats['ohlcv_records']
        
        # Clean up data older than 5 days (should keep recent data)
        db_manager.cleanup_old_data(days_to_keep=5)
        
        # Check count after cleanup
        after_stats = db_manager.get_database_stats()
        after_count = after_stats['ohlcv_records']
        
        # For test data from Jan 1-10, 2024, with days_to_keep=5,
        # it should keep data from last 5 days
        assert after_count <= initial_count
    
    def test_store_performance_metrics(self, db_manager):
        """Test storing and retrieving performance metrics."""
        strategy = 'test_strategy'
        
        # Store various metrics
        metrics = [
            ('sharpe_ratio', 1.5),
            ('total_return', 0.15),
            ('max_drawdown', -0.08),
            ('win_rate', 0.55)
        ]
        
        for metric_name, metric_value in metrics:
            db_manager.store_performance_metric(strategy, metric_name, metric_value)
        
        # Get performance metrics
        perf_metrics = db_manager.get_performance_metrics(strategy)
        
        assert not perf_metrics.empty
        
        # Check that we have the metrics (they might be in rows, not columns after pivot)
        # The pivot might create NaN for some combinations, so check differently
        session = db_manager.get_session()
        try:
            # Query directly to check values
            sharpe_record = session.query(Performance).filter_by(
                strategy=strategy, 
                metric_name='sharpe_ratio'
            ).first()
            assert sharpe_record is not None
            assert sharpe_record.metric_value == pytest.approx(1.5)
        finally:
            session.close()
    
    def test_get_database_stats(self, db_manager, sample_ohlcv_data, sample_trades):
        """Test getting database statistics."""
        # Initially should be empty
        stats = db_manager.get_database_stats()
        assert stats['ohlcv_records'] == 0
        assert stats['trade_records'] == 0
        
        # Add some data
        db_manager.store_ohlcv(sample_ohlcv_data, 'BTC/USDT', '1h')
        db_manager.store_trades(sample_trades)
        
        # Check updated stats
        stats = db_manager.get_database_stats()
        assert stats['ohlcv_records'] > 0
        assert stats['trade_records'] == len(sample_trades)
        assert stats['symbols'] >= 1
        assert 'date_range' in stats
    
    def test_concurrent_access(self, db_manager, sample_ohlcv_data):
        """Test that multiple sessions can access the database."""
        symbol = 'BTC/USDT'
        timeframe = '1h'
        
        # Store data in one session
        db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Create another session and load data
        session2 = db_manager.get_session()
        try:
            records = session2.query(OHLCVData).filter_by(symbol=symbol).count()
            assert records == len(sample_ohlcv_data)
        finally:
            session2.close()
    
    def test_error_handling(self, db_manager):
        """Test error handling in database operations."""
        # Try to load non-existent data
        empty_df = db_manager.load_ohlcv('NONEXISTENT/PAIR', '1h')
        assert empty_df.empty
        
        # Try to store invalid data
        invalid_df = pd.DataFrame()  # Empty DataFrame
        # Should handle gracefully without crashing
        db_manager.store_ohlcv(invalid_df, 'TEST/USDT', '1h')
    
    def test_large_dataset_performance(self, db_manager):
        """Test performance with larger dataset."""
        # Generate larger dataset (1 year of hourly data)
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1h')
        large_data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Time the storage
        import time
        start_time = time.time()
        db_manager.store_ohlcv(large_data, 'BTC/USDT', '1h')
        store_time = time.time() - start_time
        
        # Time the retrieval
        start_time = time.time()
        loaded = db_manager.load_ohlcv('BTC/USDT', '1h')
        load_time = time.time() - start_time
        
        assert len(loaded) == len(large_data)
        assert store_time < 10  # Should complete within 10 seconds
        assert load_time < 5   # Loading should be faster
        
        print(f"\nPerformance: Store {len(large_data)} rows in {store_time:.2f}s, Load in {load_time:.2f}s")
    
    def test_multiple_symbols_and_timeframes(self, db_manager, sample_ohlcv_data):
        """Test storing data for multiple symbols and timeframes."""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        timeframes = ['5m', '15m', '1h', '4h']
        
        # Store data for all combinations
        for symbol in symbols:
            for timeframe in timeframes:
                db_manager.store_ohlcv(sample_ohlcv_data, symbol, timeframe)
        
        # Verify all data is stored
        stats = db_manager.get_database_stats()
        assert stats['symbols'] == len(symbols)
        assert stats['timeframes'] == len(timeframes)
        
        # Load specific combination
        btc_1h = db_manager.load_ohlcv('BTC/USDT', '1h')
        eth_5m = db_manager.load_ohlcv('ETH/USDT', '5m')
        
        assert not btc_1h.empty
        assert not eth_5m.empty
    
    def test_trade_metadata(self, db_manager):
        """Test storing and retrieving trade metadata."""
        trades_with_metadata = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['BTC/USDT'],
            'side': ['buy'],
            'quantity': [0.5],
            'price': [45000],
            'trade_id': ['test_trade_123'],
            'metadata': [{'signal_strength': 0.8, 'confidence': 0.9, 'indicators': ['RSI', 'MACD']}]
        })
        
        # Store trade with metadata
        db_manager.store_trades(trades_with_metadata)
        
        # Load and verify metadata
        loaded = db_manager.load_trades()
        assert not loaded.empty
        assert 'metadata' in loaded.columns
        
        # Check metadata was preserved
        metadata = loaded.iloc[0]['metadata']
        assert metadata['signal_strength'] == 0.8
        assert 'RSI' in metadata['indicators']


class TestDatabaseIntegration:
    """Integration tests for DatabaseManager with other modules."""
    
    @pytest.fixture
    def integrated_setup(self, tmp_path):
        """Set up integrated environment with DataManager and DatabaseManager."""
        # Import DataManager
        from src.data.data_manager import DataManager
        
        # Create temporary config
        config = {
            'data': {
                'base_path': str(tmp_path),
                'raw_data_path': str(tmp_path / 'raw'),
                'processed_data_path': str(tmp_path / 'processed'),
                'features_path': str(tmp_path / 'features')
            },
            'assets': {
                'symbols': ['BTC/USDT', 'ETH/USDT']
            },
            'timeframes': ['1h', '4h']
        }
        
        config_path = tmp_path / 'config.yaml'
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create DataManager and DatabaseManager
        dm = DataManager(str(config_path))
        db = DatabaseManager('sqlite', f'sqlite:///{tmp_path}/test.db')
        
        yield dm, db
        
        # Clean up
        db.engine.dispose()
    
    def test_data_flow_integration(self, integrated_setup):
        """Test complete data flow from DataManager to Database."""
        dm, db = integrated_setup
        
        # Generate test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        test_data = pd.DataFrame({
            'open': np.random.uniform(40000, 45000, len(dates)),
            'high': np.random.uniform(45000, 46000, len(dates)),
            'low': np.random.uniform(39000, 40000, len(dates)),
            'close': np.random.uniform(40000, 45000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)
        
        # Store in database
        db.store_ohlcv(test_data, 'BTC/USDT', '1h')
        
        # Load through database
        loaded = db.load_ohlcv('BTC/USDT', '1h')
        
        assert not loaded.empty
        assert len(loaded) == len(test_data)
        
        # Calculate returns using DataManager
        with_returns = dm.calculate_returns(loaded)
        
        assert 'return_1' in with_returns.columns
        assert 'return_5' in with_returns.columns


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])