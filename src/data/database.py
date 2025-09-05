"""
Database Module

This module handles database operations for persistent storage of OHLCV data,
features, trades, and performance metrics. Supports both SQLite and PostgreSQL.

Author: Trading Bot System
Date: 2024
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
import json

# Set up logging
logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()


class OHLCVData(Base):
    """Table for storing OHLCV data."""
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Create composite index for faster queries
    __table_args__ = (
        Index('idx_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )


class Features(Base):
    """Table for storing calculated features."""
    __tablename__ = 'features'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_set_name = Column(String(100), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    features_json = Column(Text, nullable=False)  # JSON string of features
    
    __table_args__ = (
        Index('idx_feature_set_symbol_timestamp', 'feature_set_name', 'symbol', 'timestamp'),
    )


class Trades(Base):
    """Table for storing trade history."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # buy/sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    fee = Column(Float)
    pnl = Column(Float)
    strategy = Column(String(50))
    is_paper = Column(Boolean, default=True)
    metadata_json = Column(Text)  # Additional metadata as JSON
    
    __table_args__ = (
        Index('idx_trades_timestamp', 'timestamp'),
        Index('idx_trades_symbol', 'symbol'),
    )


class Performance(Base):
    """Table for storing performance metrics."""
    __tablename__ = 'performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    strategy = Column(String(50), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    period = Column(String(20))  # daily, weekly, monthly
    is_paper = Column(Boolean, default=True)
    
    __table_args__ = (
        Index('idx_performance_strategy_timestamp', 'strategy', 'timestamp'),
    )


class DatabaseManager:
    """
    Database manager for handling all database operations.
    
    Supports both SQLite and PostgreSQL backends.
    """
    
    def __init__(self, db_type: str = 'sqlite', connection_string: Optional[str] = None):
        """
        Initialize the DatabaseManager.
        
        Args:
            db_type: Type of database ('sqlite' or 'postgresql')
            connection_string: Database connection string (if None, uses default)
        """
        self.db_type = db_type
        
        if connection_string:
            self.connection_string = connection_string
        else:
            if db_type == 'sqlite':
                db_path = './data/trading_bot.db'
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                self.connection_string = f'sqlite:///{db_path}'
            else:
                # PostgreSQL default connection
                self.connection_string = 'postgresql://trader:password@localhost:5432/trading_bot'
        
        # Create engine and session
        self.engine = create_engine(self.connection_string, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables
        self.create_tables()
        
        logger.info(f"DatabaseManager initialized with {db_type} backend")
    
    def create_tables(self) -> None:
        """Create all database tables if they don't exist."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def store_ohlcv(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Store OHLCV data in the database.
        
        Args:
            df: DataFrame with OHLCV data (must have datetime index)
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
        """
        session = self.get_session()
        try:
            # Prepare data for insertion
            records = []
            for timestamp, row in df.iterrows():
                # Check if record already exists
                existing = session.query(OHLCVData).filter_by(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = row['open']
                    existing.high = row['high']
                    existing.low = row['low']
                    existing.close = row['close']
                    existing.volume = row['volume']
                else:
                    # Create new record
                    record = OHLCVData(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    records.append(record)
            
            # Bulk insert new records
            if records:
                session.bulk_save_objects(records)
            
            session.commit()
            logger.info(f"Stored {len(df)} OHLCV records for {symbol} {timeframe}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing OHLCV data: {e}")
            raise
        finally:
            session.close()
    
    def load_ohlcv(self, symbol: str, timeframe: str,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load OHLCV data from the database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe of the data
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with OHLCV data
        """
        session = self.get_session()
        try:
            query = session.query(OHLCVData).filter_by(
                symbol=symbol,
                timeframe=timeframe
            )
            
            if start_date:
                query = query.filter(OHLCVData.timestamp >= start_date)
            if end_date:
                query = query.filter(OHLCVData.timestamp <= end_date)
            
            query = query.order_by(OHLCVData.timestamp)
            
            # Convert to DataFrame
            data = []
            for record in query.all():
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
            
            if data:
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading OHLCV data: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_features(self, df: pd.DataFrame, feature_set_name: str,
                      symbol: str, timeframe: str) -> None:
        """
        Store calculated features in the database.
        
        Args:
            df: DataFrame with features (must have datetime index)
            feature_set_name: Name of the feature set
            symbol: Trading pair symbol
            timeframe: Timeframe of the features
        """
        session = self.get_session()
        try:
            records = []
            for timestamp, row in df.iterrows():
                # Convert row to JSON
                features_json = row.to_json()
                
                # Check if record exists
                existing = session.query(Features).filter_by(
                    feature_set_name=feature_set_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp
                ).first()
                
                if existing:
                    existing.features_json = features_json
                else:
                    record = Features(
                        feature_set_name=feature_set_name,
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=timestamp,
                        features_json=features_json
                    )
                    records.append(record)
            
            if records:
                session.bulk_save_objects(records)
            
            session.commit()
            logger.info(f"Stored {len(df)} feature records for {feature_set_name}")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing features: {e}")
            raise
        finally:
            session.close()
    
    def load_features(self, feature_set_name: str, symbol: str, timeframe: str,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load features from the database.
        
        Args:
            feature_set_name: Name of the feature set
            symbol: Trading pair symbol
            timeframe: Timeframe of the features
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with features
        """
        session = self.get_session()
        try:
            query = session.query(Features).filter_by(
                feature_set_name=feature_set_name,
                symbol=symbol,
                timeframe=timeframe
            )
            
            if start_date:
                query = query.filter(Features.timestamp >= start_date)
            if end_date:
                query = query.filter(Features.timestamp <= end_date)
            
            query = query.order_by(Features.timestamp)
            
            # Convert to DataFrame
            data = []
            timestamps = []
            for record in query.all():
                features_dict = json.loads(record.features_json)
                data.append(features_dict)
                timestamps.append(record.timestamp)
            
            if data:
                df = pd.DataFrame(data)
                df.index = pd.DatetimeIndex(timestamps)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def store_trades(self, trades_df: pd.DataFrame) -> None:
        """
        Store trade records in the database.
        
        Args:
            trades_df: DataFrame with trade data
        """
        session = self.get_session()
        try:
            for _, trade in trades_df.iterrows():
                # Generate trade ID if not present
                trade_id = trade.get('trade_id', f"trade_{datetime.now().timestamp()}")
                
                record = Trades(
                    trade_id=trade_id,
                    timestamp=trade.get('timestamp', datetime.now()),
                    symbol=trade['symbol'],
                    side=trade['side'],
                    quantity=trade['quantity'],
                    price=trade['price'],
                    fee=trade.get('fee', 0),
                    pnl=trade.get('pnl', 0),
                    strategy=trade.get('strategy', 'unknown'),
                    is_paper=trade.get('is_paper', True),
                    metadata_json=json.dumps(trade.get('metadata', {}))
                )
                session.add(record)
            
            session.commit()
            logger.info(f"Stored {len(trades_df)} trade records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing trades: {e}")
            raise
        finally:
            session.close()
    
    def load_trades(self, symbol: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   strategy: Optional[str] = None) -> pd.DataFrame:
        """
        Load trade records from the database.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            strategy: Filter by strategy
            
        Returns:
            DataFrame with trade records
        """
        session = self.get_session()
        try:
            query = session.query(Trades)
            
            if symbol:
                query = query.filter(Trades.symbol == symbol)
            if start_date:
                query = query.filter(Trades.timestamp >= start_date)
            if end_date:
                query = query.filter(Trades.timestamp <= end_date)
            if strategy:
                query = query.filter(Trades.strategy == strategy)
            
            query = query.order_by(Trades.timestamp)
            
            # Convert to DataFrame
            data = []
            for trade in query.all():
                data.append({
                    'trade_id': trade.trade_id,
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'fee': trade.fee,
                    'pnl': trade.pnl,
                    'strategy': trade.strategy,
                    'is_paper': trade.is_paper,
                    'metadata': json.loads(trade.metadata_json) if trade.metadata_json else {}
                })
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """
        Get the latest timestamp for a symbol/timeframe combination.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            Latest timestamp or None if no data exists
        """
        session = self.get_session()
        try:
            result = session.query(OHLCVData.timestamp).filter_by(
                symbol=symbol,
                timeframe=timeframe
            ).order_by(OHLCVData.timestamp.desc()).first()
            
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {e}")
            return None
        finally:
            session.close()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> None:
        """
        Remove old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        session = self.get_session()
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old OHLCV data
            deleted_ohlcv = session.query(OHLCVData).filter(
                OHLCVData.timestamp < cutoff_date
            ).delete()
            
            # Delete old features
            deleted_features = session.query(Features).filter(
                Features.timestamp < cutoff_date
            ).delete()
            
            # Delete old trades
            deleted_trades = session.query(Trades).filter(
                Trades.timestamp < cutoff_date
            ).delete()
            
            # Delete old performance metrics
            deleted_perf = session.query(Performance).filter(
                Performance.timestamp < cutoff_date
            ).delete()
            
            session.commit()
            
            logger.info(f"Cleaned up old data: {deleted_ohlcv} OHLCV, "
                       f"{deleted_features} features, {deleted_trades} trades, "
                       f"{deleted_perf} performance records")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old data: {e}")
            raise
        finally:
            session.close()
    
    def store_performance_metric(self, strategy: str, metric_name: str,
                                metric_value: float, period: str = 'daily',
                                is_paper: bool = True) -> None:
        """
        Store a performance metric.
        
        Args:
            strategy: Strategy name
            metric_name: Name of the metric
            metric_value: Value of the metric
            period: Time period for the metric
            is_paper: Whether this is paper trading
        """
        session = self.get_session()
        try:
            record = Performance(
                timestamp=datetime.now(),
                strategy=strategy,
                metric_name=metric_name,
                metric_value=metric_value,
                period=period,
                is_paper=is_paper
            )
            session.add(record)
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing performance metric: {e}")
            raise
        finally:
            session.close()
    
    def get_performance_metrics(self, strategy: str,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy: Strategy name
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame with performance metrics
        """
        session = self.get_session()
        try:
            query = session.query(Performance).filter_by(strategy=strategy)
            
            if start_date:
                query = query.filter(Performance.timestamp >= start_date)
            if end_date:
                query = query.filter(Performance.timestamp <= end_date)
            
            query = query.order_by(Performance.timestamp)
            
            data = []
            for metric in query.all():
                data.append({
                    'timestamp': metric.timestamp,
                    'metric_name': metric.metric_name,
                    'metric_value': metric.metric_value,
                    'period': metric.period,
                    'is_paper': metric.is_paper
                })
            
            if data:
                df = pd.DataFrame(data)
                return df.pivot(index='timestamp', columns='metric_name', values='metric_value')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        session = self.get_session()
        try:
            stats = {
                'ohlcv_records': session.query(OHLCVData).count(),
                'feature_records': session.query(Features).count(),
                'trade_records': session.query(Trades).count(),
                'performance_records': session.query(Performance).count(),
                'symbols': session.query(OHLCVData.symbol).distinct().count(),
                'timeframes': session.query(OHLCVData.timeframe).distinct().count(),
            }
            
            # Get date ranges
            earliest_ohlcv = session.query(OHLCVData.timestamp).order_by(
                OHLCVData.timestamp
            ).first()
            latest_ohlcv = session.query(OHLCVData.timestamp).order_by(
                OHLCVData.timestamp.desc()
            ).first()
            
            if earliest_ohlcv and latest_ohlcv:
                stats['date_range'] = {
                    'start': earliest_ohlcv[0],
                    'end': latest_ohlcv[0]
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
        finally:
            session.close()


# Example usage and testing
if __name__ == "__main__":
    # Initialize DatabaseManager
    db = DatabaseManager(db_type='sqlite')
    
    # Get database stats
    print("\n=== Database Statistics ===")
    stats = db.get_database_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Create sample OHLCV data
    print("\n=== Storing Sample OHLCV Data ===")
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(45000, 46000, len(dates)),
        'low': np.random.uniform(39000, 40000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Store the data
    db.store_ohlcv(sample_data, 'BTC/USDT', '1h')
    
    # Load it back
    loaded_data = db.load_ohlcv('BTC/USDT', '1h')
    print(f"Loaded {len(loaded_data)} rows")
    
    # Store sample trades
    print("\n=== Storing Sample Trades ===")
    trades = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1D'),
        'symbol': ['BTC/USDT'] * 5,
        'side': ['buy', 'sell', 'buy', 'sell', 'buy'],
        'quantity': [0.1, 0.1, 0.2, 0.2, 0.15],
        'price': [42000, 43000, 41500, 44000, 43500],
        'pnl': [0, 100, 0, 500, 0],
        'strategy': 'test_strategy'
    })
    
    db.store_trades(trades)
    
    # Load trades back
    loaded_trades = db.load_trades(strategy='test_strategy')
    print(f"Loaded {len(loaded_trades)} trades")
    
    # Store performance metrics
    print("\n=== Storing Performance Metrics ===")
    db.store_performance_metric('test_strategy', 'sharpe_ratio', 1.5)
    db.store_performance_metric('test_strategy', 'total_return', 0.15)
    db.store_performance_metric('test_strategy', 'max_drawdown', -0.08)
    
    # Get performance metrics
    perf_metrics = db.get_performance_metrics('test_strategy')
    print(f"Performance metrics shape: {perf_metrics.shape}")
    
    print("\n=== DatabaseManager test complete ===")