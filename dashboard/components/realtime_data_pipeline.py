"""
Real-Time Data Pipeline Manager
Continuous WebSocket data collection with quality monitoring and auto-recovery
"""

import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class DataHealth:
    """Data health status for a symbol/timeframe pair"""
    symbol: str
    timeframe: str
    last_update: datetime
    gap_count: int
    missing_points: int
    outliers: int
    quality_score: float  # 0-100
    status: str  # 'healthy', 'warning', 'critical'
    is_streaming: bool
    candles_received: int
    last_gap_check: datetime
    

class RealTimeDataPipeline:
    """
    Manages real-time data collection, storage, and quality monitoring
    Integrates WebSocket feeds with database storage and health monitoring
    """
    
    def __init__(self, kraken_connector, data_manager, validator):
        """
        Initialize the real-time data pipeline
        
        Args:
            kraken_connector: KrakenConnector instance
            data_manager: DataManager instance
            validator: DataValidator instance
        """
        self.kraken = kraken_connector
        self.data_manager = data_manager
        self.validator = validator
        
        # Threading components
        self.pipeline_running = False
        self.threads = {}
        self.data_queue = queue.Queue(maxsize=1000)
        self.health_status = {}
        
        # Monitoring intervals
        self.gap_check_interval = 60  # Check for gaps every minute
        self.quality_check_interval = 300  # Full quality check every 5 minutes
        self.health_update_interval = 10  # Update health status every 10 seconds
        
        # Tracked assets and timeframes
        self.tracked_pairs = self.kraken.ALL_PAIRS
        self.tracked_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        
        # WebSocket data buffers
        self.data_buffers = defaultdict(list)
        self.buffer_size = 100  # Batch size before saving
        
        # Health monitoring
        self.health_callbacks = []
        self.last_health_broadcast = datetime.now()
        
        # Statistics
        self.stats = {
            'total_candles_processed': 0,
            'gaps_filled': 0,
            'errors_recovered': 0,
            'quality_fixes': 0,
            'uptime_start': None
        }
        
        logger.info("Real-time data pipeline initialized")
    
    # ================== MAIN CONTROL ==================
    
    def start_pipeline(self, pairs: Optional[List[str]] = None):
        """
        Start the complete real-time data pipeline
        
        Args:
            pairs: List of pairs to track (None = all pairs)
        """
        if self.pipeline_running:
            logger.warning("Pipeline already running")
            return
        
        self.pipeline_running = True
        self.stats['uptime_start'] = datetime.now()
        
        if pairs is None:
            pairs = self.tracked_pairs[:8]  # Start with first 8 to avoid overwhelming
        
        logger.info(f"Starting pipeline for {len(pairs)} pairs")
        
        # Initialize health status for all pairs
        self._initialize_health_status(pairs)
        
        # Start WebSocket connections
        self._start_websocket_feeds(pairs)
        
        # Start processing threads
        self.threads['processor'] = threading.Thread(target=self._data_processor_loop, daemon=True)
        self.threads['gap_monitor'] = threading.Thread(target=self._gap_monitor_loop, daemon=True)
        self.threads['quality_monitor'] = threading.Thread(target=self._quality_monitor_loop, daemon=True)
        self.threads['health_reporter'] = threading.Thread(target=self._health_reporter_loop, daemon=True)
        
        for thread in self.threads.values():
            thread.start()
        
        logger.info("Real-time pipeline started successfully")
        return True
    
    def stop_pipeline(self):
        """Stop the real-time data pipeline"""
        logger.info("Stopping real-time pipeline...")
        self.pipeline_running = False
        
        # Flush remaining buffers
        self._flush_all_buffers()
        
        # Stop WebSocket
        if self.kraken.ws:
            self.kraken.ws.close()
        
        # Wait for threads to complete
        for name, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
                logger.info(f"Stopped {name} thread")
        
        logger.info("Pipeline stopped")
    
    # ================== WEBSOCKET MANAGEMENT ==================
    
    def _start_websocket_feeds(self, pairs: List[str]):
        """Start WebSocket feeds for specified pairs"""
        
        # Register callbacks for data processing
        self.kraken.register_price_callback(self._on_price_update)
        
        # Custom OHLC callback for candle data
        def ohlc_callback(pair: str, ohlc_data: Dict):
            self._on_ohlc_update(pair, ohlc_data)
        
        # Connect WebSocket with required channels
        self.kraken.connect_websocket(
            pairs=pairs,
            channels=['ticker', 'ohlc', 'trade']
        )
        
        # Override the WebSocket message handler to capture OHLC
        original_on_message = self.kraken.ws.on_message
        
        def enhanced_on_message(ws, message):
            try:
                data = json.loads(message)
                
                # Process OHLC updates
                if isinstance(data, list) and len(data) > 1:
                    channel_name = data[-2] if len(data) > 2 else None
                    
                    if channel_name and 'ohlc' in channel_name:
                        pair = channel_name.split('-')[-1]
                        ohlc_data = data[1]
                        self._on_ohlc_update(pair, ohlc_data)
                
                # Call original handler
                original_on_message(ws, message)
                
            except Exception as e:
                logger.error(f"Error in enhanced message handler: {e}")
        
        self.kraken.ws.on_message = enhanced_on_message
        
        logger.info(f"WebSocket feeds started for {len(pairs)} pairs")
    
    def _on_price_update(self, price_data: Dict):
        """Handle real-time price updates"""
        try:
            # Add to processing queue
            self.data_queue.put({
                'type': 'price',
                'data': price_data,
                'timestamp': datetime.now()
            })
        except queue.Full:
            logger.warning("Data queue full, dropping price update")
    
    def _on_ohlc_update(self, pair: str, ohlc_data: Any):
        """Handle real-time OHLC updates"""
        try:
            # Parse OHLC data based on Kraken format
            if isinstance(ohlc_data, list) and len(ohlc_data) >= 6:
                candle = {
                    'timestamp': datetime.fromtimestamp(float(ohlc_data[0])),
                    'open': float(ohlc_data[1]),
                    'high': float(ohlc_data[2]),
                    'low': float(ohlc_data[3]),
                    'close': float(ohlc_data[4]),
                    'volume': float(ohlc_data[6]) if len(ohlc_data) > 6 else 0
                }
                
                # Add to buffer
                buffer_key = f"{pair}_1m"  # WebSocket usually sends 1m candles
                self.data_buffers[buffer_key].append(candle)
                
                # Update health status
                if pair in self.health_status:
                    for tf in self.health_status[pair]:
                        self.health_status[pair][tf].candles_received += 1
                        self.health_status[pair][tf].last_update = datetime.now()
                
                # Flush buffer if full
                if len(self.data_buffers[buffer_key]) >= self.buffer_size:
                    self._flush_buffer(pair, '1m')
                
        except Exception as e:
            logger.error(f"Error processing OHLC update: {e}")
    
    # ================== DATA PROCESSING ==================
    
    def _data_processor_loop(self):
        """Main loop for processing incoming data"""
        logger.info("Data processor started")
        
        while self.pipeline_running:
            try:
                # Get data from queue with timeout
                item = self.data_queue.get(timeout=1)
                
                if item['type'] == 'price':
                    self._process_price_update(item['data'])
                elif item['type'] == 'ohlc':
                    self._process_ohlc_update(item['data'])
                
                self.stats['total_candles_processed'] += 1
                
            except queue.Empty:
                # Periodic buffer flush
                self._flush_old_buffers()
            except Exception as e:
                logger.error(f"Error in data processor: {e}")
                time.sleep(1)
    
    def _flush_buffer(self, pair: str, timeframe: str):
        """Flush data buffer to database"""
        buffer_key = f"{pair}_{timeframe}"
        
        if buffer_key not in self.data_buffers or not self.data_buffers[buffer_key]:
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.data_buffers[buffer_key])
            df.set_index('timestamp', inplace=True)
            
            # Update database
            self.data_manager.update_data(pair, timeframe, df)
            
            # Clear buffer
            self.data_buffers[buffer_key] = []
            
            logger.debug(f"Flushed {len(df)} candles for {pair} {timeframe}")
            
        except Exception as e:
            logger.error(f"Error flushing buffer for {pair} {timeframe}: {e}")
    
    def _flush_old_buffers(self):
        """Flush buffers that haven't been updated recently"""
        current_time = datetime.now()
        
        for buffer_key in list(self.data_buffers.keys()):
            if self.data_buffers[buffer_key]:
                # Get last candle timestamp
                last_candle = self.data_buffers[buffer_key][-1]
                if 'timestamp' in last_candle:
                    age = (current_time - last_candle['timestamp']).seconds
                    
                    # Flush if older than 30 seconds
                    if age > 30:
                        pair, timeframe = buffer_key.rsplit('_', 1)
                        self._flush_buffer(pair, timeframe)
    
    def _flush_all_buffers(self):
        """Flush all data buffers"""
        for buffer_key in list(self.data_buffers.keys()):
            if '_' in buffer_key:
                pair, timeframe = buffer_key.rsplit('_', 1)
                self._flush_buffer(pair, timeframe)
    
    # ================== GAP MONITORING ==================
    
    def _gap_monitor_loop(self):
        """Monitor and fill data gaps"""
        logger.info("Gap monitor started")
        
        while self.pipeline_running:
            try:
                for pair in self.tracked_pairs:
                    for timeframe in self.tracked_timeframes:
                        # Check and fill gaps
                        gaps_filled = self._check_and_fill_gaps(pair, timeframe)
                        
                        if gaps_filled > 0:
                            self.stats['gaps_filled'] += gaps_filled
                            logger.info(f"Filled {gaps_filled} gaps for {pair} {timeframe}")
                        
                        # Update health status
                        self._update_gap_status(pair, timeframe)
                
                # Wait before next check
                time.sleep(self.gap_check_interval)
                
            except Exception as e:
                logger.error(f"Error in gap monitor: {e}")
                time.sleep(10)
    
    def _check_and_fill_gaps(self, pair: str, timeframe: str) -> int:
        """Check for gaps and fill them"""
        try:
            # Load current data
            df = self.data_manager.load_existing_data(pair, timeframe)
            
            if df.empty:
                # No data, fetch initial data
                self.kraken.check_and_fill_gap(pair, timeframe)
                return 1
            
            # Check for gaps
            time_diff = df.index.to_series().diff()
            
            # Expected time difference based on timeframe
            expected_diff = self._get_expected_time_diff(timeframe)
            
            # Find gaps (where time difference is more than expected)
            gaps = time_diff[time_diff > expected_diff * 1.5]
            
            if len(gaps) > 0:
                # Fill gaps using Kraken connector
                for gap_start in gaps.index:
                    gap_end = df.index[df.index.get_loc(gap_start) - 1]
                    
                    # Fetch missing data
                    success = self.kraken.check_and_fill_gap(
                        pair, 
                        timeframe,
                        start_time=gap_end,
                        end_time=gap_start
                    )
                    
                    if success:
                        logger.info(f"Filled gap for {pair} {timeframe}: {gap_end} to {gap_start}")
                
                return len(gaps)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error checking gaps for {pair} {timeframe}: {e}")
            return 0
    
    def _get_expected_time_diff(self, timeframe: str) -> timedelta:
        """Get expected time difference between candles"""
        mapping = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return mapping.get(timeframe, timedelta(hours=1))
    
    # ================== QUALITY MONITORING ==================
    
    def _quality_monitor_loop(self):
        """Monitor data quality and auto-fix issues"""
        logger.info("Quality monitor started")
        
        while self.pipeline_running:
            try:
                for pair in self.tracked_pairs:
                    for timeframe in self.tracked_timeframes:
                        # Run quality check
                        quality_score = self._check_data_quality(pair, timeframe)
                        
                        # Auto-fix if needed
                        if quality_score < 80:
                            fixes_applied = self._auto_fix_quality_issues(pair, timeframe)
                            if fixes_applied:
                                self.stats['quality_fixes'] += fixes_applied
                
                # Wait before next check
                time.sleep(self.quality_check_interval)
                
            except Exception as e:
                logger.error(f"Error in quality monitor: {e}")
                time.sleep(30)
    
    def _check_data_quality(self, pair: str, timeframe: str) -> float:
        """Check data quality and return score (0-100)"""
        try:
            # Load data
            df = self.data_manager.load_existing_data(pair, timeframe)
            
            if df.empty:
                return 0.0
            
            # Run validation
            validation_result = self.validator.validate_ohlcv(df, symbol=pair, timeframe=timeframe)
            
            # Calculate quality score
            score = 100.0
            
            # Deduct for missing data
            if validation_result.missing_data_count > 0:
                score -= min(20, validation_result.missing_data_count * 2)
            
            # Deduct for gaps
            if validation_result.gaps_found > 0:
                score -= min(30, validation_result.gaps_found * 5)
            
            # Deduct for outliers
            if validation_result.outliers_count > 0:
                score -= min(20, validation_result.outliers_count)
            
            # Deduct for validation errors
            if validation_result.validation_errors:
                score -= len(validation_result.validation_errors) * 10
            
            # Update health status
            if pair in self.health_status and timeframe in self.health_status[pair]:
                health = self.health_status[pair][timeframe]
                health.quality_score = max(0, score)
                health.missing_points = validation_result.missing_data_count
                health.gap_count = validation_result.gaps_found
                health.outliers = validation_result.outliers_count
                
                # Set status based on score
                if score >= 90:
                    health.status = 'healthy'
                elif score >= 70:
                    health.status = 'warning'
                else:
                    health.status = 'critical'
            
            return max(0, score)
            
        except Exception as e:
            logger.error(f"Error checking quality for {pair} {timeframe}: {e}")
            return 0.0
    
    def _auto_fix_quality_issues(self, pair: str, timeframe: str) -> int:
        """Automatically fix quality issues"""
        try:
            # Load data
            df = self.data_manager.load_existing_data(pair, timeframe)
            
            if df.empty:
                return 0
            
            # Run validation with auto-fix
            result = self.validator.validate_and_fix(
                df,
                symbol=pair,
                timeframe=timeframe,
                auto_fix=True
            )
            
            if result.fixed_df is not None:
                # Save fixed data
                self.data_manager.update_data(pair, timeframe, result.fixed_df)
                
                fixes = len(result.fixes_applied) if hasattr(result, 'fixes_applied') else 1
                logger.info(f"Applied {fixes} fixes to {pair} {timeframe}")
                return fixes
            
            return 0
            
        except Exception as e:
            logger.error(f"Error auto-fixing {pair} {timeframe}: {e}")
            return 0
    
    # ================== HEALTH REPORTING ==================
    
    def _health_reporter_loop(self):
        """Report health status to UI"""
        logger.info("Health reporter started")
        
        while self.pipeline_running:
            try:
                # Generate health report
                health_report = self.get_overall_health()
                
                # Broadcast to registered callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(health_report)
                    except Exception as e:
                        logger.error(f"Error in health callback: {e}")
                
                self.last_health_broadcast = datetime.now()
                
                # Wait before next report
                time.sleep(self.health_update_interval)
                
            except Exception as e:
                logger.error(f"Error in health reporter: {e}")
                time.sleep(10)
    
    def _initialize_health_status(self, pairs: List[str]):
        """Initialize health status for pairs"""
        for pair in pairs:
            self.health_status[pair] = {}
            for timeframe in self.tracked_timeframes:
                self.health_status[pair][timeframe] = DataHealth(
                    symbol=pair,
                    timeframe=timeframe,
                    last_update=datetime.now(),
                    gap_count=0,
                    missing_points=0,
                    outliers=0,
                    quality_score=100.0,
                    status='healthy',
                    is_streaming=False,
                    candles_received=0,
                    last_gap_check=datetime.now()
                )
    
    def _update_gap_status(self, pair: str, timeframe: str):
        """Update gap status in health monitoring"""
        if pair in self.health_status and timeframe in self.health_status[pair]:
            self.health_status[pair][timeframe].last_gap_check = datetime.now()
    
    # ================== PUBLIC INTERFACE ==================
    
    def register_health_callback(self, callback: Callable):
        """Register a callback for health updates"""
        self.health_callbacks.append(callback)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_pairs = len(self.health_status)
        healthy_count = 0
        warning_count = 0
        critical_count = 0
        
        pair_summaries = {}
        
        for pair, timeframes in self.health_status.items():
            pair_health = {
                'status': 'healthy',
                'quality_avg': 0,
                'streaming': False,
                'last_update': None,
                'issues': []
            }
            
            quality_scores = []
            
            for tf, health in timeframes.items():
                quality_scores.append(health.quality_score)
                
                if health.status == 'critical':
                    pair_health['status'] = 'critical'
                elif health.status == 'warning' and pair_health['status'] != 'critical':
                    pair_health['status'] = 'warning'
                
                if health.is_streaming:
                    pair_health['streaming'] = True
                
                if pair_health['last_update'] is None or health.last_update > pair_health['last_update']:
                    pair_health['last_update'] = health.last_update
                
                # Collect issues
                if health.gap_count > 0:
                    pair_health['issues'].append(f"{health.gap_count} gaps in {tf}")
                if health.outliers > 0:
                    pair_health['issues'].append(f"{health.outliers} outliers in {tf}")
            
            pair_health['quality_avg'] = np.mean(quality_scores) if quality_scores else 0
            
            # Count status
            if pair_health['status'] == 'healthy':
                healthy_count += 1
            elif pair_health['status'] == 'warning':
                warning_count += 1
            else:
                critical_count += 1
            
            pair_summaries[pair] = pair_health
        
        # Calculate overall score
        overall_score = (healthy_count * 100) / total_pairs if total_pairs > 0 else 0
        
        # System uptime
        uptime = None
        if self.stats['uptime_start']:
            uptime = (datetime.now() - self.stats['uptime_start']).total_seconds()
        
        return {
            'overall_status': 'healthy' if overall_score >= 80 else 'warning' if overall_score >= 60 else 'critical',
            'overall_score': overall_score,
            'healthy_pairs': healthy_count,
            'warning_pairs': warning_count,
            'critical_pairs': critical_count,
            'total_pairs': total_pairs,
            'pairs': pair_summaries,
            'stats': {
                'total_candles': self.stats['total_candles_processed'],
                'gaps_filled': self.stats['gaps_filled'],
                'quality_fixes': self.stats['quality_fixes'],
                'errors_recovered': self.stats['errors_recovered'],
                'uptime_seconds': uptime
            },
            'pipeline_running': self.pipeline_running,
            'websocket_connected': self.kraken.ws_running if hasattr(self.kraken, 'ws_running') else False,
            'last_health_update': self.last_health_broadcast,
            'timestamp': datetime.now()
        }
    
    def get_pair_health(self, pair: str) -> Dict[str, DataHealth]:
        """Get health status for a specific pair"""
        return self.health_status.get(pair, {})
    
    def get_timeframe_health(self, pair: str, timeframe: str) -> Optional[DataHealth]:
        """Get health status for a specific pair/timeframe"""
        if pair in self.health_status and timeframe in self.health_status[pair]:
            return self.health_status[pair][timeframe]
        return None
    
    def force_quality_check(self, pair: str, timeframe: str) -> float:
        """Force an immediate quality check"""
        return self._check_data_quality(pair, timeframe)
    
    def force_gap_fill(self, pair: str, timeframe: str) -> int:
        """Force immediate gap filling"""
        return self._check_and_fill_gaps(pair, timeframe)


# ================== USAGE EXAMPLE ==================

if __name__ == "__main__":
    # This would be integrated into your main app
    from src.data.kraken_connector import KrakenConnector
    from src.data.data_manager import DataManager
    from src.data.validator import DataValidator
    
    # Initialize components
    kraken = KrakenConnector(mode='paper')
    data_manager = DataManager()
    validator = DataValidator()
    
    # Create pipeline
    pipeline = RealTimeDataPipeline(kraken, data_manager, validator)
    
    # Register health callback for UI updates
    def health_update(health_report):
        print(f"Health Update: {health_report['overall_status']}")
        print(f"Score: {health_report['overall_score']:.1f}%")
        print(f"Candles processed: {health_report['stats']['total_candles']}")
    
    pipeline.register_health_callback(health_update)
    
    # Start pipeline
    pipeline.start_pipeline(pairs=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'])
    
    try:
        # Run for a while
        time.sleep(60)
    finally:
        # Stop pipeline
        pipeline.stop_pipeline()