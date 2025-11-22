"""
Live Trading System - $100 on Kraken
Real trading with trained ML/RL models

✅ ALL CRITICAL FIXES APPLIED:
1. Added current_price to all place_order() calls
2. Added Binance WebSocket cleanup in stop()
3. Removed dry_run checks from order execution
4. Set dry_run=False (no paper trading)
5. Fixed ternary operator to 'live' mode
6. Added confirmation prompt
"""

import sys
import time
import json
import signal
import threading
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timedelta, timezone
from collections import deque
import pandas as pd
import numpy as np
import joblib
import logging
from dataclasses import dataclass, asdict
import torch


# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.binance_connector import BinanceDataConnector
from src.data.kraken_connector import KrakenConnector, KrakenOrder
from src.features.feature_engineer import FeatureEngineer
from src.models.ml_predictor import MLPredictor
from src.models.dqn_agent import DQNAgent
from src.trading.portfolio import Portfolio
from src.trading.risk_manager import RiskManager, RiskConfig
from src.trading.position_sizer import PositionSizer
from src.trading.order_reconciliation import OrderReconciliation, verify_order_status_on_kraken


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class LiveTradingConfig:
    """Configuration for live trading"""
    # Capital settings
    initial_capital: float = 100.0
    max_position_size: float = 0.3  # Max 30% per position
    max_positions: int = 2  # Max 2 positions at once with $100
    
    # Trading symbols
    symbols: List[str] = None
    execution_timeframe: str = '5m'
    
    # Model paths
    ml_model_path: str = 'models/ml/ml_predictor.pkl'
    rl_model_path: str = 'models/rl/dqn_agent.pth'
    feature_selector_path: str = 'models/feature_engineer.pkl'
    
    # Risk settings
    max_drawdown: float = 0.15  # 15% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    stop_loss: float = 0.03  # 3% stop loss
    take_profit: float = 0.06  # 6% take profit
    
    # Kraken settings
    kraken_api_key: str = None
    kraken_api_secret: str = None
    
    # ✅ FIXED: Always live mode (no paper trading)
    dry_run: bool = False  # CHANGED from True to False
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['SOL_USDT']
        
        # ✅ NEW: Validate live mode
        if self.dry_run:
            raise ValueError(
                " PAPER TRADING DISABLED!\n"
                "This bot only trades with REAL MONEY.\n"
                "Set dry_run=False or remove this check."
            )


class LiveFeatureCalculator:
    """
    Real-time feature calculator with MULTI-TIMEFRAME SUPPORT
    Maintains rolling buffers and calculates features on each new candle
    """
    
    def __init__(self, 
                 feature_engineer: FeatureEngineer,
                 selected_features: List[str],
                 buffer_size: int = 200,
                 timeframes: List[str] = ['5m', '15m', '1h']):
        """
        Initialize calculator with multi-timeframe support
        
        Args:
            feature_engineer: Existing FeatureEngineer instance
            selected_features: List of 50 feature names from training
            buffer_size: Number of candles to keep (200 for MA200)
            timeframes: List of timeframes to track
        """
        self.feature_engineer = feature_engineer
        self.selected_features = selected_features
        self.buffer_size = buffer_size
        self.timeframes = timeframes
        
        # Buffer keyed by (symbol, timeframe) tuple
        self.candle_buffers = {}
        
        logger.info(f"Initialized LiveFeatureCalculator")
        logger.info(f"  Features: {len(selected_features)}")
        logger.info(f"  Timeframes: {timeframes}")
        logger.info(f"  Buffer size: {buffer_size} candles")
    
    def initialize_buffer(self, symbol: str, timeframe: str, historical_df: pd.DataFrame):
        """
        Initialize buffer with historical data for specific symbol+timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (5m, 15m, 1h)
            historical_df: Historical OHLCV data
        """
        buffer_key = (symbol, timeframe)
        
        if buffer_key not in self.candle_buffers:
            self.candle_buffers[buffer_key] = deque(maxlen=self.buffer_size)
        
        # Add last N candles to buffer
        for idx, row in historical_df.tail(self.buffer_size).iterrows():
            self.candle_buffers[buffer_key].append({
                'timestamp': idx,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        logger.info(f"Initialized buffer: {symbol} {timeframe} with {len(self.candle_buffers[buffer_key])} candles")
    
    def update_candle(self, symbol: str, timeframe: str, new_candle: Dict):
        """
        Add new completed candle to specific buffer WITH DUPLICATE CHECKING
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            new_candle: Dict with keys: timestamp, open, high, low, close, volume
        """
        buffer_key = (symbol, timeframe)
        
        if buffer_key not in self.candle_buffers:
            logger.warning(f"Buffer not initialized for {symbol} {timeframe}")
            return
        
        # ✅ SOLUTION 2: CHECK FOR DUPLICATES
        if len(self.candle_buffers[buffer_key]) > 0:
            last_candle = self.candle_buffers[buffer_key][-1]
            
            # Get timestamps for comparison
            last_ts = last_candle['timestamp']
            new_ts = new_candle['timestamp']
            
            # Convert to pandas datetime for reliable comparison
            if not isinstance(last_ts, pd.Timestamp):
                last_ts = pd.to_datetime(last_ts, utc=True)  # ← Add utc=True
            if not isinstance(new_ts, pd.Timestamp):
                new_ts = pd.to_datetime(new_ts, utc=True) 

            # Check for duplicate timestamp
            if last_ts == new_ts:
                logger.debug(f" Duplicate candle detected: {symbol} {timeframe} @ {new_ts}")
                logger.debug(f"   Skipping (already in buffer)")
                return  # Don't add duplicate
            
            # Check for out-of-order candle (older than last)
            if new_ts < last_ts:
                logger.warning(f" Out-of-order candle detected: {symbol} {timeframe}")
                logger.warning(f"   Last: {last_ts}, New: {new_ts}")
                logger.warning(f"   Skipping old candle")
                return  # Don't add old data
            
            # Check for large gap (> 2 candle periods)
            tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            expected_diff = tf_minutes.get(timeframe, 5)
            actual_diff = (new_ts - last_ts).total_seconds() / 60
            
            if actual_diff > (expected_diff * 2):
                logger.warning(f" Large gap detected: {symbol} {timeframe}")
                logger.warning(f"   Last: {last_ts}, New: {new_ts}")
                logger.warning(f"   Gap: {actual_diff:.1f} minutes (expected: {expected_diff})")
                logger.warning(f"   Adding anyway, but features may be affected")
                # Still add it, but warn user
        
        # ✅ ADD CANDLE (no duplicate, not out-of-order)
        self.candle_buffers[buffer_key].append(new_candle)
        logger.debug(f" Updated buffer: {symbol} {timeframe} @ {new_candle['timestamp']}")
    
    def get_current_features(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Calculate features for current state of specific symbol+timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with selected features (1 row × 50 columns)
        """
        buffer_key = (symbol, timeframe)
        
        if buffer_key not in self.candle_buffers:
            logger.error(f"Buffer not initialized for {symbol} {timeframe}")
            return None
        
        if len(self.candle_buffers[buffer_key]) < self.buffer_size:
            logger.warning(
                f"Insufficient data for {symbol} {timeframe}: "
                f"{len(self.candle_buffers[buffer_key])}/{self.buffer_size}"
            )
            return None
        
        try:
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(list(self.candle_buffers[buffer_key]))
            buffer_df.set_index('timestamp', inplace=True)
            
            # Calculate ALL features using existing engineer
            all_features = self.feature_engineer.calculate_all_features(buffer_df, symbol)
            
            # Get only latest row
            latest_features = all_features.iloc[-1:].copy()
            
            # Select only the features used in training
            selected_features_df = latest_features[self.selected_features]
            
            return selected_features_df
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def get_features_with_lookback(self, symbol: str, timeframe: str, 
                                num_rows: int = 2) -> Optional[pd.DataFrame]:
        """
        Calculate features and return last N rows (for getting previous bars)
        
        Args:
            symbol: Trading symbol (e.g., 'ETH_USDT')
            timeframe: Timeframe (e.g., '5m')
            num_rows: Number of rows to return (default 2: [T-1, T])
            
        Returns:
            DataFrame with last N rows of selected features
            
        Example:
            >>> features = calc.get_features_with_lookback('ETH_USDT', '5m', num_rows=2)
            >>> features_prev = features.iloc[0:1]   # T-1
            >>> features_current = features.iloc[1:2] # T
        """
        buffer_key = (symbol, timeframe)
        
        # Validate buffer exists
        if buffer_key not in self.candle_buffers:
            logger.error(f"Buffer not initialized for {symbol} {timeframe}")
            return None
        
        # Validate sufficient data
        if len(self.candle_buffers[buffer_key]) < self.buffer_size:
            logger.warning(
                f"Insufficient data for {symbol} {timeframe}: "
                f"{len(self.candle_buffers[buffer_key])}/{self.buffer_size}"
            )
            return None
        
        try:
            # Convert buffer to DataFrame
            buffer_df = pd.DataFrame(list(self.candle_buffers[buffer_key]))
            buffer_df.set_index('timestamp', inplace=True)
            
            # Calculate ALL features using existing engineer
            all_features = self.feature_engineer.calculate_all_features(buffer_df, symbol)
            
            # Validate we have enough rows
            if len(all_features) < num_rows:
                logger.warning(
                    f"Not enough feature history: have {len(all_features)}, need {num_rows}"
                )
                return None
            
            # Get last N rows
            last_n_features = all_features.iloc[-num_rows:].copy()
            
            # Select only the features used in training
            selected_features_df = last_n_features[self.selected_features]
            
            logger.debug(
                f" Calculated {num_rows} rows of features for {symbol} {timeframe}"
            )
            
            return selected_features_df
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol} {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def check_and_fill_runtime_gaps(self, symbol: str, timeframe: str, 
                                    binance_connector) -> int:
        """
        Check for data gaps during runtime and fill them
        
        Returns:
            Number of candles filled
        """
        buffer_key = (symbol, timeframe)
        
        if buffer_key not in self.candle_buffers:
            return 0
        
        if len(self.candle_buffers[buffer_key]) < 2:
            return 0
        
        # Get last candle in buffer
        last_candle = list(self.candle_buffers[buffer_key])[-1]
        last_timestamp = last_candle['timestamp']
        
        # Check gap size
        now = datetime.now()
        tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}[timeframe]
        gap_minutes = (now - last_timestamp).total_seconds() / 60
        
        # If gap > 2 candles, fill it
        if gap_minutes > (tf_minutes * 2):
            logger.warning(f"  Gap detected during runtime: {symbol} {timeframe}")
            logger.warning(f"   Last candle: {last_timestamp}")
            logger.warning(f"   Gap: {gap_minutes:.1f} minutes ({gap_minutes/tf_minutes:.1f} candles)")
            logger.info(f"   Fetching missing candles from Binance...")
            
            try:
                # Fetch missing candles
                start_time = int(last_timestamp.timestamp() * 1000) + 1
                missing_data = binance_connector.fetch_ohlc(
                    symbol, timeframe,
                    start_time=start_time,
                    limit=100
                )
                
                if missing_data is not None and len(missing_data) > 0:
                    # Filter out candles we already have
                    new_candles = missing_data[missing_data.index > last_timestamp]
                    
                    if len(new_candles) > 0:
                        # Add to buffer
                        for idx, row in new_candles.iterrows():
                            self.update_candle(symbol, timeframe, {
                                'timestamp': idx,
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': float(row['volume'])
                            })
                        
                        logger.info(f"    Filled gap with {len(new_candles)} candles")
                        return len(new_candles)
                    else:
                        logger.warning(f"   No new candles fetched")
                        return 0
                else:
                    logger.error(f"   Failed to fetch gap data")
                    return 0
                    
            except Exception as e:
                logger.error(f"   Error filling gap: {e}")
                return 0
        
        return 0


def build_state_for_agent(
    features_dict: Dict[str, pd.DataFrame],
    portfolio: Portfolio,
    current_price: float,
    initial_balance: float,
    symbol: str,
    position_opened_step: Optional[int] = None,  # NEW parameter
    current_step: Optional[int] = None           # NEW parameter
) -> np.ndarray:
    """
    Build complete state vector for RL agent - MATCHES TRAINING EXACTLY
    
    Training state structure (183 dimensions):
    - Market features: 150 dims (50 features × 3 timeframes)
    - Position info: 5 dims
    - Account info: 5 dims
    - Asset encoding: 5 dims
    - Timeframe encodings: 18 dims (6 × 3 timeframes)
    
    Args:
        features_dict: Dictionary with keys ['5m', '15m', '1h'], values are 50-feature DataFrames
        portfolio: Portfolio instance
        current_price: Current market price
        initial_balance: Initial capital
        symbol: Trading symbol
        
    Returns:
        183-dimensional state vector
    """
    state_parts = []
    
    # PART 1: Market features from ALL timeframes (150 dims = 50 × 3)
    timeframes = ['5m', '15m', '1h']
    for tf in timeframes:
        if tf not in features_dict:
            raise ValueError(f"Missing features for timeframe {tf}")
        
        tf_features = features_dict[tf].values.flatten()
        
        if len(tf_features) != 50:
            raise ValueError(f"Expected 50 features for {tf}, got {len(tf_features)}")
        
        state_parts.append(tf_features)
    
    # PART 2: Position information (5 dims)
    has_position = symbol in portfolio.positions
    if has_position:
        position = portfolio.positions[symbol]
        position_value = 1.0  # LONG
        entry_ratio = position.entry_price / current_price if current_price > 0 else 0.0
        size_ratio = (position.quantity * current_price) / initial_balance
        unrealized_pnl = (current_price - position.entry_price) * position.quantity
        unrealized_pnl_ratio = unrealized_pnl / initial_balance
        if position_opened_step is not None and current_step is not None:
            position_duration = current_step - position_opened_step
            duration_ratio = position_duration / 100  # Normalize
        else:
            duration_ratio = 0.0  # No position or unknown
    else:
        position_value = 0.0  # FLAT
        entry_ratio = 0.0
        size_ratio = 0.0
        unrealized_pnl_ratio = 0.0
        duration_ratio = 0.0
    
    position_info = np.array([
        position_value,
        entry_ratio,
        size_ratio,
        unrealized_pnl_ratio,
        duration_ratio
    ], dtype=np.float32)
    state_parts.append(position_info)
    
    # PART 3: Account information (5 dims)
    portfolio_value = portfolio.total_value
    total_fees = sum(t.get('fees', 0) for t in portfolio.transaction_history)
    
    account_info = np.array([
        portfolio.cash_balance / initial_balance,
        portfolio_value / initial_balance,
        portfolio.realized_pnl / initial_balance,
        unrealized_pnl_ratio if has_position else 0.0,
        total_fees / initial_balance if total_fees > 0 else 0.0
    ], dtype=np.float32)
    state_parts.append(account_info)
    
    # PART 4: Asset encoding (5 dims) - MATCHES TRAINING
    asset_map = {
        'ETH_USDT': 0,
        'SOL_USDT': 1,
        'DOT_USDT': 2,
        'AVAX_USDT': 3,
        'ADA_USDT': 4
    }
    
    asset_encoding = np.zeros(5, dtype=np.float32)
    if symbol in asset_map:
        asset_encoding[asset_map[symbol]] = 1.0
    else:
        logger.error(f"Unknown asset {symbol} - not in training set!")
        logger.error(f"Available assets: {list(asset_map.keys())}")
        raise ValueError(f"Cannot trade {symbol} - not in training set")
    
    state_parts.append(asset_encoding)
    
    # PART 5: Timeframe encodings (18 dims = 6 × 3)
    timeframe_map = {
        '1m': 0, '5m': 1, '15m': 2, '30m': 3, '1h': 4, '4h': 5
    }
    
    all_tf_encodings = []
    for tf in timeframes:
        encoding = np.zeros(6, dtype=np.float32)
        if tf in timeframe_map:
            encoding[timeframe_map[tf]] = 1.0
        all_tf_encodings.append(encoding)
    
    timeframe_encodings = np.concatenate(all_tf_encodings)
    state_parts.append(timeframe_encodings)
    
    # COMBINE ALL PARTS
    state = np.concatenate(state_parts).astype(np.float32)
    
    # Safety: Clean and validate
    state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    state = np.clip(state, -10, 10)
    
    # CRITICAL VALIDATION
    if len(state) != 183:
        raise ValueError(
            f"State dimension mismatch!\n"
            f"  Expected: 183 dims\n"
            f"  Got: {len(state)} dims\n"
            f"  Breakdown: {[len(p) for p in state_parts]}"
        )
    
    return state



class LiveTrader:
    """
    Main live trading system
    Orchestrates all components for real trading on Kraken
    """
    
    def __init__(self, config: LiveTradingConfig):
        """Initialize live trader"""
        self.config = config
        self.running = False
        self.trading_thread = None
        
        # Initialize state file
        self.state_file = Path('logs/live_trading_state.json')
        self.state_file.parent.mkdir(exist_ok=True)

        # NEW: Track position timing
        self.position_opened_at = {}  # {symbol: datetime}
        self.position_opened_step = {}  # {symbol: step_count}
        self.total_steps = 0  # Total candles processed

        # ✅ NEW: Error tracking
        self.error_counts = {symbol: 0 for symbol in config.symbols}
        self.consecutive_api_errors = 0
        self.last_successful_trade = datetime.now()
        self.connection_unhealthy_since = None
        
        # ✅ NEW: Daily tracking
        self.daily_start_value = config.initial_capital  # ← This will be updated later
        self.daily_start_time = datetime.now()
        
        # ✅ NEW: Performance tracking
        self.cycle_count = 0
        self.successful_cycles = 0
        self.failed_cycles = 0

        # ✅ Initialize reconciliation system
        self.reconciliation = OrderReconciliation()
        logger.info(" Reconciliation system initialized")
        
        # ⚠️ NOTE: Reconciliation check moved to AFTER portfolio initialization
        # (Can't reconcile without a portfolio!)
        
        logger.info(" Error tracking initialized")
        logger.info(" Daily tracking initialized")
        
        logger.info("="*80)
        logger.info("INITIALIZING LIVE TRADING SYSTEM")
        logger.info("="*80)
        logger.info(f"Capital (config): ${config.initial_capital}")
        logger.info(f"Symbols: {config.symbols}")
        logger.info(f"Mode: {'PAPER' if config.dry_run else '  LIVE (REAL MONEY)'}")
        logger.info("="*80)
        
        # ═══════════════════════════════════════════════════════════════
        # Initialize all components (Binance, Kraken, ML, RL, etc.)
        # ═══════════════════════════════════════════════════════════════
        self._initialize_components()
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ NEW: QUERY ACTUAL KRAKEN BALANCE AND SYNC PORTFOLIO
        # ═══════════════════════════════════════════════════════════════
        logger.info("\n" + "="*80)
        logger.info(" SYNCING WITH KRAKEN ACCOUNT")
        logger.info("="*80)
        
        if not self.config.dry_run:  # Live mode only
            self._sync_with_kraken()
        else:
            logger.info(" Paper trading mode - using config balance")
            logger.info(f"   Starting capital: ${self.config.initial_capital:.2f}")
        
        logger.info("="*80)
        
        # ═══════════════════════════════════════════════════════════════
        # ✅ NOW CHECK FOR UNRECONCILED ORDERS (after portfolio exists)
        # ═══════════════════════════════════════════════════════════════
        unreconciled_count = self.reconciliation.get_unreconciled_count()
        if unreconciled_count > 0:
            logger.critical("\n" + "="*80)
            logger.critical(f"  {unreconciled_count} UNRECONCILED ORDER(S) FROM PREVIOUS SESSION")
            logger.critical("="*80)
            logger.critical("Attempting automatic reconciliation...")
            
            # Now portfolio exists, so this will work
            success = self.reconciliation.attempt_reconciliation(self.portfolio)
            
            if success:
                logger.info(" All orders successfully reconciled")
            else:
                remaining = self.reconciliation.get_unreconciled_count()
                logger.critical(f" {remaining} order(s) could not be reconciled automatically")
                logger.critical("Manual intervention may be required")
                logger.critical("Review: logs/unreconciled_orders.json")
                
                # Ask user if they want to continue
                print("\n" + "!"*80)
                print("  UNRECONCILED ORDERS DETECTED")
                print("!"*80)
                print(f"  {remaining} order(s) from previous session could not be reconciled")
                print("  Review logs/unreconciled_orders.json for details")
                print("  ")
                print("  Options:")
                print("    1. Type 'CONTINUE' to start trading anyway (not recommended)")
                print("    2. Type 'EXIT' to stop and fix manually")
                print("!"*80)
                
                response = input("\nYour choice: ")
                if response.upper() != 'CONTINUE':
                    logger.info("Exiting - please reconcile manually first")
                    sys.exit(1)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_components(self):
        """Initialize all trading components"""
        try:
            # 1. Initialize Binance for DATA
            logger.info("Connecting to Binance (for market data)...")
            self.binance = BinanceDataConnector(
                data_path='./data/raw/',
                update_existing_data=True
            )
            logger.info(" Binance data connector ready")
            
            # Start Binance WebSocket for real-time data
            logger.info("Starting Binance WebSocket...")
            self.binance.connect_websocket(
                pairs=self.config.symbols,
                channels=['ticker', 'kline']
            )
            logger.info(" Binance WebSocket connected")

            logger.warning("\n" + "!"*30)
            logger.warning("  INITIALIZING KRAKEN IN LIVE MODE")
            logger.warning("  REAL MONEY WILL BE TRADED")
            logger.warning("  REAL ORDERS WILL BE PLACED")
            logger.warning("!"*30 + "\n")
            
            # 2. Initialize Kraken for TRADING
            logger.info("Connecting to Kraken (for order execution)...")
            self.kraken = KrakenConnector(
                api_key=self.config.kraken_api_key,
                api_secret=self.config.kraken_api_secret,
                mode='live',  # ✅ FIXED: Hardcoded to 'live'
                data_path='./data/raw/'
            )
            
            # ✅ Verify mode
            if self.kraken.mode != 'live':
                raise RuntimeError(" CRITICAL: Kraken not in LIVE mode!")
            
            logger.info(" Kraken trading connector ready")
            logger.info(f" Verified: Kraken mode = {self.kraken.mode}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 3. Load trained ML model (FIXED - handle dict structure)
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Loading trained ML model...")
            
            ml_dict = joblib.load(self.config.ml_model_path)
            
            # Extract components from dict
            self.ml_model = ml_dict['model']
            self.ml_scaler = ml_dict['scaler']
            self.ml_selected_features = ml_dict['selected_features']
            
            logger.info(f" ML model loaded: {type(self.ml_model).__name__}")
            logger.info(f" Scaler loaded: {type(self.ml_scaler).__name__}")
            logger.info(f" Model features: {len(self.ml_selected_features)}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 4. Load trained RL agent (FIXED - proper checkpoint loading)
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Loading trained RL agent...")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(self.config.rl_model_path, weights_only=False, map_location=device)
            
            if 'config_dict' not in checkpoint:
                raise RuntimeError(" RL checkpoint missing config_dict!")
            
            config_dict = checkpoint['config_dict']
            
            from src.models.dqn_agent import DQNConfig
            rl_config = DQNConfig(
                state_dim=int(config_dict['state_dim']),
                action_dim=int(config_dict['action_dim']),
                hidden_dims=[int(x) for x in config_dict['hidden_dims']]
            )
            
            self.rl_agent = DQNAgent(config=rl_config)
            self.rl_agent.q_network.load_state_dict(checkpoint['q_network_state'])
            self.rl_agent.device = device
            self.rl_agent.q_network = self.rl_agent.q_network.to(device)
            self.rl_agent.q_network.eval()
            
            logger.info(f" RL agent loaded: {rl_config.state_dim}-dim state")
            logger.info(f" Device: {device}")

            # Validate dimensions
            expected_state_dim = 183
            actual_state_dim = rl_config.state_dim

            if actual_state_dim != expected_state_dim:
                raise ValueError(
                    f"\n{'='*60}\n"
                    f" MODEL DIMENSION MISMATCH!\n"
                    f"{'='*60}\n"
                    f"  Expected state dimension: {expected_state_dim}\n"
                    f"  Model trained with: {actual_state_dim}\n"
                    f"\n"
                    f"  Your agent was trained with a different configuration!\n"
                    f"  Please retrain the agent with the correct state size.\n"
                    f"{'='*60}"
                )

            logger.info(f" Model validated: {actual_state_dim}-dim state")
            
            # ═══════════════════════════════════════════════════════════════════
            # 5. Set selected features (FIXED - use from ML dict)
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Loading feature configuration...")
            
            # Use features from ML model dict (already loaded above)
            self.selected_features = self.ml_selected_features
            
            # Validate feature count
            if len(self.selected_features) != 50:
                raise ValueError(
                    f" Expected 50 features, got {len(self.selected_features)}\n"
                    f"First 5: {self.selected_features[:5]}"
                )

            logger.info(f" Features validated: {len(self.selected_features)} features")
            logger.info(f"  First 5: {self.selected_features[:5]}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 6. Initialize feature engineer
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Initializing feature engineer...")
            self.feature_engineer = FeatureEngineer()
            
            # ═══════════════════════════════════════════════════════════════════
            # 7. Initialize live feature calculator
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Setting up feature calculator...")
            self.feature_calculator = LiveFeatureCalculator(
                feature_engineer=self.feature_engineer,
                selected_features=self.selected_features,
                buffer_size=250
            )

            # ═══════════════════════════════════════════════════════════════════
            # 8. Validate configured symbols
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Validating configured symbols...")
            trained_assets = ['ETH_USDT', 'SOL_USDT', 'DOT_USDT', 'AVAX_USDT', 'ADA_USDT']

            for symbol in self.config.symbols:
                if symbol not in trained_assets:
                    raise ValueError(
                        f"INVALID SYMBOL: {symbol} - not in training set.\n"
                        f"Available: {trained_assets}"
                    )

            logger.info(f" All symbols validated: {self.config.symbols}")
            
            # ═══════════════════════════════════════════════════════════════════
            # 9. Load historical data from BINANCE
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Loading historical data for feature buffers...")
            timeframes = ['5m', '15m', '1h']

            for symbol in self.config.symbols:
                logger.info(f"\nLoading data for {symbol}...")
                
                for tf in timeframes:
                    # Load existing CSV data
                    historical_data = self.binance.load_existing_data(symbol, timeframe=tf)
                    
                    if historical_data.empty:
                        logger.error(f" No historical data found for {symbol} {tf}")
                        logger.error(f"   Run: python main.py data --fetch")
                        raise RuntimeError(f"Missing historical data for {symbol} {tf}")
                    
                    # ✅ CHECK FOR GAPS AND AUTO-FILL
                    last_timestamp = historical_data.index[-1]

                    # ✅ FIXED: Make both timestamps timezone-aware (UTC)
                    if last_timestamp.tzinfo is None:
                        # CSV timestamp is naive, assume UTC
                        last_timestamp = last_timestamp.tz_localize('UTC')

                    # Get current time in UTC
                    now = datetime.now(timezone.utc)

                    # Now safe to compare
                    gap_minutes = (now - last_timestamp).total_seconds() / 60
                    
                    # Convert timeframe to minutes
                    tf_to_minutes = {
                        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                        '1h': 60, '4h': 240, '1d': 1440
                    }
                    tf_minutes = tf_to_minutes.get(tf, 5)
                    
                    # If gap is more than 2 candles, fill it
                    gap_threshold = tf_minutes * 2
                    
                    if gap_minutes > gap_threshold:
                        logger.warning(f"   Gap detected: {gap_minutes:.1f} minutes old")
                        logger.warning(f"     (Last candle: {last_timestamp.strftime('%Y-%m-%d %H:%M')})")
                        logger.info(f"     Auto-filling gap from Binance API...")
                        
                        try:
                            # Fetch missing candles
                            start_time = int(last_timestamp.timestamp() * 1000) + 1
                            new_data = self.binance.fetch_ohlc(
                                symbol, 
                                tf,
                                start_time=start_time,
                                limit=100  # Enough to fill most gaps
                            )
                            
                            if new_data is not None and len(new_data) > 0:
                                # Update CSV file with new data
                                self.binance.update_historical_data(symbol, new_data, tf)
                                
                                # Reload data with filled gap
                                historical_data = self.binance.load_existing_data(symbol, timeframe=tf)
                                
                                new_candles = len(new_data)
                                logger.info(f"       Gap filled! Added {new_candles} candle(s)")
                                
                                # Verify gap is now closed
                                new_last_ts = historical_data.index[-1]
                                new_gap_minutes = (now - new_last_ts).total_seconds() / 60
                                logger.info(f"      New gap: {new_gap_minutes:.1f} minutes (was {gap_minutes:.1f})")
                                
                            else:
                                logger.warning(f"      Could not fetch gap data from API")
                                logger.warning(f"      Starting with gap - features may be suboptimal")
                        
                        except Exception as e:
                            logger.error(f"      Error filling gap: {e}")
                            logger.warning(f"      Continuing with gap - features may be suboptimal")
                    
                    else:
                        logger.info(f"   Data is recent (gap: {gap_minutes:.1f} min)")
                    
                    # Initialize buffer with (now complete) historical data
                    self.feature_calculator.initialize_buffer(symbol, tf, historical_data)
                    logger.info(f"    Loaded {len(historical_data)} candles: {symbol} {tf}")

            # ═══════════════════════════════════════════════════════════════════
            # 10. Initialize portfolio
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Initializing portfolio...")
            self.portfolio = Portfolio(
                initial_capital=self.config.initial_capital,
                max_positions=self.config.max_positions
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 11. Initialize risk manager
            # ═══════════════════════════════════════════════════════════════════
            logger.info("Setting up risk management...")
            risk_config = RiskConfig(
                max_risk_per_trade=self.config.max_position_size,
                max_drawdown_limit=self.config.max_drawdown,
                max_daily_risk=self.config.daily_loss_limit
            )
            self.risk_manager = RiskManager(
                initial_capital=self.config.initial_capital,
                config=risk_config
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 12. Initialize position sizer
            # ═══════════════════════════════════════════════════════════════════
            self.position_sizer = PositionSizer(
                capital=self.config.initial_capital,
                max_risk_per_trade=self.config.max_position_size,
                kelly_safety_factor=0.25
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # 13. Performance tracking
            # ═══════════════════════════════════════════════════════════════════
            self.trades_history = []
            self.start_time = None
            
            logger.info(" All components initialized successfully")
            logger.info("")
            logger.info("="*60)
            logger.info(" DATA SOURCE: Binance (historical + real-time)")
            logger.info(" EXECUTION: Kraken (order placement)")
            logger.info(" MODE:  LIVE TRADING WITH REAL MONEY")
            logger.info("="*60)

            # ═══════════════════════════════════════════════════════════════════
            # 14. Test buffer quality
            # ═══════════════════════════════════════════════════════════════════
            if not self._test_buffer_quality():
                raise RuntimeError(
                    "Buffer quality check failed! "
                    "Data has duplicates or gaps that will affect trading. "
                    "Run: python quick_update.py and try again."
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _sync_with_kraken(self):
        """
        Sync portfolio with actual Kraken account balance
        
        ✅ FIXED: Properly handles Kraken's balance API format:
        - Balance API returns: {'USD': '95.23', 'SOL': '0.5'}
        - NOT trading pairs like 'SOLUSD'
        """
        logger.info(" Querying Kraken account balance...")
        
        try:
            # Get actual balance from Kraken API
            kraken_balances = self.kraken.get_account_balance()
            
            if not kraken_balances:
                logger.error(" Could not retrieve Kraken balance!")
                logger.error("   Possible causes:")
                logger.error("   - Invalid API keys")
                logger.error("   - API permissions not set (need 'Query Funds')")
                logger.error("   - Network issue")
                raise RuntimeError("Failed to get Kraken balance - cannot start live trading")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 1: Calculate total USD balance
            # ═══════════════════════════════════════════════════════════════
            actual_usd = 0.0
            
            # ✅ FIXED: Kraken returns 'USD' or 'ZUSD' for USD balance
            usd_currencies = ['USD', 'ZUSD']  # Not 'USDT' - Kraken uses USD
            
            for currency in usd_currencies:
                if currency in kraken_balances:
                    balance = kraken_balances[currency].balance
                    actual_usd += balance
                    if balance > 0:
                        logger.info(f"   {currency}: ${balance:.2f}")
            
            if actual_usd == 0:
                logger.error(" No USD balance found on Kraken!")
                logger.error("   Please deposit USD before starting live trading")
                logger.error(f"   Available currencies: {list(kraken_balances.keys())}")
                raise RuntimeError("No funds available on Kraken")
            
            logger.info(f"\n Total USD Balance: ${actual_usd:.2f}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 2: Check for existing crypto positions
            # ═══════════════════════════════════════════════════════════════
            existing_positions = {}
            
            logger.info(f"\n Checking for existing positions...")
            logger.info(f"   Kraken currencies found: {list(kraken_balances.keys())}")
            
            for currency, balance_obj in kraken_balances.items():
                # Skip USD currencies
                if currency in usd_currencies:
                    continue
                
                # Skip dust (< $0.01 worth)
                if balance_obj.balance < 0.0001:
                    continue
                
                # Map Kraken currency code to our symbol format
                symbol = self._map_kraken_currency_to_symbol(currency)
                
                if symbol:
                    # Check if this symbol is in our trading list
                    if symbol in self.config.symbols:
                        existing_positions[symbol] = balance_obj.balance
                        logger.warning(f"  Existing position: {symbol} = {balance_obj.balance:.8f} (from Kraken currency '{currency}')")
                    else:
                        logger.info(f"   Ignoring {currency} (not in trading list)")
                else:
                    logger.debug(f"   Unknown Kraken currency: {currency}")
            
            if not existing_positions:
                logger.info("    No existing positions found")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 3: Compare with config and handle mismatch
            # ═══════════════════════════════════════════════════════════════
            config_capital = self.config.initial_capital
            difference = actual_usd - config_capital
            
            if abs(difference) > 1.0:  # More than $1 difference
                logger.critical("\n" + "="*80)
                logger.critical("  BALANCE MISMATCH DETECTED")
                logger.critical("="*80)
                logger.critical(f"  Config initial_capital: ${config_capital:.2f}")
                logger.critical(f"  Actual Kraken balance:  ${actual_usd:.2f}")
                logger.critical(f"  Difference:             ${difference:+.2f}")
                logger.critical("="*80)
                logger.critical("")
                logger.critical("This bot MUST use your actual Kraken balance.")
                logger.critical("Using wrong balance = wrong position sizing = potential failures")
                logger.critical("")
                logger.critical("Options:")
                logger.critical("  1. Type 'USE_KRAKEN' - Use actual Kraken balance (recommended)")
                logger.critical("  2. Type 'EXIT' - Stop and update config manually")
                logger.critical("")
                
                response = input("Your choice: ")
                
                if response.upper() == 'USE_KRAKEN':
                    logger.info(f" Using actual Kraken balance: ${actual_usd:.2f}")
                    self.config.initial_capital = actual_usd
                    self.daily_start_value = actual_usd
                else:
                    logger.error("Exiting - please update initial_capital in your config")
                    logger.error(f"Set initial_capital={actual_usd:.2f} in main.py")
                    sys.exit(1)
            else:
                logger.info(f" Config matches Kraken (within $1)")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 4: Re-initialize portfolio with REAL balance
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"\n Re-initializing portfolio with real balance...")
            
            self.portfolio = Portfolio(
                initial_capital=self.config.initial_capital,
                max_positions=self.config.max_positions
            )
            
            logger.info(f" Portfolio initialized: ${self.config.initial_capital:.2f}")
            
            # ═══════════════════════════════════════════════════════════════
            # STEP 5: Handle existing positions (if any)
            # ═══════════════════════════════════════════════════════════════
            if existing_positions:
                logger.warning("\n" + "="*80)
                logger.warning("  EXISTING POSITIONS DETECTED ON KRAKEN")
                logger.warning("="*80)
                logger.warning("The following positions exist on your Kraken account:")
                logger.warning("")
                
                for symbol, qty in existing_positions.items():
                    logger.warning(f"  {symbol}: {qty:.8f}")
                
                logger.warning("")
                logger.warning("  IMPORTANT:")
                logger.warning("  These positions are NOT tracked by this bot!")
                logger.warning("  The bot has no knowledge of entry price, timing, etc.")
                logger.warning("")
                logger.warning("Recommended action:")
                logger.warning("  1. Close all positions manually on Kraken")
                logger.warning("  2. Restart the bot with clean slate")
                logger.warning("")
                logger.warning("Or you can continue, but:")
                logger.warning("  - Bot won't manage these positions")
                logger.warning("  - P&L calculations will be inaccurate")
                logger.warning("  - May cause unexpected behavior")
                logger.warning("="*80)
                logger.warning("")
                
                response = input("Continue anyway? Type 'YES' to continue: ")
                if response.upper() != 'YES':
                    logger.info("Exiting - please close positions on Kraken first")
                    sys.exit(1)
                else:
                    logger.warning("  Continuing with untracked positions...")
            
            logger.info("\n Kraken account sync complete")
            logger.info(f"   Trading with: ${self.config.initial_capital:.2f}")
            
        except Exception as e:
            logger.error(f"\n Error syncing with Kraken: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to sync with Kraken: {e}")
        
    def _map_kraken_currency_to_symbol(self, kraken_currency: str) -> Optional[str]:
        """
        Map Kraken BALANCE currency code to our trading symbol format
        
        Context:
        - Kraken Balance API returns: {'SOL': '0.5', 'XBT': '0.1', 'USD': '100'}
        - These are CURRENCY CODES (what you own)
        - NOT trading pairs (which are 'SOLUSD', 'XBTUSD')
        
        We need to map these currency codes to our internal symbol format:
        - 'SOL' (Kraken balance) → 'SOL_USDT' (our format)
        - 'XBT' (Kraken balance) → 'BTC_USDT' (our format)
        
        Args:
            kraken_currency: Currency code from Kraken Balance API
                            Examples: 'SOL', 'XBT', 'XETH', 'ADA'
            
        Returns:
            Our trading symbol format (e.g., 'SOL_USDT')
            or None if currency not recognized
        """
        mapping = {
            # Standard coins (Kraken balance code → Our symbol)
            'SOL': 'SOL_USDT',
            'ADA': 'ADA_USDT',
            'DOT': 'DOT_USDT',
            'AVAX': 'AVAX_USDT',
            'MATIC': 'MATIC_USDT',
            'LINK': 'LINK_USDT',
            'UNI': 'UNI_USDT',
            'ATOM': 'ATOM_USDT',
            'ALGO': 'ALGO_USDT',
            'XRP': 'XRP_USDT',
            'DOGE': 'DOGE_USDT',
            'LTC': 'LTC_USDT',
            
            # Ethereum (Kraken uses XETH prefix in balance API)
            'XETH': 'ETH_USDT',
            'ETH': 'ETH_USDT',
            
            # Bitcoin (Kraken uses XBT ticker, not BTC)
            'XXBT': 'BTC_USDT',  # With XX prefix
            'XBT': 'BTC_USDT',   # With X prefix
            'BTC': 'BTC_USDT',   # Just in case
        }
        
        return mapping.get(kraken_currency)

    def _test_buffer_quality(self):
        """
        Verify buffers are clean and continuous (no duplicates, no gaps)
        Call this after initialization to ensure data quality.
        
        Returns:
            bool: True if all buffers pass quality checks
        """
        logger.info("\n" + "="*80)
        logger.info("BUFFER QUALITY CHECK")
        logger.info("="*80)
        
        all_passed = True
        timeframes = ['5m', '15m', '1h']
        
        for symbol in self.config.symbols:
            for tf in timeframes:
                buffer_key = (symbol, tf)
                
                if buffer_key not in self.feature_calculator.candle_buffers:
                    logger.error(f" {symbol} {tf}: Buffer not found!")
                    all_passed = False
                    continue
                
                buffer = list(self.feature_calculator.candle_buffers[buffer_key])
                
                if len(buffer) < 2:
                    logger.warning(f" {symbol} {tf}: Buffer too small ({len(buffer)} candles)")
                    continue
                
                # Extract timestamps
                timestamps = [candle['timestamp'] for candle in buffer]
                
                # Convert to pandas datetime for comparison
                timestamps = [pd.to_datetime(ts) if not isinstance(ts, pd.Timestamp) else ts 
                            for ts in timestamps]
                
                # TEST 1: No duplicates
                unique_timestamps = list(set(timestamps))
                if len(timestamps) != len(unique_timestamps):
                    logger.error(f" {symbol} {tf}: DUPLICATE timestamps detected!")
                    duplicates = [ts for ts in timestamps if timestamps.count(ts) > 1]
                    logger.error(f"   Duplicates: {duplicates[:3]}")
                    all_passed = False
                    continue
                
                # TEST 2: Continuous (no large gaps)
                tf_to_minutes = {'5m': 5, '15m': 15, '1h': 60}
                expected_diff = tf_to_minutes[tf]
                
                max_gap = 0
                gap_count = 0
                
                for i in range(len(timestamps) - 1):
                    t1 = timestamps[i]
                    t2 = timestamps[i + 1]
                    diff_minutes = (t2 - t1).total_seconds() / 60
                    
                    # Allow 10% tolerance
                    if abs(diff_minutes - expected_diff) > (expected_diff * 0.1):
                        gap_count += 1
                        max_gap = max(max_gap, abs(diff_minutes - expected_diff))
                
                if gap_count > 0:
                    logger.warning(f" {symbol} {tf}: {gap_count} gap(s) detected")
                    logger.warning(f"   Largest gap: {max_gap:.1f} minutes")
                    # Don't fail, just warn - small gaps are acceptable
                else:
                    logger.info(f" {symbol} {tf}: Clean and continuous ({len(buffer)} candles)")
        
        logger.info("="*80)
        
        if all_passed:
            logger.info(" ALL BUFFERS PASSED QUALITY CHECKS")
        else:
            logger.error(" SOME BUFFERS FAILED QUALITY CHECKS")
            logger.error("   This may affect feature calculation and trading decisions")
        
        logger.info("="*80 + "\n")
        
        return all_passed
    
    def start(self):
        """Start live trading"""
        if self.running:
            logger.warning("Trading already running")
            return
        
        logger.info("="*80)
        logger.info("STARTING LIVE TRADING")
        logger.info("="*80)
        
        # ✅ NEW: Confirmation prompt for live trading
        if not self.config.dry_run:
            logger.warning("\n" + "!"*40)
            logger.warning("  YOU ARE ABOUT TO TRADE WITH REAL MONEY")
            logger.warning("  Capital: ${:.2f}".format(self.config.initial_capital))
            logger.warning("  Exchange: Kraken")
            logger.warning("  Symbols: {}".format(', '.join(self.config.symbols)))
            logger.warning("!"*40)
            logger.warning("\nThis will place REAL ORDERS with REAL MONEY.")
            logger.warning("Type exactly: 'START TRADING'")
            logger.warning("(You have 30 seconds)\n")
            
            # Cross-platform timeout implementation
            from threading import Thread, Event
            
            user_input = []
            timeout_event = Event()
            
            def get_input():
                try:
                    response = input("Confirmation: ")
                    user_input.append(response)
                except (EOFError, KeyboardInterrupt):
                    pass
                finally:
                    timeout_event.set()
            
            input_thread = Thread(target=get_input, daemon=True)
            input_thread.start()
            
            # Wait 30 seconds
            if not timeout_event.wait(timeout=30):
                logger.error(" Confirmation timeout (30 seconds) - cancelled")
                return
            
            if not user_input or user_input[0] != 'START TRADING':
                logger.error(f" Incorrect confirmation - cancelled")
                if user_input:
                    logger.error(f"   You typed: '{user_input[0]}'")
                logger.error(f"   Expected: 'START TRADING'")
                return
            
            logger.info(" Confirmation received - starting live trading\n")
        
        logger.info(" LIVE MODE - Real orders will be placed with real money!")
        logger.info(" Press Ctrl+C to stop gracefully")
        logger.info("="*80)
        
        self.running = True
        self.start_time = datetime.now()
        
        # Save initial state
        self._save_state()
        
        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        logger.info(" Live trading started successfully")
        logger.info(f"Trading on: {', '.join(self.config.symbols)}")
        logger.info(f"Timeframe: {self.config.execution_timeframe}")
        logger.info(f"Log file: logs/live_trading.log")
    
    def stop(self):
        """Stop live trading gracefully"""
        if not self.running:
            logger.warning("Trading not running")
            return
        
        logger.info("="*80)
        logger.info("STOPPING LIVE TRADING")
        logger.info("="*80)
        
        self.running = False
        
        # Wait for trading thread to finish
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        
        # ✅ FIXED: Disconnect Binance WebSocket
        logger.info("Disconnecting Binance WebSocket...")
        if hasattr(self, 'binance'):
            try:
                self.binance.disconnect_websocket()
                logger.info(" Binance WebSocket disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting Binance: {e}")
        
        # Close all positions
        logger.info("Closing all open positions...")
        self._close_all_positions()
        
        # Save final state
        self._save_state()
        
        # Print summary
        self._print_summary()
        
        logger.info(" Live trading stopped successfully")
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        if not self.running:
            return {
                'status': 'stopped',
                'message': 'Trading is not running'
            }
        
        # Get current portfolio value
        portfolio_value = self.portfolio.total_value
        
        # Calculate P&L
        pnl = portfolio_value - self.config.initial_capital
        pnl_pct = (pnl / self.config.initial_capital) * 100
        
        # Calculate runtime
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        status = {
            'status': 'running',
            'runtime': str(runtime).split('.')[0],
            'portfolio_value': f"${portfolio_value:.2f}",
            'pnl': f"${pnl:.2f} ({pnl_pct:+.2f}%)",
            'open_positions': len(self.portfolio.positions),
            'total_trades': len(self.trades_history),
            'symbols': self.config.symbols,
            'mode': 'LIVE' if not self.config.dry_run else 'PAPER'
        }
        
        return status
    
    def _trading_loop(self):
        """Main trading loop - runs in separate thread"""
        logger.info("Trading loop started")

        last_daily_reset = datetime.now().date()
        last_health_check = datetime.now()
        
        # Error tracking per symbol
        error_counts = {symbol: 0 for symbol in self.config.symbols}
        max_errors = 3
        
        # Wait for next candle completion
        next_candle_time = self._get_next_candle_time()
        
        while self.running:
            try:
                current_time = datetime.now()

                # ═══════════════════════════════════════════════════════════
                # STEP 0: Check if new trading day (reset daily tracking)
                # ═══════════════════════════════════════════════════════════
                if current_time.date() > last_daily_reset:
                    logger.info(" New trading day detected, resetting daily tracking...")
                    self._reset_daily_tracking()
                    last_daily_reset = current_time.date()
                
                # ═══════════════════════════════════════════════════════════
                # STEP 1: Emergency stop check
                # ═══════════════════════════════════════════════════════════
                if self._check_emergency_stop():
                    logger.critical("Emergency stop triggered - exiting trading loop")
                    break
                
                # ═══════════════════════════════════════════════════════════
                # STEP 2: System health check (every 5 minutes)
                # ═══════════════════════════════════════════════════════════
                if (current_time - last_health_check).total_seconds() > 300:
                    if not self._check_system_health():
                        logger.warning("System health degraded - monitoring closely")
                    last_health_check = current_time
                
                # Wait until next candle completes
                if current_time < next_candle_time:
                    time.sleep(1)
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"New candle completed: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Check Binance connection health
                if not self.binance.connection_healthy:
                    logger.warning(" Binance connection unhealthy, waiting...")
                    time.sleep(10)
                    continue
                
                # ✅ NEW: Monitor positions FIRST (before processing new signals)
                self._monitor_positions()
                
                # Process each symbol
                for symbol in self.config.symbols:
                    # Skip if too many consecutive errors
                    if error_counts[symbol] >= max_errors:
                        logger.warning(f" Skipping {symbol} - too many errors ({max_errors})")
                        continue
                    
                    try:
                        self._process_symbol(symbol)
                        error_counts[symbol] = 0  # Reset on success
                        
                    except Exception as e:
                        logger.error(f" Error processing {symbol}: {e}")
                        error_counts[symbol] += 1
                        
                        if error_counts[symbol] >= max_errors:
                            logger.error(f" {symbol} DISABLED after {max_errors} consecutive errors")
                        
                        continue  # Continue to next symbol
                
                self.total_steps += 1

                # Update next candle time
                next_candle_time = self._get_next_candle_time()
                
                # Save state periodically
                self._save_state()
                
            except Exception as e:
                logger.error(f" Error in trading loop: {e}")
                import traceback
                logger.error(traceback.format_exc())

                self.failed_cycles += 1
                self.consecutive_api_errors += 1

                time.sleep(5)
        
        logger.info("Trading loop stopped")
    
    def _process_symbol(self, symbol: str):
        """
        Process trading decision for a symbol - PRODUCTION VERSION
        
        ✅ ALL CRITICAL FIXES APPLIED:
        1. Aligned feature timing (ML and RL both use T-1)
        2. Fixed execution price (uses T-1's close, not T's open)
        3. Added data validation BEFORE processing
        4. Added duplicate/gap detection
        5. Added data freshness checks
        6. Better error handling
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}...")
        logger.info(f"{'='*60}")
        
        timeframes = ['5m', '15m', '1h']

        # ✅ NEW: Check for runtime gaps BEFORE processing
        logger.info(" Step 0a: Checking for data gaps...")
        for tf in timeframes:
            filled = self.feature_calculator.check_and_fill_runtime_gaps(
                symbol, tf, self.binance
            )
            if filled > 0:
                logger.info(f"   Filled {filled} missing {tf} candles")
        
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 0: PRE-VALIDATION - Check data quality BEFORE using it
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 0: Validating data quality...")
        
        all_dataframes = {}
        for tf in timeframes:
            df = self.binance.fetch_ohlc(symbol, tf, limit=3)
            
            if df is None or len(df) < 3:
                logger.warning(f"   Insufficient {tf} data for {symbol} (need 3 bars)")
                return
            
            all_dataframes[tf] = df
            
            # ✅ FIX: Check for duplicate timestamps
            if df.index[-2] == df.index[-3]:
                logger.warning(f"   Duplicate candles detected: {symbol} {tf}")
                logger.warning(f"    Timestamps: {df.index[-3]} == {df.index[-2]}")
                return
            
            # ✅ FIX: Check for large gaps
            tf_to_minutes = {'5m': 5, '15m': 15, '1h': 60}
            expected_diff = tf_to_minutes[tf]
            actual_diff = (df.index[-2] - df.index[-3]).total_seconds() / 60
            
            if abs(actual_diff - expected_diff) > (expected_diff * 0.3):  # 30% tolerance
                logger.warning(f"   Gap detected in {symbol} {tf} data")
                logger.warning(f"    Expected: {expected_diff} min, Got: {actual_diff:.1f} min")
                return
            
            # ✅ FIX: Check data freshness
            last_candle_age = (datetime.now() - df.index[-2]).total_seconds() / 60
            max_age = expected_diff * 1.5
            
            if last_candle_age > max_age:
                logger.warning(f"   Stale data: {symbol} {tf}")
                logger.warning(f"    Last candle: {df.index[-2]} ({last_candle_age:.1f} min old)")
                logger.warning(f"    Max acceptable age: {max_age} min")
                return
        
        logger.info("   All data quality checks passed")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: Extract and update with T-1 candles (just completed)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 1: Extracting completed candles...")
        
        latest_candles = {}
        for tf in timeframes:
            df = all_dataframes[tf]
            
            # ✅ CORRECT: Use T-1 (second-to-last = just completed)
            completed_candle = df.iloc[-2]
            
            latest_candles[tf] = {
                'timestamp': completed_candle.name,
                'open': float(completed_candle['open']),
                'high': float(completed_candle['high']),
                'low': float(completed_candle['low']),
                'close': float(completed_candle['close']),
                'volume': float(completed_candle['volume'])
            }
            
            logger.info(f"  {tf}: {completed_candle.name} (close: ${completed_candle['close']:.2f})")
        
        # Update buffers with T-1 candles
        logger.info(" Step 1b: Updating feature buffers...")
        for tf, candle in latest_candles.items():
            self.feature_calculator.update_candle(symbol, tf, candle)
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: Calculate features - NOW UNIFIED (both use T-1)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 2: Calculating features from T-1...")
        
        # ✅ FIX: Both ML and RL use SAME features (T-1)
        all_features = {}
        
        for tf in timeframes:
            # Get just the latest features (T-1)
            features = self.feature_calculator.get_current_features(symbol, tf)
            
            if features is None:
                logger.warning(f"   Could not get features for {symbol} {tf}")
                return
            
            # Validate we got correct shape
            if len(features) != 1 or features.shape[1] != 50:
                logger.error(f"   Invalid feature shape: {features.shape} (expected: (1, 50))")
                return
            
            all_features[tf] = features
            logger.debug(f"  {tf}: {features.shape} features calculated")
        
        logger.info("   Features calculated for all timeframes")
        logger.info("    Note: ML and RL both use T-1 features (properly aligned)")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: ML Prediction (using T-1 features)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 3: Getting ML prediction...")
        
        # ✅ FIX: ML now uses T-1 features (same as RL)
        try:
            ml_input = all_features['5m']
            ml_scaled = self.ml_scaler.transform(ml_input)
            ml_prediction = self.ml_model.predict_proba(ml_scaled)
            logger.info(f"  ML Prediction (from T-1): {ml_prediction}")
            
            # Interpret prediction
            pred_class = ml_prediction.argmax()
            pred_probs = ml_prediction[0] if len(ml_prediction.shape) > 1 else ml_prediction
            
            class_names = ['DOWN', 'FLAT', 'UP']
            logger.info(f"    Prediction: {class_names[pred_class]}")
            logger.info(f"    Confidence: DOWN={pred_probs[0]:.3f}, FLAT={pred_probs[1]:.3f}, UP={pred_probs[2]:.3f}")
            
        except Exception as e:
            logger.error(f"   ML prediction failed: {e}")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: Get execution price (FIXED - no look-ahead bias)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 4: Determining execution price...")
        
        # ✅ FIX: Use T-1's close OR live WebSocket price (no future data)
        
        # Option 1: Try to get real-time price from WebSocket
        live_price_data = self.binance.latest_prices.get(symbol, {})
        live_price = live_price_data.get('last')
        live_price_timestamp = live_price_data.get('timestamp')
        
        if live_price and live_price_timestamp:
            age_seconds = (datetime.now() - live_price_timestamp).total_seconds()
            
            if age_seconds < 30:  # Fresh (<30 seconds)
                current_price = live_price
                price_source = "WebSocket (real-time)"
                logger.info(f"  Using WebSocket price: ${current_price:.2f} (age: {age_seconds:.1f}s)")
            else:
                # Too old, don't use it
                logger.warning(f"  WebSocket price STALE: {age_seconds:.0f}s old")
                current_price = latest_candles['5m']['close']
                price_source = "T-1 close (stale WebSocket)"
        else:
            # No WebSocket data
            current_price = latest_candles['5m']['close']
            price_source = "T-1 close (no WebSocket)"
        
        logger.info(f"  Execution price: ${current_price:.2f} (from {price_source})")
        
        # ✅ ADDITIONAL SAFETY: Validate price is reasonable
        t_minus_1_close = latest_candles['5m']['close']
        price_deviation = abs(current_price - t_minus_1_close) / t_minus_1_close
        
        if price_deviation > 0.05:  # 5% deviation
            logger.warning(f"   Large price deviation detected:")
            logger.warning(f"    T-1 close: ${t_minus_1_close:.2f}")
            logger.warning(f"    Current: ${current_price:.2f}")
            logger.warning(f"    Deviation: {price_deviation:.2%}")
            logger.warning(f"    Using T-1 close for safety")
            current_price = t_minus_1_close
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: Build State for RL Agent (using T-1 features)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 5: Building state for RL agent...")
        
        opened_step = self.position_opened_step.get(symbol, None)
        
        try:
            # ✅ CORRECT: State uses T-1 features (same as ML)
            state = build_state_for_agent(
                features_dict=all_features,  # ← T-1 features (aligned with ML)
                portfolio=self.portfolio,
                current_price=current_price,
                initial_balance=self.config.initial_capital,
                symbol=symbol,
                position_opened_step=opened_step,
                current_step=self.total_steps
            )
            
            logger.info(f"   State built: {len(state)} dimensions")
            logger.info(f"    Market features: 150 dims (50 x 3 timeframes)")
            logger.info(f"    Position info: 5 dims")
            logger.info(f"    Account info: 5 dims")
            logger.info(f"    Asset encoding: 5 dims")
            logger.info(f"    Timeframe encoding: 18 dims")
            
        except Exception as e:
            logger.error(f"   Error building state: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: RL Agent Decision
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 6: Getting RL agent decision...")
        
        try:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rl_agent.device)
                q_values = self.rl_agent.q_network(state_tensor)
                action = q_values.argmax().item()
            
            action_name = ['HOLD', 'BUY', 'SELL'][action]
            logger.info(f"  RL Action: {action_name}")
            logger.info(f"  Q-values: HOLD={q_values[0][0]:.3f}, BUY={q_values[0][1]:.3f}, SELL={q_values[0][2]:.3f}")
            
        except Exception as e:
            logger.error(f"   RL agent error: {e}")
            return
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 7: Execute Trade (if action is not HOLD)
        # ═══════════════════════════════════════════════════════════════════
        logger.info(" Step 7: Executing trade decision...")
        
        if action != 0:  # Not HOLD
            logger.info(f"   Executing {action_name} for {symbol}")
            
            # ✅ FIX: Pass validated current_price (not future data)
            self._execute_trade(
                symbol=symbol,
                action=action,
                price=current_price,  # ← Validated price (no look-ahead)
                features=all_features,
                ml_prediction=ml_prediction
            )
        else:
            logger.info("   Action: HOLD - No trade executed")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 8: Post-Processing
        # ═══════════════════════════════════════════════════════════════════
        self.total_steps += 1
        
        logger.info(f"{'='*60}")
        logger.info(f" Processing complete for {symbol}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"{'='*60}\n")


    def _check_emergency_stop(self) -> bool:
        """
        Check if trading should be halted immediately
        """
        emergency_triggered = False
        emergency_reasons = []
        
        # CHECK 1: Total Drawdown
        portfolio_value = self.portfolio.total_value
        drawdown = (self.config.initial_capital - portfolio_value) / self.config.initial_capital
        
        if drawdown >= self.config.max_drawdown:
            emergency_triggered = True
            emergency_reasons.append(f"MAX DRAWDOWN EXCEEDED: {drawdown:.2%} (max: {self.config.max_drawdown:.2%})")
        
        # CHECK 2: Daily Loss Limit
        if hasattr(self, 'daily_start_value'):
            daily_pnl = (portfolio_value - self.daily_start_value) / self.daily_start_value
            
            if daily_pnl <= -self.config.daily_loss_limit:
                emergency_triggered = True
                emergency_reasons.append(f"DAILY LOSS LIMIT HIT: {daily_pnl:.2%} (max: -{self.config.daily_loss_limit:.2%})")
        
        # CHECK 3: Data Connection Health
        if not self.binance.connection_healthy:
            # ✅ FIXED: Check if None instead of hasattr
            if self.connection_unhealthy_since is None:
                self.connection_unhealthy_since = datetime.now()
            
            unhealthy_duration = (datetime.now() - self.connection_unhealthy_since).total_seconds()
            
            if unhealthy_duration > 300:  # 5 minutes
                emergency_triggered = True
                emergency_reasons.append(f"DATA CONNECTION UNHEALTHY: {unhealthy_duration:.0f}s")
        else:
            # Reset counter if connection is healthy
            self.connection_unhealthy_since = None  # ✅ FIXED: Set to None instead of deleting
        
        # CHECK 4: Consecutive API Errors
        if hasattr(self, 'consecutive_api_errors') and self.consecutive_api_errors >= 10:
            emergency_triggered = True
            emergency_reasons.append(f"TOO MANY API ERRORS: {self.consecutive_api_errors} consecutive failures")
        
        # CHECK 5: No Successful Trades for Extended Period
        if hasattr(self, 'last_successful_trade'):
            time_since_trade = (datetime.now() - self.last_successful_trade).total_seconds() / 3600
            
            if time_since_trade > 24:
                logger.warning(f" WARNING: No successful trades in {time_since_trade:.1f} hours")
        
        # CHECK 6: Portfolio Value Sanity Check
        if portfolio_value < self.config.initial_capital * 0.5:
            emergency_triggered = True
            emergency_reasons.append(f"CATASTROPHIC LOSS: Portfolio at ${portfolio_value:.2f} (50% of initial)")
        
        if portfolio_value > self.config.initial_capital * 100:
            emergency_triggered = True
            emergency_reasons.append(f"UNREALISTIC GAINS: Portfolio at ${portfolio_value:.2f} (likely bug)")
        
        # CHECK 7: Balance Sanity Check
        if self.portfolio.cash_balance < 0:
            emergency_triggered = True
            emergency_reasons.append(f"NEGATIVE BALANCE: ${self.portfolio.cash_balance:.2f}")
        
        # TRIGGER EMERGENCY STOP IF NEEDED
        if emergency_triggered:
            logger.critical("\n" + "="*80)
            logger.critical("   EMERGENCY STOP TRIGGERED ")
            logger.critical("="*80)
            
            for reason in emergency_reasons:
                logger.critical(f"   {reason}")
            
            logger.critical("")
            logger.critical("  SYSTEM STATUS:")
            logger.critical(f"    Portfolio Value: ${portfolio_value:.2f}")
            logger.critical(f"    Initial Capital: ${self.config.initial_capital:.2f}")
            logger.critical(f"    Total Drawdown: {drawdown:.2%}")
            logger.critical(f"    Cash Balance: ${self.portfolio.cash_balance:.2f}")
            logger.critical(f"    Open Positions: {len(self.portfolio.positions)}")
            logger.critical("")
            logger.critical("  ACTIONS TAKEN:")
            logger.critical("    1. Stopping all trading")
            logger.critical("    2. Closing all open positions")
            logger.critical("    3. Saving final state")
            logger.critical("="*80 + "\n")

            logger.critical("  Cancelling all pending orders...")
            self._cancel_all_pending_orders()
            
            time.sleep(5)
            
            self._close_all_positions()
            self._save_state()
            self.running = False
            
            return True
        
        return False
    
    def _cancel_all_pending_orders(self):
        """Cancel all open orders on Kraken"""
        if self.kraken.mode == 'paper':
            return
        
        try:
            open_orders = self.kraken.get_open_orders()
            
            for order_id in open_orders.keys():
                logger.info(f"  Cancelling order: {order_id}")
                self.kraken.cancel_order(order_id)
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    def _log_cycle_metrics(self):
        """
        Log performance metrics for current trading session
        
        ✅ Called periodically to track system health
        """
        if self.cycle_count % 10 == 0:  # Every 10 cycles
            uptime = datetime.now() - self.start_time
            success_rate = (self.successful_cycles / self.cycle_count * 100) if self.cycle_count > 0 else 0
            
            logger.info(f"\n{'='*60}")
            logger.info(f"  SESSION METRICS")
            logger.info(f"{'='*60}")
            logger.info(f"  Uptime: {str(uptime).split('.')[0]}")
            logger.info(f"  Total cycles: {self.cycle_count}")
            logger.info(f"  Successful: {self.successful_cycles} ({success_rate:.1f}%)")
            logger.info(f"  Failed: {self.failed_cycles}")
            logger.info(f"  Consecutive API errors: {self.consecutive_api_errors}")
            logger.info(f"  Portfolio value: ${self.portfolio.total_value:.2f}")
            logger.info(f"  Open positions: {len(self.portfolio.positions)}")
            logger.info(f"  Total trades: {len(self.trades_history)}")
            
            if self.trades_history:
                winning_trades = len([t for t in self.trades_history if t.get('pnl', 0) > 0])
                win_rate = winning_trades / len([t for t in self.trades_history if 'pnl' in t]) * 100 if len([t for t in self.trades_history if 'pnl' in t]) > 0 else 0
                logger.info(f"  Win rate: {win_rate:.1f}%")
            
            logger.info(f"{'='*60}\n")


    def _handle_symbol_error(self, symbol: str, error: Exception):
        """
        Handle errors for a specific symbol
        
        ✅ Tracks error counts per symbol
        ✅ Disables symbols after repeated failures
        ✅ Logs detailed error information
        
        Args:
            symbol: Trading symbol that had an error
            error: The exception that occurred
        """
        self.error_counts[symbol] += 1
        self.consecutive_api_errors += 1
        self.failed_cycles += 1
        
        logger.error(f"\n{'='*60}")
        logger.error(f"  ERROR PROCESSING {symbol}")
        logger.error(f"{'='*60}")
        logger.error(f"  Error: {str(error)}")
        logger.error(f"  Symbol error count: {self.error_counts[symbol]}")
        logger.error(f"  Consecutive API errors: {self.consecutive_api_errors}")
        logger.error(f"{'='*60}")
        
        # Log stack trace for debugging
        import traceback
        logger.error(f"  Stack trace:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                logger.error(f"    {line}")
        logger.error(f"{'='*60}\n")
        
        # Disable symbol if too many errors
        max_errors = 5
        if self.error_counts[symbol] >= max_errors:
            logger.critical(f"\n{'!'*60}")
            logger.critical(f"  DISABLING {symbol}")
            logger.critical(f"{'!'*60}")
            logger.critical(f"  Reason: {max_errors} consecutive errors")
            logger.critical(f"  Symbol will be skipped for remainder of session")
            logger.critical(f"  Restart bot to re-enable")
            logger.critical(f"{'!'*60}\n")
            
            # Remove from active symbols
            if symbol in self.config.symbols:
                self.config.symbols.remove(symbol)


    def _handle_successful_cycle(self, symbol: str):
        """
        Handle successful processing of a symbol
        
        ✅ Resets error counters
        ✅ Updates success metrics
        
        Args:
            symbol: Trading symbol that was processed successfully
        """
        self.error_counts[symbol] = 0
        self.consecutive_api_errors = 0
        self.successful_cycles += 1


    def _check_system_health(self) -> bool:
        """
        Comprehensive system health check
        
        ✅ Checks all critical systems
        ✅ Returns False if any system is degraded
        
        Returns:
            True if all systems healthy, False otherwise
        """
        issues = []
        
        # Check 1: Data connection
        if not self.binance.connection_healthy:
            issues.append("Binance connection unhealthy")
        
        # Check 2: API error rate
        if self.consecutive_api_errors >= 5:
            issues.append(f"High API error rate ({self.consecutive_api_errors} consecutive)")
        
        # Check 3: Trading activity
        if hasattr(self, 'last_successful_trade'):
            minutes_since_trade = (datetime.now() - self.last_successful_trade).total_seconds() / 60
            if minutes_since_trade > 120:  # 2 hours
                issues.append(f"No successful trades in {minutes_since_trade:.0f} minutes")
        
        # Check 4: Portfolio sanity
        portfolio_value = self.portfolio.total_value
        if portfolio_value <= 0 or portfolio_value > self.config.initial_capital * 100:
            issues.append(f"Unrealistic portfolio value: ${portfolio_value:.2f}")
        
        # Check 5: Balance sanity
        if self.portfolio.cash_balance < 0:
            issues.append(f"Negative cash balance: ${self.portfolio.cash_balance:.2f}")
        
        # Check 6: Active symbols
        if len(self.config.symbols) == 0:
            issues.append("No active trading symbols remaining")
        
        if issues:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"  SYSTEM HEALTH DEGRADED")
            logger.warning(f"{'='*60}")
            for issue in issues:
                logger.warning(f"   {issue}")
            logger.warning(f"{'='*60}\n")
            
            return False
        
        return True


    def _reset_daily_tracking(self):
        """
        Reset daily tracking metrics (call at start of each trading day)
        
        ✅ Tracks daily P&L for daily loss limits
        ✅ Resets error counters
        """
        # Store starting value for daily P&L calculation
        self.daily_start_value = self.portfolio.total_value
        self.daily_start_time = datetime.now()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"  DAILY TRACKING RESET")
        logger.info(f"{'='*60}")
        logger.info(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"  Starting Value: ${self.daily_start_value:.2f}")
        logger.info(f"  Daily Loss Limit: {self.config.daily_loss_limit:.2%}")
        logger.info(f"{'='*60}\n")

    def _wait_for_order_fill(self, order_id: str, timeout: int = 30) -> Optional[Dict]:
        """
        Wait for order to be filled with timeout
        
        Args:
            order_id: Kraken order ID
            timeout: Maximum seconds to wait
            
        Returns:
            Dict with fill details if successful, None if timeout/failed
        """
        start_time = time.time()
        check_interval = 1  # Check every second
        
        logger.info(f"   Waiting for order {order_id} to fill (timeout: {timeout}s)...")
        
        while time.time() - start_time < timeout:
            try:
                # Query Kraken for order status
                if self.kraken.mode == 'paper':
                    # Paper trading always fills immediately
                    return {
                        'filled': True,
                        'status': 'closed',
                        'fill_price': None,  # Already handled in paper trading
                        'fill_size': None,
                        'timestamp': datetime.now()
                    }
                
                # For live trading, check actual order status
                result = self.kraken.api.query_private('QueryOrders', {'txid': order_id})
                
                if result.get('error'):
                    logger.error(f"   Error checking order status: {result['error']}")
                    time.sleep(check_interval)
                    continue
                
                orders = result.get('result', {})
                
                if order_id not in orders:
                    logger.warning(f"   Order {order_id} not found in query results")
                    time.sleep(check_interval)
                    continue
                
                order_info = orders[order_id]
                status = order_info.get('status')
                
                logger.debug(f"  Order status: {status}")
                
                # Check if order is filled
                if status == 'closed':
                    # Order is fully filled
                    vol_exec = float(order_info.get('vol_exec', 0))
                    avg_price = float(order_info.get('price', 0))
                    
                    logger.info(f"   Order FILLED!")
                    logger.info(f"    Filled volume: {vol_exec}")
                    logger.info(f"    Average price: ${avg_price:.2f}")
                    
                    return {
                        'filled': True,
                        'status': 'closed',
                        'fill_price': avg_price,
                        'fill_size': vol_exec,
                        'timestamp': datetime.now()
                    }
                
                elif status in ['canceled', 'expired']:
                    logger.error(f"   Order {status.upper()}!")
                    return None
                
                elif status == 'pending':
                    elapsed = time.time() - start_time
                    logger.debug(f"  Order still pending ({elapsed:.1f}s elapsed)...")
                    time.sleep(check_interval)
                    continue
                
                else:
                    logger.debug(f"  Order status: {status}, waiting...")
                    time.sleep(check_interval)
                    continue
                    
            except Exception as e:
                logger.error(f"   Error checking order status: {e}")
                time.sleep(check_interval)
        
        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(f"   Order fill timeout ({elapsed:.1f}s)")
        logger.error(f"    Order {order_id} did not fill within {timeout} seconds")
        
        if self.kraken.mode == 'live':
            cancel_result = self.kraken.cancel_order(order_id)
            
            if cancel_result['success']:
                logger.info(f"   Order cancelled successfully")
                return None  # Safe to exit
            else:
                logger.critical(f"   FAILED TO CANCEL ORDER!")
                logger.critical(f"    Order {order_id} may still be live")
                logger.critical(f"    EMERGENCY: Check Kraken manually!")
                # Could trigger emergency stop here
                return None
        else:
            return None
        
    def _validate_balance_sync(self):
        """
        Validate that portfolio balance matches Kraken balance
        Called periodically to catch drift
        """
        if self.kraken.mode == 'paper':
            return  # No validation needed for paper trading
        
        try:
            # Get current Kraken balance
            kraken_balances = self.kraken.get_account_balance()
            
            actual_usd = 0.0
            for currency in ['USD', 'USDT', 'ZUSD']:
                if currency in kraken_balances:
                    actual_usd += kraken_balances[currency].balance
            
            # Compare with portfolio
            portfolio_cash = self.portfolio.cash_balance
            
            # Allow 2% tolerance for fees/rounding
            tolerance = max(2.0, portfolio_cash * 0.02)
            
            if abs(actual_usd - portfolio_cash) > tolerance:
                logger.warning("="*80)
                logger.warning("  BALANCE DRIFT DETECTED")
                logger.warning("="*80)
                logger.warning(f"  Portfolio shows: ${portfolio_cash:.2f}")
                logger.warning(f"  Kraken shows: ${actual_usd:.2f}")
                logger.warning(f"  Difference: ${abs(actual_usd - portfolio_cash):.2f}")
                logger.warning("="*80)
                logger.warning("  This may indicate:")
                logger.warning("    - Fees not properly tracked")
                logger.warning("    - Manual trades on Kraken")
                logger.warning("    - Reconciliation issues")
                logger.warning("="*80)
                
                # Don't stop trading, just warn
                # User can investigate and stop manually if needed
            else:
                logger.debug(f"✅ Balance sync OK: Portfolio=${portfolio_cash:.2f}, Kraken=${actual_usd:.2f}")
                
        except Exception as e:
            logger.error(f"Error validating balance: {e}")

    
    def _execute_trade(self, symbol: str, action: int, price: float, 
                    features: Optional[Dict], ml_prediction: Optional[np.ndarray]):
        """
        Execute trade - TRAINING-ALIGNED VERSION WITH RECONCILIATION
        
        ✅ Matches training environment exactly:
        - 95% position sizing
        - Simple balance checks only
        - No Kelly Criterion
        - No sophisticated risk management
        - Max 2 positions (configurable)
        
        ✅ NEW: Full reconciliation safety:
        - Retry logic for portfolio updates
        - Kraken order status verification
        - Automatic reconciliation on failure
        - Emergency stop if unreconcilable
        """
        has_position = symbol in self.portfolio.positions
        
        # ═══════════════════════════════════════════════════════════
        # BUY ACTION
        # ═══════════════════════════════════════════════════════════
        if action == 1:

            if len(self.trades_history) % 10 == 0:
                self._validate_balance_sync()

            if has_position:
                logger.info(f"  Already have position in {symbol}, skipping BUY")
                return
            
            # ✅ TRAINING-ALIGNED: Check max positions
            if len(self.portfolio.positions) >= self.config.max_positions:
                logger.info(f"  Max positions ({self.config.max_positions}) reached, skipping BUY")
                return
            
            # ✅ TRAINING-ALIGNED: Use 95% of available balance
            position_value = self.portfolio.cash_balance * 0.95
            position_size = position_value / price
            
            # ✅ TRAINING-ALIGNED: Simple balance check
            fee = position_value * 0.0026
            total_cost = position_value + fee
            
            if total_cost > self.portfolio.cash_balance:
                logger.warning(f"  Insufficient balance: need ${total_cost:.2f}, have ${self.portfolio.cash_balance:.2f}")
                return
            
            # Validate minimum trade size ($10 minimum for Kraken)
            if position_value < 10:
                logger.warning(f"  Position size too small: ${position_value:.2f} (min $10)")
                return
            
            # Log trade intent
            logger.info(f"\n{'='*60}")
            logger.info(f"  PLACING BUY ORDER (TRAINING-ALIGNED + RECONCILIATION)")
            logger.info(f"{'='*60}")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Size: {position_size:.8f}")
            logger.info(f"  Price: ${price:.2f}")
            logger.info(f"  Value: ${position_value:.2f}")
            logger.info(f"  Position Sizing: 95% of balance (matches training)")
            logger.info(f"  Balance: ${self.portfolio.cash_balance:.2f}")
            logger.info(f"  Mode: {self.kraken.mode.upper()}")
            logger.info(f"{'='*60}")
            
            # ════════════════════════════════════════════════════════
            # STEP 1: Place order on Kraken
            # ════════════════════════════════════════════════════════
            order = KrakenOrder(
                pair=symbol,
                type='buy',
                ordertype='market',
                volume=position_size
            )
            
            order_result = self.kraken.place_order(order, current_price=price)
            
            if not order_result['success']:
                logger.error(f"  Order placement failed: {order_result.get('error', 'Unknown error')}")
                return
            
            order_id = order_result['order_id']
            logger.info(f"   Order placed: {order_id}")
            
            # ════════════════════════════════════════════════════════
            # STEP 2: Wait for fill confirmation
            # ════════════════════════════════════════════════════════
            logger.info(f"  Waiting for order fill (timeout: 30s)...")
            fill_result = self._wait_for_order_fill(order_id, timeout=30)
            
            if fill_result is None:
                # ✅ NEW: Order didn't fill within timeout - verify on Kraken
                logger.error(f"    Order did not fill within timeout")
                logger.warning(f"  Verifying order status directly on Kraken...")
                
                # Import verification function
                from src.trading.order_reconciliation import verify_order_status_on_kraken
                
                actual_status = verify_order_status_on_kraken(self.kraken, order_id)
                
                if actual_status and actual_status['filled']:
                    logger.warning(f"   Order WAS filled on Kraken!")
                    logger.warning(f"     Using real fill data from exchange...")
                    fill_result = actual_status
                else:
                    logger.error(f"   Order truly did not fill on Kraken")
                    logger.error(f"     Trade aborted - no position opened")
                    return
            
            # ════════════════════════════════════════════════════════
            # STEP 3: Extract fill details
            # ════════════════════════════════════════════════════════
            actual_price = fill_result.get('fill_price') or price
            actual_size = fill_result.get('fill_size') or position_size
            fee = actual_price * actual_size * 0.0026
            
            logger.info(f"   Fill Details:")
            logger.info(f"     Price: ${actual_price:.2f}")
            logger.info(f"     Size: {actual_size:.8f}")
            logger.info(f"     Fee: ${fee:.4f}")
            logger.info(f"     Total Cost: ${actual_price * actual_size + fee:.2f}")
            
            # ════════════════════════════════════════════════════════
            # STEP 4: Update portfolio with RETRY LOGIC
            # ════════════════════════════════════════════════════════
            logger.info(f"\n   Updating portfolio (order filled on Kraken)...")
            
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    success = self.portfolio.open_position(
                        symbol=symbol,
                        quantity=actual_size,
                        price=actual_price,
                        position_type='long',
                        fees=fee
                    )
                    
                    if success:
                        logger.info(f"      Portfolio updated (attempt {attempt + 1})")
                        break
                    else:
                        logger.error(f"      Portfolio update returned False (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            logger.info(f"     Retrying in 1 second...")
                            time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"      Portfolio update exception (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"     Retrying in 1 second...")
                        time.sleep(1)
            
            # ════════════════════════════════════════════════════════
            # STEP 5: Handle reconciliation if update failed
            # ════════════════════════════════════════════════════════
            if not success:
                logger.critical(f"\n   CRITICAL: Portfolio update failed after {max_retries} attempts!")
                logger.critical(f"     Order {order_id} is FILLED on Kraken but NOT tracked locally!")
                logger.critical(f"     This is a state mismatch - attempting reconciliation...")
                
                # Record unreconciled order
                self.reconciliation.record_unreconciled_order(
                    order_id=order_id,
                    symbol=symbol,
                    side='buy',
                    fill_price=actual_price,
                    fill_size=actual_size
                )
                
                # Attempt immediate reconciliation
                logger.info(f"\n   Attempting IMMEDIATE reconciliation...")
                reconciled = self.reconciliation.attempt_reconciliation(
                    portfolio=self.portfolio,
                    max_retries=5  # More aggressive for immediate reconciliation
                )
                
                if reconciled:
                    logger.info(f"      RECONCILIATION SUCCESSFUL!")
                    logger.info(f"        Portfolio now matches Kraken state")
                    success = True  # Mark as success since we reconciled
                else:
                    logger.critical(f"\n   RECONCILIATION FAILED ")
                    logger.critical(f"     Order {order_id} is filled on Kraken")
                    logger.critical(f"     But portfolio state cannot be updated")
                    logger.critical(f"     Manual intervention REQUIRED!")
                    logger.critical(f"     Review: logs/unreconciled_orders.json")
                    logger.critical(f"\n   TRIGGERING EMERGENCY STOP")
                    
                    # Stop trading immediately
                    self.running = False
                    
                    # Try to close all other positions for safety
                    logger.critical(f"     Attempting to close all other positions...")
                    self._close_all_positions()
                    
                    return  # Exit - system is now stopped
            
            # ════════════════════════════════════════════════════════
            # STEP 6: Record successful trade
            # ════════════════════════════════════════════════════════
            # Record timing
            self.position_opened_at[symbol] = datetime.now()
            self.position_opened_step[symbol] = self.total_steps
            
            # Log trade
            self.trades_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'BUY',
                'size': actual_size,
                'price': actual_price,
                'value': actual_size * actual_price,
                'order_id': order_id,
                'fill_confirmed': True,
                'portfolio_updated': success,
                'reconciliation_required': not success,
                'position_sizing_method': 'training_aligned_95pct'
            })
            
            logger.info(f"\n   BUY COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"  Position opened: {actual_size:.8f} {symbol} @ ${actual_price:.2f}")
            logger.info(f"  Portfolio updated: {success}")
            logger.info(f"  New balance: ${self.portfolio.cash_balance:.2f}")
            logger.info(f"  Open positions: {len(self.portfolio.positions)}")
            logger.info(f"{'='*60}\n")
        
        # ═══════════════════════════════════════════════════════════
        # SELL ACTION
        # ═══════════════════════════════════════════════════════════
        elif action == 2:
            if not has_position:
                logger.info(f"  No position in {symbol}, skipping SELL")
                return
            
            position = self.portfolio.positions[symbol]
            
            # Log trade intent
            logger.info(f"\n{'='*60}")
            logger.info(f"  PLACING SELL ORDER (TRAINING-ALIGNED + RECONCILIATION)")
            logger.info(f"{'='*60}")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Size: {position.quantity:.8f}")
            logger.info(f"  Price: ${price:.2f}")
            logger.info(f"  Value: ${position.quantity * price:.2f}")
            logger.info(f"  Entry Price: ${position.entry_price:.2f}")
            logger.info(f"  Expected P&L: ${(price - position.entry_price) * position.quantity:.2f}")
            logger.info(f"  Mode: {self.kraken.mode.upper()}")
            logger.info(f"{'='*60}")
            
            # ════════════════════════════════════════════════════════
            # STEP 1: Place order on Kraken
            # ════════════════════════════════════════════════════════
            order = KrakenOrder(
                pair=symbol,
                type='sell',
                ordertype='market',
                volume=position.quantity
            )
            
            order_result = self.kraken.place_order(order, current_price=price)
            
            if not order_result['success']:
                logger.error(f"  Order placement failed: {order_result.get('error', 'Unknown error')}")
                return
            
            order_id = order_result['order_id']
            logger.info(f"   Order placed: {order_id}")
            
            # ════════════════════════════════════════════════════════
            # STEP 2: Wait for fill confirmation
            # ════════════════════════════════════════════════════════
            logger.info(f"  Waiting for order fill (timeout: 30s)...")
            fill_result = self._wait_for_order_fill(order_id, timeout=30)
            
            if fill_result is None:
                # ✅ NEW: Order didn't fill within timeout - verify on Kraken
                logger.error(f"    Order did not fill within timeout")
                logger.warning(f"  Verifying order status directly on Kraken...")
                
                # Import verification function
                from src.trading.order_reconciliation import verify_order_status_on_kraken
                
                actual_status = verify_order_status_on_kraken(self.kraken, order_id)
                
                if actual_status and actual_status['filled']:
                    logger.warning(f"   Order WAS filled on Kraken!")
                    logger.warning(f"     Using real fill data from exchange...")
                    fill_result = actual_status
                else:
                    logger.error(f"   Order truly did not fill on Kraken")
                    logger.error(f"     Trade aborted - position still open")
                    logger.error(f"     WARNING: You may need to manually close this position!")
                    return
            
            # ════════════════════════════════════════════════════════
            # STEP 3: Extract fill details
            # ════════════════════════════════════════════════════════
            actual_price = fill_result.get('fill_price') or price
            actual_size = fill_result.get('fill_size') or position.quantity
            fee = actual_price * actual_size * 0.0026
            
            logger.info(f"   Fill Details:")
            logger.info(f"     Price: ${actual_price:.2f}")
            logger.info(f"     Size: {actual_size:.8f}")
            logger.info(f"     Fee: ${fee:.4f}")
            logger.info(f"     Total Revenue: ${actual_price * actual_size - fee:.2f}")
            
            # ════════════════════════════════════════════════════════
            # STEP 4: Update portfolio with RETRY LOGIC
            # ════════════════════════════════════════════════════════
            logger.info(f"\n   Closing position in portfolio (order filled on Kraken)...")
            
            max_retries = 3
            trade_result = None
            
            for attempt in range(max_retries):
                try:
                    trade_result = self.portfolio.close_position(
                        symbol=symbol,
                        price=actual_price,
                        fees=fee,
                        reason='sell_signal'
                    )
                    
                    if trade_result:
                        logger.info(f"      Position closed (attempt {attempt + 1})")
                        break
                    else:
                        logger.error(f"      Position close returned None (attempt {attempt + 1}/{max_retries})")
                        if attempt < max_retries - 1:
                            logger.info(f"     Retrying in 1 second...")
                            time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"      Position close exception (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"     Retrying in 1 second...")
                        time.sleep(1)
            
            # ════════════════════════════════════════════════════════
            # STEP 5: Handle reconciliation if close failed
            # ════════════════════════════════════════════════════════
            if not trade_result:
                logger.critical(f"\n   CRITICAL: Position close failed after {max_retries} attempts!")
                logger.critical(f"     Order {order_id} is FILLED on Kraken (position closed)")
                logger.critical(f"     But portfolio still shows position as OPEN!")
                logger.critical(f"     This is a state mismatch - attempting reconciliation...")
                
                # Record unreconciled order
                self.reconciliation.record_unreconciled_order(
                    order_id=order_id,
                    symbol=symbol,
                    side='sell',
                    fill_price=actual_price,
                    fill_size=actual_size
                )
                
                # Attempt immediate reconciliation
                logger.info(f"\n   Attempting IMMEDIATE reconciliation...")
                reconciled = self.reconciliation.attempt_reconciliation(
                    portfolio=self.portfolio,
                    max_retries=5
                )
                
                if reconciled:
                    logger.info(f"      RECONCILIATION SUCCESSFUL!")
                    logger.info(f"        Portfolio now matches Kraken state")
                    # Retrieve trade result after reconciliation
                    # (We don't have it, but position is closed)
                    trade_result = {'net_pnl': 0, 'return_pct': 0}  # Placeholder
                else:
                    logger.critical(f"\n   RECONCILIATION FAILED ")
                    logger.critical(f"     Order {order_id} is filled on Kraken")
                    logger.critical(f"     But portfolio position cannot be closed")
                    logger.critical(f"     Manual intervention REQUIRED!")
                    logger.critical(f"     Review: logs/unreconciled_orders.json")
                    logger.critical(f"\n   TRIGGERING EMERGENCY STOP")
                    
                    # Stop trading immediately
                    self.running = False
                    
                    # Try to close all other positions for safety
                    logger.critical(f"     Attempting to close all other positions...")
                    self._close_all_positions()
                    
                    return  # Exit - system is now stopped
            
            # ════════════════════════════════════════════════════════
            # STEP 6: Record successful trade
            # ════════════════════════════════════════════════════════
            # Get P&L
            pnl = trade_result.get('net_pnl', 0)
            pnl_pct = trade_result.get('return_pct', 0) * 100
            
            # Clear timing
            hold_duration = None
            if symbol in self.position_opened_at:
                hold_duration = datetime.now() - self.position_opened_at[symbol]
                del self.position_opened_at[symbol]
            
            if symbol in self.position_opened_step:
                del self.position_opened_step[symbol]
            
            # Log trade
            self.trades_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'size': actual_size,
                'price': actual_price,
                'value': actual_size * actual_price,
                'pnl': pnl,
                'pnl_pct': trade_result.get('return_pct', 0),
                'order_id': order_id,
                'hold_duration': str(hold_duration).split('.')[0] if hold_duration else None,
                'fill_confirmed': True,
                'portfolio_updated': trade_result is not None,
                'reconciliation_required': trade_result is None
            })
            
            logger.info(f"\n   SELL COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"  Position closed: {actual_size:.8f} {symbol} @ ${actual_price:.2f}")
            logger.info(f"  P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
            if hold_duration:
                logger.info(f"  Hold duration: {hold_duration}")
            logger.info(f"  Portfolio updated: {trade_result is not None}")
            logger.info(f"  New balance: ${self.portfolio.cash_balance:.2f}")
            logger.info(f"  Open positions: {len(self.portfolio.positions)}")
            logger.info(f"{'='*60}\n")

    def _monitor_positions(self):
        """
        Monitor all open positions for stop-loss and take-profit
        
        ✅ Checks every position against risk limits
        ✅ Automatically closes positions that hit thresholds
        ✅ Called BEFORE processing new signals each cycle
        """
        if not self.portfolio.positions:
            return  # No positions to monitor
        
        logger.info(f"\n{'='*60}")
        logger.info(f"  MONITORING {len(self.portfolio.positions)} OPEN POSITION(S)")
        logger.info(f"{'='*60}")
        
        for symbol in list(self.portfolio.positions.keys()):  # Use list() to allow removal during iteration
            position = self.portfolio.positions[symbol]
            
            try:
                # ═══════════════════════════════════════════════════════════
                # Get current price
                # ═══════════════════════════════════════════════════════════
                
                # Try WebSocket first
                current_price = self.binance.latest_prices.get(symbol, {}).get('last')
                
                if not current_price:
                    # Fallback to REST API
                    current_price = self.binance.get_current_price(symbol)
                
                if not current_price:
                    logger.warning(f"   Cannot get price for {symbol}, skipping monitor")
                    continue
                
                # ═══════════════════════════════════════════════════════════
                # Calculate P&L
                # ═══════════════════════════════════════════════════════════
                entry_price = position.entry_price
                pnl_pct = (current_price - entry_price) / entry_price
                pnl_dollars = (current_price - entry_price) * position.quantity
                position_value = position.quantity * current_price
                
                # Calculate hold duration
                if symbol in self.position_opened_at:
                    hold_duration = datetime.now() - self.position_opened_at[symbol]
                    hold_minutes = hold_duration.total_seconds() / 60
                else:
                    hold_duration = None
                    hold_minutes = 0
                
                # Log position status
                logger.info(f"\n  {symbol}:")
                logger.info(f"    Entry: ${entry_price:.2f}")
                logger.info(f"    Current: ${current_price:.2f}")
                logger.info(f"    P&L: ${pnl_dollars:.2f} ({pnl_pct:+.2%})")
                logger.info(f"    Size: {position.quantity:.8f}")
                logger.info(f"    Value: ${position_value:.2f}")
                if hold_duration:
                    logger.info(f"    Hold time: {str(hold_duration).split('.')[0]}")
                
                # ═══════════════════════════════════════════════════════════
                # Check STOP-LOSS
                # ═══════════════════════════════════════════════════════════
                if pnl_pct <= -self.config.stop_loss:
                    logger.warning(f"\n   STOP-LOSS TRIGGERED for {symbol} ")
                    logger.warning(f"    P&L: {pnl_pct:.2%} (threshold: -{self.config.stop_loss:.2%})")
                    logger.warning(f"    Loss: ${pnl_dollars:.2f}")
                    logger.warning(f"     EMERGENCY SELL")
                    
                    # Force sell immediately
                    self._execute_emergency_exit(
                        symbol=symbol,
                        current_price=current_price,
                        reason='stop_loss',
                        pnl_pct=pnl_pct
                    )
                    continue
                
                # ═══════════════════════════════════════════════════════════
                # Check TAKE-PROFIT
                # ═══════════════════════════════════════════════════════════
                elif pnl_pct >= self.config.take_profit:
                    logger.info(f"\n   TAKE-PROFIT TRIGGERED for {symbol} ")
                    logger.info(f"    P&L: {pnl_pct:.2%} (threshold: +{self.config.take_profit:.2%})")
                    logger.info(f"    Profit: ${pnl_dollars:.2f}")
                    logger.info(f"     TAKING PROFIT")
                    
                    # Take profit
                    self._execute_emergency_exit(
                        symbol=symbol,
                        current_price=current_price,
                        reason='take_profit',
                        pnl_pct=pnl_pct
                    )
                    continue
                
                # ═══════════════════════════════════════════════════════════
                # Check MAXIMUM HOLD TIME (optional safety)
                # ═══════════════════════════════════════════════════════════
                max_hold_minutes = 1440  # 24 hours default
                
                if hold_minutes > max_hold_minutes:
                    logger.warning(f"\n   MAX HOLD TIME EXCEEDED for {symbol}")
                    logger.warning(f"    Hold time: {hold_minutes:.0f} minutes ({hold_minutes/60:.1f} hours)")
                    logger.warning(f"    Max allowed: {max_hold_minutes} minutes")
                    logger.warning(f"    Current P&L: {pnl_pct:+.2%}")
                    logger.warning(f"     CLOSING POSITION")
                    
                    # Close position
                    self._execute_emergency_exit(
                        symbol=symbol,
                        current_price=current_price,
                        reason='max_hold_time',
                        pnl_pct=pnl_pct
                    )
                    continue
                
                # Position is within acceptable range
                logger.info(f"    Status:  OK (within limits)")
                
            except Exception as e:
                logger.error(f"   Error monitoring {symbol}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"{'='*60}\n")

    def _execute_emergency_exit(self, symbol: str, current_price: float, 
                            reason: str, pnl_pct: float):
        """
        Execute emergency exit (stop-loss, take-profit, or max hold time)
        
        ✅ Bypasses normal RL decision making
        ✅ Forces immediate market order
        ✅ Waits for fill confirmation
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            reason: Exit reason ('stop_loss', 'take_profit', 'max_hold_time')
            pnl_pct: Current P&L percentage
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  EMERGENCY EXIT: {reason.upper().replace('_', ' ')}")
        logger.info(f"{'='*60}")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  P&L: {pnl_pct:+.2%}")
        logger.info(f"  Price: ${current_price:.2f}")
        logger.info(f"{'='*60}")
        
        position = self.portfolio.positions[symbol]
        
        # Place SELL order
        order = KrakenOrder(
            pair=symbol,
            type='sell',
            ordertype='market',
            volume=position.quantity
        )
        
        order_result = self.kraken.place_order(order, current_price=current_price)
        
        if not order_result['success']:
            logger.error(f"   EMERGENCY EXIT FAILED: {order_result.get('error')}")
            logger.error(f"    Position still open - manual intervention required!")
            return
        
        order_id = order_result['order_id']
        logger.info(f"   Emergency order placed: {order_id}")
        
        # Wait for fill
        fill_result = self._wait_for_order_fill(order_id, timeout=30)
        
        if fill_result is None:
            logger.error(f"   EMERGENCY EXIT ORDER DID NOT FILL")
            logger.error(f"    WARNING: Position may still be open!")
            logger.error(f"    Immediate manual intervention required!")
            return
        
        # Update portfolio
        actual_price = fill_result.get('fill_price') or current_price
        fee = actual_price * position.quantity * 0.0026
        
        trade_result = self.portfolio.close_position(
            symbol=symbol,
            price=actual_price,
            fees=fee,
            reason=reason
        )
        
        if trade_result:
            logger.info(f"   EMERGENCY EXIT COMPLETED")
            logger.info(f"    Final P&L: ${trade_result['net_pnl']:.2f} ({trade_result['return_pct']*100:+.2f}%)")
            
            # Clear timing
            if symbol in self.position_opened_at:
                del self.position_opened_at[symbol]
            if symbol in self.position_opened_step:
                del self.position_opened_step[symbol]
            
            # Log trade
            self.trades_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'EMERGENCY_EXIT',
                'size': position.quantity,
                'price': actual_price,
                'pnl': trade_result['net_pnl'],
                'pnl_pct': trade_result['return_pct'],
                'reason': reason,
                'order_id': order_id
            })
        else:
            logger.error(f"   Failed to update portfolio after emergency exit")
        
        logger.info(f"{'='*60}\n")
    
    def _get_next_candle_time(self) -> datetime:
        """Calculate when next candle will complete + buffer for API availability"""
        now = datetime.now()
        
        # Calculate next candle close time
        if self.config.execution_timeframe == '5m':
            minutes = (now.minute // 5 + 1) * 5
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
        elif self.config.execution_timeframe == '15m':
            minutes = (now.minute // 15 + 1) * 15
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
        elif self.config.execution_timeframe == '1h':
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = now + timedelta(minutes=5)  # Default 5m
        
        # Add 5-second buffer to ensure candle is fully available from Binance
        next_time += timedelta(seconds=10)
        
        return next_time
    
    def _close_all_positions(self):
        """Close all open positions"""
        for symbol, position in list(self.portfolio.positions.items()):
            try:
                logger.info(f"Closing position in {symbol}...")
                
                # Get current price from BINANCE
                current_price = self.binance.get_current_price(symbol)
                
                if not current_price:
                    logger.error(f"Cannot get price for {symbol}, skipping close")
                    continue
                
                order = KrakenOrder(
                    pair=symbol,
                    type='sell',
                    ordertype='market',
                    volume=position.quantity
                )
                
                # ✅ FIXED: Added current_price parameter
                result = self.kraken.place_order(order, current_price=current_price)
                
                if result['success']:
                    # Update portfolio
                    trade_result = self.portfolio.close_position(
                        symbol=symbol,
                        price=current_price,
                        fees=position.quantity * current_price * 0.0026,
                        reason='emergency_close'
                    )
                    if trade_result:
                        logger.info(f" Closed {symbol} at ${current_price:.2f}, P&L: ${trade_result['net_pnl']:.2f}")
                    else:
                        logger.error(f" Failed to close {symbol} in portfolio")
                else:
                    logger.error(f" Failed to close {symbol}: {result.get('error')}")
                
            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
    
    def _save_state(self):
        """Save current state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'running': self.running,
            'status': self.get_status(),
            'portfolio': {
                'balance': self.portfolio.cash_balance,
                'positions': [asdict(pos) for pos in self.portfolio.positions.values()]
            },
            'trades_count': len(self.trades_history)
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _print_summary(self):
        """Print trading summary"""
        logger.info("\n" + "="*80)
        logger.info("LIVE TRADING SUMMARY")
        logger.info("="*80)
        
        status = self.get_status()
        for key, value in status.items():
            logger.info(f"{key}: {value}")
        
        logger.info("="*80)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        logger.info("\nReceived stop signal, shutting down gracefully...")
        self.stop()
        sys.exit(0)


# ==================== CLI INTERFACE ====================

def start_live_trading(capital: float = 100.0):
    """Start live trading - ALWAYS REAL MONEY"""
    import os
    
    config = LiveTradingConfig(
        initial_capital=capital,
        kraken_api_key=os.getenv('KRAKEN_API_KEY'),
        kraken_api_secret=os.getenv('KRAKEN_API_SECRET'),
        dry_run=False  # Always live
    )
    
    trader = LiveTrader(config)
    trader.start()
    
    # Keep main thread alive
    try:
        while trader.running:
            time.sleep(1)
    except KeyboardInterrupt:
        trader.stop()


def stop_live_trading():
    """Stop live trading"""
    state_file = Path('logs/live_trading_state.json')
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        if state.get('running', False):
            logger.info("Sending stop signal...")
            logger.info("Please use Ctrl+C in the trading terminal")
        else:
            logger.info("Trading is not running")
    else:
        logger.info("No active trading session found")


def get_live_trading_status():
    """Get current trading status"""
    state_file = Path('logs/live_trading_state.json')
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        print("\n" + "="*80)
        print("LIVE TRADING STATUS")
        print("="*80)
        
        for key, value in state.get('status', {}).items():
            print(f"{key}: {value}")
        
        print("="*80)
    else:
        print("No active trading session found")


if __name__ == "__main__":
    # Test mode
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--start':
            start_live_trading()
        elif sys.argv[1] == '--stop':
            stop_live_trading()
        elif sys.argv[1] == '--status':
            get_live_trading_status()
    else:
        print("Usage: python live_trader.py [--start|--stop|--status]")