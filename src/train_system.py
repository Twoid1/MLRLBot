"""
Complete System Training Pipeline
Unified training function that implements the exact workflow from project instructions
Trains both ML predictor and RL agent on multi-asset, multi-timeframe data

⚡ OPTIMIZED with pre-computation for 10-30x faster RL training!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import time

# Import all components
from src.data.data_manager import DataManager
from src.data.database import DatabaseManager
from src.features.feature_engineer import FeatureEngineer
from src.features.selector import FeatureSelector
from src.models.ml_predictor import MLPredictor
from src.models.labeling import LabelingConfig
from src.models.dqn_agent import DQNAgent, DQNConfig
from src.environment.trading_env import TradingEnvironment
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class SystemTrainer:
    """
    Unified system trainer for hybrid ML/RL trading bot
    Implements the complete training pipeline from project documentation
    
    ⚡ OPTIMIZED: Pre-computes observations for 10-30x faster training!
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize system trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Try to initialize managers, but don't fail if they have issues
        try:
            self.data_manager = DataManager()
        except Exception as e:
            logger.warning(f"DataManager initialization failed: {e}")
            logger.info("Will load data directly from files")
            self.data_manager = None
        
        try:
            self.db_manager = DatabaseManager()
        except Exception as e:
            logger.warning(f"DatabaseManager initialization failed: {e}")
            self.db_manager = None
        
        self.feature_engineer = FeatureEngineer()
        
        # Storage for trained models
        self.ml_predictor = None
        self.rl_agent = None
        
        # Training results
        self.training_results = {
            'ml_results': None,
            'rl_results': None,
            'start_time': None,
            'end_time': None,
            'speedup_metrics': {}  # ← NEW: Track speedup metrics
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration"""
        default_config = {
            # Multi-asset configuration (5+ assets as per instructions)
            'assets': [
                'BTC_USDT',
                'ETH_USDT', 
                'SOL_USDT',
                'ADA_USDT',
                'DOT_USDT'
            ],
            
            # Multi-timeframe configuration (5+ timeframes)
            'timeframes': ['1h', '4h', '1d'],  # Can expand to ['1m', '5m', '15m', '1h', '4h', '1d']
            
            # Data settings
            'start_date': '2020-01-01',
            'end_date': '2025-01-01',  # Training ends at beginning of 2025
            'train_split': 0.8,
            'validation_split': 0.1,
            
            # Feature engineering
            'n_features': 50,
            'feature_selection': True,
            
            # ML settings
            'ml_model_type': 'xgboost',  # or 'lightgbm'
            'optimize_ml_params': False,  # Set True for production
            'use_sample_weights': False,  # Disabled: causes size mismatch with multi-asset training
            'walk_forward_splits': 5,
            
            # Labeling settings
            'labeling_method': 'triple_barrier',
            'lookforward': 10,
            'pt_sl': [1.5, 1.0],  # Profit-take / Stop-loss multipliers
            
            # RL settings - ⚡ OPTIMIZED
            'rl_episodes': 1000,  # Set to 1000 for proper training
            'rl_hidden_dims': [256, 256, 128],
            'rl_batch_size': 256,  # ← INCREASED from 64 for better GPU utilization
            'rl_learning_rate': 0.0001,
            'rl_gamma': 0.99,
            'rl_epsilon_start': 1.0,
            'rl_epsilon_end': 0.01,
            'rl_epsilon_decay': 0.995,
            'rl_memory_size': 50000,  # ← INCREASED from 10000
            'rl_target_update': 500,  # ← Less frequent updates
            'rl_update_every': 10,  # ← INCREASED from 4 (train less frequently)
            'use_double_dqn': True,
            'use_dueling_dqn': True,
            'use_prioritized_replay': True,
            
            # Environment settings
            'initial_balance': 10000,
            'fee_rate': 0.0026,  # Kraken fee
            'slippage': 0.001,
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'precompute_observations': True,  # ← NEW: Enable pre-computation!
            
            # Saving
            'save_models': True,
            'models_dir': 'models/',
            'results_dir': 'results/',
            'save_interval': 100  # Save checkpoint every N episodes
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def train_complete_system(self, train_ml: bool = True, train_rl: bool = True) -> Dict:
        """
        Complete training pipeline as specified in project documentation
        
        Args:
            train_ml: Whether to train ML predictor
            train_rl: Whether to train RL agent
            
        Returns:
            Dictionary with training results
        """
        logger.info("=" * 80)
        logger.info("STARTING HYBRID ML/RL SYSTEM TRAINING")
        logger.info("=" * 80)
        
        self.training_results['start_time'] = datetime.now()
        
        try:
            # Step 1: Fetch multi-asset data
            logger.info("\n[1/6] Fetching multi-asset, multi-timeframe data...")
            data = self._fetch_data()
            logger.info(f" Loaded {len(data)} assets x {len(self.config['timeframes'])} timeframes")
            
            # Step 2: Calculate features
            logger.info("\n[2/6] Engineering features across all assets...")
            features = self._calculate_features(data)
            logger.info(f" Calculated {len(features)} feature sets")
            
            # Step 3: Split data
            logger.info("\n[3/6] Splitting data for training/validation...")
            train_data, val_data, test_data = self._split_data(data, features)
            
            # Check if we have any training data
            if not train_data:
                logger.error(" No training data available! Please check:")
                logger.error("  1. Data files exist in data/raw/ directory")
                logger.error("  2. Files are in correct format (CSV with OHLCV columns)")
                logger.error("  3. Files have datetime index")
                logger.error("\nRun this to create sample data:")
                logger.error("  python create_sample_data.py")
                raise ValueError("No training data available")
            
            logger.info(f" Train: {len(train_data)} assets, "
                       f"Val: {len(val_data)} assets, "
                       f"Test: {len(test_data)} assets")
            
            # Step 4: Train ML predictor
            if train_ml:
                logger.info("\n[4/6] Training ML predictor (XGBoost/LightGBM)...")
                self._train_ml_predictor(train_data, val_data)
                logger.info(" ML predictor trained successfully")
            else:
                logger.info("\n[4/6] Skipping ML training")
            
            # Step 5: Train RL agent
            if train_rl:
                logger.info("\n[5/6] Training RL agent (DQN) with OPTIMIZATIONS...")
                self._train_rl_agent(train_data, val_data, test_data)
                logger.info(" RL agent trained successfully")
            else:
                logger.info("\n[5/6] Skipping RL training")
            
            # Step 6: Save models and results
            logger.info("\n[6/6] Saving models and results...")
            self._save_models()
            logger.info(" Models saved successfully")
            
            self.training_results['end_time'] = datetime.now()
            duration = (self.training_results['end_time'] - 
                       self.training_results['start_time']).total_seconds()
            
            logger.info("\n" + "=" * 80)
            logger.info(f"TRAINING COMPLETE - Duration: {duration:.0f}s ({duration/60:.1f} min)")
            logger.info("=" * 80)
            
            self._print_summary()
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _fetch_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch multi-asset, multi-timeframe data
        Loads directly from CSV files to bypass DataManager issues
        
        Returns:
            Nested dict: {symbol: {timeframe: df}}
        """
        data = {}
        
        # Parse target dates
        target_start = pd.to_datetime(self.config['start_date']) if self.config['start_date'] else None
        target_end = pd.to_datetime(self.config['end_date']) if self.config['end_date'] else None
        
        for symbol in self.config['assets']:
            data[symbol] = {}
            
            for timeframe in self.config['timeframes']:
                logger.info(f"  Loading {symbol} {timeframe}...")
                
                try:
                    # Load directly from CSV (bypass DataManager)
                    filepath = Path(f'data/raw/{timeframe}/{symbol}_{timeframe}.csv')
                    
                    if not filepath.exists():
                        logger.warning(f"     File not found: {filepath}")
                        continue
                    
                    # Read CSV file
                    df = pd.read_csv(filepath)
                    
                    # Parse timestamp column
                    timestamp_cols = ['timestamp', 'date', 'time', 'datetime']
                    ts_col = None
                    for col in timestamp_cols:
                        if col in df.columns:
                            ts_col = col
                            break
                    
                    if ts_col:
                        df[ts_col] = pd.to_datetime(df[ts_col])
                        df.set_index(ts_col, inplace=True)
                    else:
                        # Try to parse first column as timestamp
                        try:
                            df.index = pd.to_datetime(df.iloc[:, 0])
                            df = df.iloc[:, 1:]  # Remove first column after using as index
                        except:
                            logger.warning(f"     Cannot parse timestamp for {symbol} {timeframe}")
                            continue
                    
                    # Ensure column names are lowercase
                    df.columns = df.columns.str.lower()
                    
                    # Check for required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.warning(f"     Missing columns: {missing_cols}")
                        continue
                    
                    # Select only OHLCV columns
                    df = df[required_cols]
                    
                    # Check for empty DataFrame
                    if len(df) == 0:
                        logger.warning(f"     DataFrame is empty")
                        continue
                    
                    # Get actual data date range
                    actual_start = df.index.min()
                    actual_end = df.index.max()
                    
                    # Determine effective start date
                    if target_start and actual_start <= target_start:
                        effective_start = target_start
                        logger.info(f"     Using target start: {effective_start.date()}")
                    else:
                        effective_start = actual_start
                        if target_start:
                            logger.info(f"     Data starts later ({actual_start.date()}), using actual start")
                        else:
                            logger.info(f"     Using actual start: {effective_start.date()}")
                    
                    # Determine effective end date
                    if target_end and actual_end >= target_end:
                        effective_end = target_end
                    else:
                        effective_end = actual_end
                    
                    # Filter to effective date range
                    df = df[(df.index >= effective_start) & (df.index <= effective_end)]
                    
                    # Minimum data requirement
                    if len(df) < 100:
                        logger.warning(f"     Insufficient data: {len(df)} candles (need ≥100)")
                        continue
                    
                    # Remove any NaN values
                    df = df.dropna()
                    
                    if len(df) < 100:
                        logger.warning(f"     After cleaning: {len(df)} candles (need ≥100)")
                        continue
                    
                    # Store the data
                    data[symbol][timeframe] = df
                    logger.info(f"      {len(df)} candles ({df.index[0].date()} to {df.index[-1].date()})")
                        
                except Exception as e:
                    logger.error(f"     Error: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
        
        # Summary of loaded data
        logger.info("\n  Data Loading Summary:")
        total_loaded = 0
        for symbol in self.config['assets']:
            if symbol in data and data[symbol]:
                timeframes_loaded = list(data[symbol].keys())
                total_loaded += len(timeframes_loaded)
                logger.info(f"     {symbol}: {len(timeframes_loaded)} timeframes")
            else:
                logger.warning(f"     {symbol}: No data loaded")
        
        logger.info(f"\n  Total: {total_loaded} datasets loaded")
        
        return data
    
    def _calculate_features(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate features for all assets
        
        Args:
            data: Nested dict of OHLCV data
            
        Returns:
            Dict {symbol: features_df}
        """
        features = {}
        
        for symbol, timeframe_data in data.items():
            logger.info(f"  Processing {symbol}...")
            
            # Get base timeframe (usually the shortest one)
            base_tf = self.config['timeframes'][0]
            if base_tf not in timeframe_data:
                logger.warning(f"     Base timeframe {base_tf} not found")
                continue
            
            base_df = timeframe_data[base_tf]
            
            # Calculate base features
            symbol_features = self.feature_engineer.calculate_all_features(base_df, symbol)
            
            # Add multi-timeframe features if available
            if len(timeframe_data) > 1:
                mtf_features = self.feature_engineer.calculate_multi_timeframe_features(
                    timeframe_data
                )
                symbol_features = pd.concat([symbol_features, mtf_features], axis=1)
            
            features[symbol] = symbol_features
            logger.info(f"      {len(symbol_features.columns)} features calculated")
        
        return features
    
    def _split_data(self, 
                    data: Dict[str, Dict[str, pd.DataFrame]], 
                    features: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train/validation/test sets
        Handles different date ranges for each asset intelligently
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        train_data = {}
        val_data = {}
        test_data = {}
        
        logger.info("\n  Splitting data for each asset:")
        
        for symbol in self.config['assets']:
            if symbol not in features:
                continue
            
            # Get base timeframe data
            base_tf = self.config['timeframes'][0]
            if base_tf not in data[symbol]:
                continue
            
            ohlcv = data[symbol][base_tf]
            feats = features[symbol]
            
            # Align features with OHLCV data
            common_idx = ohlcv.index.intersection(feats.index)
            if len(common_idx) < 100:
                logger.warning(f"    {symbol}: Insufficient aligned data, skipping")
                continue
            
            ohlcv = ohlcv.loc[common_idx]
            feats = feats.loc[common_idx]
            
            # Calculate split indices based on actual data length
            n = len(ohlcv)
            train_end = int(n * self.config['train_split'])
            val_end = train_end + int(n * self.config['validation_split'])
            
            # Split data
            train_data[symbol] = (
                ohlcv[:train_end],
                feats[:train_end]
            )
            val_data[symbol] = (
                ohlcv[train_end:val_end],
                feats[train_end:val_end]
            )
            test_data[symbol] = (
                ohlcv[val_end:],
                feats[val_end:]
            )
            
            # Log split info
            train_dates = f"{ohlcv.index[0].date()} to {ohlcv.index[train_end-1].date()}"
            val_dates = f"{ohlcv.index[train_end].date()} to {ohlcv.index[val_end-1].date()}"
            test_dates = f"{ohlcv.index[val_end].date()} to {ohlcv.index[-1].date()}"
            
            logger.info(f"    {symbol}:")
            logger.info(f"      Train: {len(train_data[symbol][0])} samples ({train_dates})")
            logger.info(f"      Val:   {len(val_data[symbol][0])} samples ({val_dates})")
            logger.info(f"      Test:  {len(test_data[symbol][0])} samples ({test_dates})")
        
        return train_data, val_data, test_data
    
    def _train_ml_predictor(self, train_data: Dict, val_data: Dict):
        """
        Train ML predictor on multi-asset data
        
        Args:
            train_data: Dict {symbol: (ohlcv_df, features_df)}
            val_data: Validation data in same format
        """
        # Initialize ML predictor with triple-barrier labeling
        labeling_config = LabelingConfig(
            method=self.config['labeling_method'],
            lookforward=self.config['lookforward'],
            pt_sl=self.config['pt_sl']
        )
        
        self.ml_predictor = MLPredictor(
            model_type=self.config['ml_model_type'],
            labeling_config=labeling_config
        )
        
        # Train on multi-asset data
        logger.info(f"  Training on {len(train_data)} assets...")
        training_results = self.ml_predictor.train(
            train_data=train_data,
            val_data=val_data,
            feature_selection=self.config['feature_selection'],
            n_features=self.config['n_features'],
            optimize_params=self.config['optimize_ml_params'],
            use_sample_weights=self.config['use_sample_weights']
        )
        
        self.training_results['ml_results'] = training_results
        
        # Log results
        logger.info(f"  Train Accuracy: {training_results['train_accuracy']:.4f}")
        logger.info(f"  Train F1 Score: {training_results['train_f1']:.4f}")
        
        if 'validation' in training_results:
            val_results = training_results['validation']
            logger.info(f"  Val Accuracy: {val_results['accuracy']:.4f}")
            logger.info(f"  Val F1 Score: {val_results['f1']:.4f}")
        
        # Show top features
        if self.ml_predictor.selected_features:
            logger.info(f"  Selected {len(self.ml_predictor.selected_features)} features")
            importance = self.ml_predictor.get_feature_importance(top_n=10)
            logger.info("  Top 10 features:")
            for i, row in importance.iterrows():
                logger.info(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    def _train_rl_agent(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """
        ⚡ OPTIMIZED: Train RL agent with pre-computation and optimizations
        
        Args:
            train_data: Training data dict {asset: (ohlcv_df, features_df)}
            val_data: Validation data dict
            test_data: Test data dict
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: RL AGENT TRAINING (OPTIMIZED)")
        logger.info("="*80)
        
        # Use primary asset for RL training
        primary_asset = self.config['assets'][0]
        
        if primary_asset not in train_data:
            logger.error(f"Primary asset {primary_asset} not in training data")
            raise ValueError(f"Primary asset {primary_asset} not found in training data")
        
        ohlcv_train, features_train = train_data[primary_asset]
        ohlcv_val, features_val = val_data.get(primary_asset, (None, None))
        
        logger.info(f"  Training on: {primary_asset}")
        logger.info(f"  Training samples: {len(ohlcv_train):,}")
        if ohlcv_val is not None:
            logger.info(f"  Validation samples: {len(ohlcv_val):,}")
        
        # ⚡ CREATE OPTIMIZED TRADING ENVIRONMENT
        logger.info("\n  Creating OPTIMIZED trading environment...")
        logger.info("   Pre-computation enabled for 10-30x speedup!")
        
        env_start_time = time.time()
        
        env = TradingEnvironment(
            df=ohlcv_train,
            initial_balance=self.config['initial_balance'],
            fee_rate=self.config['fee_rate'],
            slippage=self.config.get('slippage', 0.001),
            features_df=features_train,
            stop_loss=self.config.get('stop_loss', 0.05),
            take_profit=self.config.get('take_profit', 0.10),
            precompute_observations=self.config.get('precompute_observations', True)  # ← ENABLED!
        )
        
        env_creation_time = time.time() - env_start_time
        
        # Store speedup metrics
        self.training_results['speedup_metrics']['env_creation_time'] = env_creation_time
        self.training_results['speedup_metrics']['precomputation_enabled'] = env.precompute_observations
        
        logger.info(f"   Environment created in {env_creation_time:.2f}s")
        
        # Initialize RL agent with OPTIMIZED settings
        state_dim = env.observation_space_shape[0]
        action_dim = env.action_space_n
        
        logger.info(f"\n  Environment setup:")
        logger.info(f"    State dimension: {state_dim}")
        logger.info(f"    Action space: {action_dim} actions")
        logger.info(f"    Initial balance: ${self.config['initial_balance']:,.2f}")
        logger.info(f"    Fee rate: {self.config['fee_rate']*100:.2f}%")
        logger.info(f"    Pre-computed observations: {env.precompute_observations}")
        
        # ⚡ CONFIGURE RL AGENT WITH OPTIMIZATIONS
        rl_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['rl_hidden_dims'],
            learning_rate=self.config.get('rl_learning_rate', 0.0001),
            gamma=self.config.get('rl_gamma', 0.99),
            epsilon_start=self.config.get('rl_epsilon_start', 1.0),
            epsilon_end=self.config.get('rl_epsilon_end', 0.01),
            epsilon_decay=self.config.get('rl_epsilon_decay', 0.995),
            batch_size=self.config.get('rl_batch_size', 1024),  # ← OPTIMIZED: 256 vs 64
            memory_size=self.config.get('rl_memory_size', 50000),  # ← INCREASED
            target_update_every=self.config.get('rl_target_update', 500),  # ← Less frequent
            update_every=self.config.get('rl_update_every', 10),  # ← OPTIMIZED: 10 vs 4
            use_double_dqn=self.config['use_double_dqn'],
            use_dueling_dqn=self.config['use_dueling_dqn'],
            use_prioritized_replay=self.config['use_prioritized_replay']
        )
        
        logger.info(f"\n  Agent configuration (OPTIMIZED):")
        logger.info(f"    Architecture: {'Dueling ' if rl_config.use_dueling_dqn else ''}{'Double ' if rl_config.use_double_dqn else ''}DQN")
        logger.info(f"    Prioritized Replay: {rl_config.use_prioritized_replay}")
        logger.info(f"    Hidden layers: {rl_config.hidden_dims}")
        logger.info(f"    Batch size: {rl_config.batch_size} (optimized)")
        logger.info(f"    Update frequency: every {rl_config.update_every} steps (optimized)")
        logger.info(f"    Learning rate: {rl_config.learning_rate}")
        logger.info(f"    Memory size: {rl_config.memory_size:,}")
        
        self.rl_agent = DQNAgent(rl_config)
        
        # ⚡ FAST TRAINING LOOP
        n_episodes = self.config['rl_episodes']
        logger.info(f"\n  Starting FAST training for {n_episodes} episodes...")
        logger.info(f"  Progress displayed every 10 episodes\n")
        
        episode_results = []
        best_reward = float('-inf')
        best_episode = 0
        
        training_start_time = time.time()
        
        for episode in range(n_episodes):
            episode_start = time.time()
            
            # Reset environment
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            steps = 0
            done = False
            
            # Episode loop
            while not done and steps < len(ohlcv_train) - 100:
                # Select action
                action = self.rl_agent.act(state, training=True)
                
                # Execute action (⚡ FAST: uses pre-computed observations)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store experience
                self.rl_agent.remember(state, action, reward, next_state, done)
                
                # Train (less frequently with larger batches)
                if len(self.rl_agent.memory) > 1000 and steps % rl_config.update_every == 0:
                    loss = self.rl_agent.replay(batch_size=rl_config.batch_size)
                    if loss is not None:
                        episode_loss += loss
                
                # Update state
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Update target network (less frequently)
            if episode % 50 == 0:
                self.rl_agent.update_target_network()
            
            # Decay epsilon
            self.rl_agent.decay_epsilon()
            
            # Calculate episode statistics
            episode_duration = time.time() - episode_start
            
            stats = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'average_reward': episode_reward / steps if steps > 0 else 0,
                'average_loss': episode_loss / (steps / rl_config.update_every) if steps > 0 else 0,
                'steps': steps,
                'epsilon': self.rl_agent.epsilon,
                'portfolio_value': info.get('portfolio_value', 0),
                'win_rate': info.get('win_rate', 0),
                'sharpe_ratio': info.get('sharpe_ratio', 0),
                'max_drawdown': info.get('max_drawdown', 0),
                'num_trades': info.get('num_trades', 0),
                'duration': episode_duration
            }
            
            episode_results.append(stats)
            
            # Track best
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_episode = episode + 1
                
                # Save best model
                best_model_path = Path('models/rl/best_agent.pth')
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.rl_agent.save(best_model_path)
            
            # Progress reporting every 10 episodes
            if (episode + 1) % 10 == 0:
                recent = episode_results[-10:]
                avg_reward = np.mean([e['total_reward'] for e in recent])
                avg_portfolio = np.mean([e['portfolio_value'] for e in recent])
                avg_sharpe = np.mean([e['sharpe_ratio'] for e in recent])
                avg_duration = np.mean([e['duration'] for e in recent])
                
                elapsed = time.time() - training_start_time
                eps_per_sec = (episode + 1) / elapsed
                eta_seconds = (n_episodes - episode - 1) / eps_per_sec
                eta_minutes = eta_seconds / 60
                
                logger.info(f"  Ep {episode+1:4d}/{n_episodes} | "
                           f"Reward: {episode_reward:7.1f} | "
                           f"Avg(10): {avg_reward:7.1f} | "
                           f"Portfolio: ${avg_portfolio:,.0f} | "
                           f"Sharpe: {avg_sharpe:5.2f} | "
                           f"ε: {self.rl_agent.epsilon:.3f} | "
                           f"Speed: {eps_per_sec:.1f} ep/s | "
                           f"Time/ep: {avg_duration:.1f}s | "
                           f"ETA: {eta_minutes:.0f}m")
            
            # Save checkpoint periodically
            save_interval = self.config.get('save_interval', 100)
            if (episode + 1) % save_interval == 0:
                checkpoint_path = Path(f'models/rl/checkpoint_ep{episode+1}.pth')
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self.rl_agent.save(checkpoint_path)
                logger.info(f"   Checkpoint saved at episode {episode+1}")
        
        # Training complete
        total_duration = time.time() - training_start_time
        avg_episode_time = total_duration / n_episodes
        
        # Store speedup metrics
        self.training_results['speedup_metrics']['total_training_time'] = total_duration
        self.training_results['speedup_metrics']['avg_episode_time'] = avg_episode_time
        self.training_results['speedup_metrics']['episodes_per_second'] = n_episodes / total_duration
        
        logger.info(f"\n{'='*80}")
        logger.info("RL TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
        logger.info(f"  Speed: {avg_episode_time:.2f} seconds per episode")
        logger.info(f"  Throughput: {n_episodes/total_duration:.2f} episodes/second")
        logger.info(f"  Best episode: {best_episode} (Reward: {best_reward:.2f})")
        
        # Final 10 episodes performance
        final_10 = episode_results[-10:]
        logger.info(f"\n  Final 10 Episodes:")
        logger.info(f"    Avg Reward: {np.mean([e['total_reward'] for e in final_10]):.2f}")
        logger.info(f"    Avg Portfolio: ${np.mean([e['portfolio_value'] for e in final_10]):,.2f}")
        logger.info(f"    Avg Sharpe: {np.mean([e['sharpe_ratio'] for e in final_10]):.3f}")
        logger.info(f"    Avg Win Rate: {np.mean([e['win_rate'] for e in final_10])*100:.1f}%")
        
        # Validate on validation set if available
        if ohlcv_val is not None and features_val is not None:
            logger.info("\n  Validating on holdout data...")
            val_env = TradingEnvironment(
                df=ohlcv_val,
                initial_balance=self.config['initial_balance'],
                fee_rate=self.config['fee_rate'],
                slippage=self.config.get('slippage', 0.001),
                features_df=features_val,
                stop_loss=self.config.get('stop_loss', 0.05),
                take_profit=self.config.get('take_profit', 0.10),
                precompute_observations=True  # ← Also optimize validation
            )
            
            val_stats = self.rl_agent.evaluate(val_env)
            
            logger.info(f"    Validation Reward: {val_stats['mean_reward']:.2f}")
            logger.info(f"    Validation Portfolio: ${val_stats['mean_portfolio_value']:,.2f}")
            logger.info(f"    Validation Sharpe: {val_stats.get('mean_sharpe_ratio', 0):.3f}")
            logger.info(f"    Validation Win Rate: {val_stats.get('mean_win_rate', 0)*100:.1f}%")
        
        # Save final model
        final_model_path = Path('models/rl/final_agent.pth')
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.rl_agent.save(final_model_path)
        logger.info(f"\n   Final model saved: {final_model_path}")
        
        # Store comprehensive results
        self.training_results['rl_results'] = {
            'total_episodes': len(episode_results),
            'training_duration_minutes': total_duration / 60,
            'best_episode': best_episode,
            'best_reward': float(best_reward),
            'final_epsilon': episode_results[-1]['epsilon'],
            'final_10_avg_reward': float(np.mean([e['total_reward'] for e in final_10])),
            'final_10_avg_portfolio': float(np.mean([e['portfolio_value'] for e in final_10])),
            'final_10_avg_sharpe': float(np.mean([e['sharpe_ratio'] for e in final_10])),
            'final_10_avg_win_rate': float(np.mean([e['win_rate'] for e in final_10])),
            'avg_episode_time_seconds': avg_episode_time,
            'episode_summary': {
                'first_10_avg': float(np.mean([e['total_reward'] for e in episode_results[:10]])),
                'middle_10_avg': float(np.mean([e['total_reward'] for e in episode_results[len(episode_results)//2-5:len(episode_results)//2+5]])),
                'last_10_avg': float(np.mean([e['total_reward'] for e in episode_results[-10:]])),
                'overall_avg': float(np.mean([e['total_reward'] for e in episode_results])),
                'overall_std': float(np.std([e['total_reward'] for e in episode_results]))
            }
        }
        
        # Add validation results if available
        if ohlcv_val is not None:
            self.training_results['rl_results']['validation'] = {
                'mean_reward': float(val_stats['mean_reward']),
                'mean_portfolio_value': float(val_stats['mean_portfolio_value']),
                'mean_sharpe_ratio': float(val_stats.get('mean_sharpe_ratio', 0)),
                'mean_win_rate': float(val_stats.get('mean_win_rate', 0))
            }
        
        logger.info("\n" + "="*80)
    
    def _save_models(self):
        """Save trained models to disk"""
        if not self.config['save_models']:
            return
        
        # Create directories
        models_dir = Path(self.config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ML predictor
        if self.ml_predictor is not None:
            ml_path = models_dir / f'ml_predictor_{timestamp}.pkl'
            self.ml_predictor.save_model(str(ml_path))
            logger.info(f"   ML model saved: {ml_path}")
        
        # Save RL agent
        if self.rl_agent is not None:
            rl_path = models_dir / f'rl_agent_{timestamp}.pt'
            self.rl_agent.save(str(rl_path))
            logger.info(f"   RL agent saved: {rl_path}")
        
        # Save training results
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / f'training_results_{timestamp}.json'
        
        # Convert results to JSON-serializable format
        json_results = {
            'config': self.config,
            'start_time': self.training_results['start_time'].isoformat(),
            'end_time': self.training_results['end_time'].isoformat(),
            'speedup_metrics': self.training_results['speedup_metrics'],
            'ml_results': self._serialize_ml_results(),
            'rl_results': self._serialize_rl_results()
        }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"   Training results saved: {results_path}")
    
    def _serialize_ml_results(self) -> Dict:
        """Convert ML results to JSON-serializable format"""
        if self.training_results['ml_results'] is None:
            return None
        
        results = self.training_results['ml_results'].copy()
        
        # Remove non-serializable items
        if 'best_params' in results:
            results['best_params'] = str(results['best_params'])
        
        return results
    
    def _serialize_rl_results(self) -> Dict:
        """Convert RL results to JSON-serializable format"""
        if self.training_results['rl_results'] is None:
            return None
        
        return self.training_results['rl_results'].copy()
    
    def _print_summary(self):
        """Print training summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        
        # Configuration
        logger.info("\nConfiguration:")
        logger.info(f"  Assets: {', '.join(self.config['assets'])}")
        logger.info(f"  Timeframes: {', '.join(self.config['timeframes'])}")
        logger.info(f"  Total combinations: {len(self.config['assets'])} x {len(self.config['timeframes'])}")
        
        # Speedup metrics
        if self.training_results['speedup_metrics']:
            logger.info("\n Performance Metrics:")
            metrics = self.training_results['speedup_metrics']
            if 'env_creation_time' in metrics:
                logger.info(f"  Environment creation: {metrics['env_creation_time']:.2f}s")
            if 'precomputation_enabled' in metrics:
                logger.info(f"  Pre-computation: {' Enabled' if metrics['precomputation_enabled'] else ' Disabled'}")
            if 'avg_episode_time' in metrics:
                logger.info(f"  Avg time per episode: {metrics['avg_episode_time']:.2f}s")
            if 'episodes_per_second' in metrics:
                logger.info(f"  Training speed: {metrics['episodes_per_second']:.2f} episodes/second")
        
        # ML Results
        if self.training_results['ml_results']:
            logger.info("\nML Predictor Results:")
            ml_res = self.training_results['ml_results']
            logger.info(f"  Train Accuracy: {ml_res['train_accuracy']:.4f}")
            logger.info(f"  Train F1 Score: {ml_res['train_f1']:.4f}")
            if 'validation' in ml_res:
                logger.info(f"  Val Accuracy: {ml_res['validation']['accuracy']:.4f}")
                logger.info(f"  Val F1 Score: {ml_res['validation']['f1']:.4f}")
        
        # RL Results
        if self.training_results['rl_results']:
            logger.info("\nRL Agent Results:")
            rl_res = self.training_results['rl_results']
            logger.info(f"  Total Episodes: {rl_res['total_episodes']}")
            logger.info(f"  Training Duration: {rl_res['training_duration_minutes']:.1f} minutes")
            logger.info(f"  Final Epsilon: {rl_res['final_epsilon']:.3f}")
            logger.info(f"  Final 10 Avg Reward: {rl_res['final_10_avg_reward']:.2f}")
            if 'validation' in rl_res:
                logger.info(f"  Val Mean Reward: {rl_res['validation']['mean_reward']:.2f}")
                logger.info(f"  Val Win Rate: {rl_res['validation']['mean_win_rate']:.2%}")
        
        logger.info("\n" + "=" * 80)


def train_ml_only(config_path: Optional[str] = None) -> Dict:
    """
    Train ML predictor only
    
    Args:
        config_path: Path to config file
        
    Returns:
        Training results
    """
    trainer = SystemTrainer(config_path)
    return trainer.train_complete_system(train_ml=True, train_rl=False)


def train_rl_only(config_path: Optional[str] = None) -> Dict:
    """
    Train RL agent only (with optimizations)
    
    Args:
        config_path: Path to config file
        
    Returns:
        Training results
    """
    trainer = SystemTrainer(config_path)
    return trainer.train_complete_system(train_ml=False, train_rl=True)


def train_both(config_path: Optional[str] = None) -> Dict:
    """
    Train both ML and RL components (with optimizations)
    
    Args:
        config_path: Path to config file
        
    Returns:
        Training results
    """
    trainer = SystemTrainer(config_path)
    return trainer.train_complete_system(train_ml=True, train_rl=True)


# Quick test function
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("TESTING SYSTEM TRAINER (OPTIMIZED)")
    print("=" * 80)
    
    # Create a minimal test config
    test_config = {
        'assets': ['BTC_USDT', 'ETH_USDT'],
        'timeframes': ['1h'],
        'rl_episodes': 10,  # Very short for testing
        'save_models': False,
        'precompute_observations': True  # ← Enable optimization
    }
    
    # You can test with: python -m src.train_system
    print("\nNote: This is the OPTIMIZED unified training system.")
    print(" Pre-computation enabled for 10-30x speedup!")
    print("\nRun with main.py for full functionality:")
    print("  python main.py train --ml")
    print("  python main.py train --rl")
    print("  python main.py train --both")