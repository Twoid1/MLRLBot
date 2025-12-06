"""
TRADE-BASED Training System
============================

Modified from optimized_train_system.py to work with TradeBasedMultiTimeframeEnv

KEY CHANGES:
1. Uses 2-action space (phase-dependent)
2. Episodes end when trade completes (not fixed steps)
3. Rewards only at episode end
4. Tracks trade-specific metrics
5. Supports both old and new environment types
6. MULTI-OBJECTIVE REWARDS (v4.0) - 5 separate learning signals
7. PERCENTILE ANALYSIS REPORTING - See training progression across 10 buckets

The agent learns:
- WHEN to enter (good setups)
- WHEN to exit (optimal timing)
- Trade QUALITY over QUANTITY
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import pickle
import warnings
warnings.filterwarnings('ignore')

# Progress tracking
from tqdm import tqdm
import time

# GPU support
import torch
torch.set_num_threads(8)

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Explainability import
from src.explainability_integration import ExplainableRL

# ═══════════════════════════════════════════════════════════════════════════════
# PERCENTILE REPORTER (NEW!)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from src.rl_reporter_fast import (
        FastRLTrainingReporter,
        generate_percentile_report,
        quick_percentile_summary,
        PercentileAnalyzer
    )
    PERCENTILE_REPORTER_AVAILABLE = True
except ImportError:
    PERCENTILE_REPORTER_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-OBJECTIVE EXTENSION (v4.0)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from src.multi_objective_extension import (
        MultiObjectiveDQNAgent, MODQNConfig, MORewardConfig,
        calculate_mo_rewards_from_trade, OBJECTIVES,
        MultiObjectiveRewardCalculator
    )
    MO_AVAILABLE = True
except ImportError:
    MO_AVAILABLE = False
    OBJECTIVES = []

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Real-time progress tracking for training"""
    
    def __init__(self, log_file: str = 'training_progress.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'current_stage': None,
            'overall_progress': 0.0
        }
        
    def start_stage(self, stage_name: str, total_items: int = 100):
        """Start tracking a new stage"""
        self.metrics['current_stage'] = stage_name
        self.metrics['stages'][stage_name] = {
            'start_time': time.time(),
            'total_items': total_items,
            'completed_items': 0,
            'status': 'running'
        }
        self._save()
        
    def update(self, stage_name: str, completed: int, metrics: Dict = None):
        """Update progress for a stage"""
        if stage_name in self.metrics['stages']:
            stage = self.metrics['stages'][stage_name]
            stage['completed_items'] = completed
            stage['progress'] = completed / stage['total_items'] * 100
            stage['elapsed_time'] = time.time() - stage['start_time']
            
            if metrics:
                stage['metrics'] = metrics
                
            self._save()
    
    def complete_stage(self, stage_name: str, final_metrics: Dict = None):
        """Mark stage as complete"""
        if stage_name in self.metrics['stages']:
            stage = self.metrics['stages'][stage_name]
            stage['status'] = 'completed'
            stage['end_time'] = time.time()
            stage['total_time'] = stage['end_time'] - stage['start_time']
            
            if final_metrics:
                stage['final_metrics'] = final_metrics
                
            self._save()
    
    def _save(self):
        """Save progress to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self) -> str:
        """Get progress summary"""
        elapsed = time.time() - self.start_time
        completed_stages = sum(1 for s in self.metrics['stages'].values() if s['status'] == 'completed')
        total_stages = len(self.metrics['stages'])
        
        summary = f"\n{'='*60}\n"
        summary += f"Training Progress Summary\n"
        summary += f"{'='*60}\n"
        summary += f"Elapsed Time: {elapsed/60:.1f} minutes\n"
        summary += f"Stages Completed: {completed_stages}/{total_stages}\n"
        summary += f"Current Stage: {self.metrics.get('current_stage', 'None')}\n"
        summary += f"{'='*60}\n"
        
        return summary


class TradeBasedTrainer:
    """
    Trade-Based Training System
    
    Each episode = 1 complete trade
    Agent learns trade QUALITY, not step-by-step prediction
    
    Key differences from OptimizedSystemTrainer:
    1. 2-action space (WAIT/ENTER or HOLD/EXIT depending on phase)
    2. Variable episode length (ends when trade completes)
    3. Rewards only at episode end
    4. Trade-specific metrics tracking
    5. MULTI-OBJECTIVE mode (v4.0) - 5 separate reward signals
    6. PERCENTILE ANALYSIS - See exactly where training degrades
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._validate_config()
        self.progress = ProgressTracker('logs/trade_based_training_progress.json')
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Import components (lazy loading)
        from src.features.feature_engineer import FeatureEngineer
        from src.features.selector import FeatureSelector
        from src.models.ml_predictor import MLPredictor
        from src.models.dqn_agent import DQNAgent, DQNConfig
        
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        self.ml_predictor = None
        self.rl_agent = None
        self.explainer = None
        
        # Training results
        self.training_results = {
            'ml_results': None,
            'rl_results': None,
            'trade_stats': None,
            'start_time': None,
            'end_time': None,
            'speedup_metrics': {}
        }
        
        # Trade-specific tracking
        self.trade_history = []
        self.episode_trade_results = []
        
        # ═══════════════════════════════════════════════════════════════════════
        # EPISODE RESULTS FOR PERCENTILE ANALYSIS (NEW!)
        # ═══════════════════════════════════════════════════════════════════════
        self.all_episode_results = []  # Store ALL episode stats for percentile report
        
        # ═══════════════════════════════════════════════════════════════════════
        # MULTI-OBJECTIVE MODE (v4.0)
        # ═══════════════════════════════════════════════════════════════════════
        self.use_multi_objective = False
        self.mo_reward_config = None
        self.mo_reward_calculator = None
        
        # Track per-objective rewards during training
        self.objective_rewards_history = {obj: [] for obj in OBJECTIVES} if OBJECTIVES else {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration optimized for trade-based learning"""
        config = {
            # Assets and timeframes
            'assets': ['SOL_USDT', 'AVAX_USDT', 'ADA_USDT', 'ETH_USDT', 'DOT_USDT'],
            'timeframes': ['5m', '15m', '1h'],
            'execution_timeframe': '5m',
            
            # Data settings
            'start_date': '2021-01-01',
            'end_date': '2025-10-01',
            'train_split': 0.8,
            'validation_split': 0.1,
            
            # Optimization settings
            'use_gpu': True,
            'num_workers': 8,
            'batch_size': 256,
            'cache_features': True,
            'preload_data': True,
            
            # ML settings
            'ml_model_type': 'xgboost',
            'n_features': 50,
            'feature_selection': True,
            'labeling_method': 'triple_barrier',
            'lookforward': 10,
            'pt_sl': [1.5, 1.0],
            
            # ML training options (required by train_system.py)
            'optimize_ml_params': False,
            'use_sample_weights': False,
            'walk_forward_splits': 5,
            
            # ═══════════════════════════════════════════════════════════
            # TRADE-BASED RL SETTINGS (KEY CHANGES)
            # ═══════════════════════════════════════════════════════════
            
            # Environment type
            'environment_type': 'trade_based',  # 'trade_based' or 'time_based'
            
            # ═══════════════════════════════════════════════════════════════════════
            # TRADE-BASED SETTINGS v3.0 - ASYMMETRIC REWARDS
            # ═══════════════════════════════════════════════════════════════════════
            'trade_config': {
                # Timeouts
                'max_wait_steps': 300,           # Max steps to find entry
                'max_hold_steps': 250,           # Max steps to hold trade
                
                # Penalties
                'no_trade_penalty': -2.0,        # Penalty for not trading
                'timeout_exit_penalty': -1.0,    # Penalty for forced exit
                'min_hold_before_penalty': 12,   # Min steps before no early penalty
                'early_exit_penalty_max': -1.0,  # Max penalty for immediate exit
                
                # ═══════════════════════════════════════════════════════════════════
                # ASYMMETRIC REWARD SYSTEM v3.0 (NEW!)
                # ═══════════════════════════════════════════════════════════════════
                # WINS: Scaled (bigger wins = bigger rewards)
                # LOSSES: Constant (bounded downside, removes fear)
                'use_asymmetric_rewards': True,  # Enable asymmetric system
                'constant_loss_reward': -1.0,    # All losses = -1.0 (regardless of size!)
                
                # Hold Duration Bonus (only for wins)
                'hold_duration_bonus': 0.08,     # +0.08 per step held
                'min_hold_for_bonus': 6,         # Start bonus at 6 steps (30 min)
                'max_hold_bonus': 3.0,           # Cap at 3.0
                
                # Take-profit bonus - BIG reward for letting winners run
                'take_profit_bonus': 2.0,        # +2.0 for hitting TP
                
                # Agent exit bonus - REMOVED (was causing quick exits)
                'agent_profitable_exit_bonus': 0.0,
                
                # Early Exit Multipliers (only for wins now)
                'early_winner_multiplier_min': 0.05,  # Only 5% reward for immediate exit
                'early_winner_multiplier_max': 1.0,
                'early_loser_multiplier_min': 1.0,    # Not used with asymmetric
                'early_loser_multiplier_max': 1.5,    # Not used with asymmetric
            },
            
            # RL training settings
            'rl_training_mode': 'random',
            'rl_episodes': 15000,  # More episodes for learning patience
            
            # Network architecture (same as before)
            'rl_hidden_dims': [128, 64],
            'use_double_dqn': True,
            'use_dueling_dqn': True,
            'rl_batch_size': 512,
            'rl_memory_size': 25000,
            
            # Exploration (slower decay since episodes are shorter)
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.9998,  # Slower decay
            
            # Validation
            'validation_frequency': 200,  # Check every 200 episodes
            'validation_episodes': 50,
            'early_stopping_patience': 9999,  # Effectively disabled (was 10)
            'early_stopping_min_delta': 0.1,  # Larger threshold to reduce noise
            'enable_early_stopping': False,   # Disabled by default for trading
            
            # Environment settings
            'initial_balance': 10000,
            'fee_rate': 0.001,
            'slippage': 0.0005,
            'stop_loss': 0.02,           # 3% stop loss
            'take_profit': 0.06,         # 3% take profit (was 6% - unreachable!)
            
            # Progress tracking
            'log_interval': 50,
            'save_interval': 200,
            
            # Explainability
            'explainability': {
                'enabled': False,
                'verbose': False,
                'explain_frequency': 100,
                'save_dir': 'logs/trade_explanations'
            },
            
            # ═══════════════════════════════════════════════════════════════════════
            # MULTI-OBJECTIVE RL SETTINGS (v4.0)
            # ═══════════════════════════════════════════════════════════════════════
            'use_multi_objective': False,  # Set True to enable MO rewards
            
            'mo_reward_config': {
                # Objective weights (should sum to 1.0)
                'weight_pnl_quality': 0.55,    # Maximize wins, minimize losses
                'weight_hold_duration': 0.00,   # Hold trades longer
                'weight_win_achieved': 0.25,    # Win more trades
                'weight_loss_control': 0.00,    # Cut losers early
                'weight_risk_reward': 0.20,     # Good risk/reward ratios
                
                # Settings
                'min_hold_for_bonus': 12,       # 1 hour minimum
                'target_hold_steps': 48,        # 4 hours target
                'pnl_scale': 50.0,
            },
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        
        return config
    
    def _validate_config(self):
        """Validate configuration"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATING TRADE-BASED CONFIGURATION")
        logger.info("="*60)
        
        # Check timeframes
        for tf in self.config['timeframes']:
            logger.info(f"   {tf}: configured")
        
        # Check trade config
        tc = self.config.get('trade_config', {})
        logger.info(f"\n   Trade Config:")
        logger.info(f"      Max wait steps: {tc.get('max_wait_steps', 200)}")
        logger.info(f"      Max hold steps: {tc.get('max_hold_steps', 300)}")
        logger.info(f"      Min hold for bonus: {tc.get('min_hold_for_bonus', 36)}")
        
        # Check multi-objective config
        if self.config.get('use_multi_objective', False):
            if not MO_AVAILABLE:
                logger.warning("   Multi-objective enabled but extension not found!")
                logger.warning("   Copy multi_objective_extension.py to src/")
                self.config['use_multi_objective'] = False
            else:
                logger.info(f"\n   Multi-Objective Mode: ENABLED")
                mo_cfg = self.config.get('mo_reward_config', {})
                logger.info(f"      PNL Quality weight:    {mo_cfg.get('weight_pnl_quality', 0.35)}")
                logger.info(f"      Hold Duration weight:  {mo_cfg.get('weight_hold_duration', 0.25)}")
                logger.info(f"      Win Achieved weight:   {mo_cfg.get('weight_win_achieved', 0.15)}")
                logger.info(f"      Loss Control weight:   {mo_cfg.get('weight_loss_control', 0.15)}")
                logger.info(f"      Risk Reward weight:    {mo_cfg.get('weight_risk_reward', 0.10)}")
        
        # Check percentile reporter
        if PERCENTILE_REPORTER_AVAILABLE:
            logger.info(f"\n   Percentile Reporter: AVAILABLE")
        else:
            logger.warning(f"\n   Percentile Reporter: NOT AVAILABLE")
            logger.warning(f"      Copy rl_reporter_fast.py to src/")
        
        logger.info("\n" + "="*60)
        logger.info(" CONFIGURATION VALIDATION PASSED")
        logger.info("="*60 + "\n")
    
    def train_complete_system(self, train_ml: bool = True, train_rl: bool = True) -> Dict:
        """
        Complete training pipeline for trade-based learning
        """
        logger.info("="*80)
        logger.info("TRADE-BASED HYBRID ML/RL SYSTEM TRAINING")
        logger.info("="*80)
        logger.info("  Each episode = 1 complete trade")
        logger.info("  Agent learns trade QUALITY, not step prediction")
        logger.info(f"  Device: {self.device}")
        
        if self.config.get('use_multi_objective', False) and MO_AVAILABLE:
            logger.info("  Mode: MULTI-OBJECTIVE (5 separate reward signals)")
        else:
            logger.info("  Mode: STANDARD (single reward signal)")
        
        logger.info("="*80)
        
        self.training_results['start_time'] = datetime.now()
        overall_start = time.time()
        
        try:
            # Stage 1: Data loading
            stage_start = time.time()
            self.progress.start_stage('Data Loading', len(self.config['assets']) * len(self.config['timeframes']))
            data = self._fetch_data_parallel()
            self.raw_data = data
            self.progress.complete_stage('Data Loading', {
                'datasets_loaded': sum(len(v) for v in data.values()),
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Data loaded in {time.time() - stage_start:.1f}s")
            
            # Stage 2: Feature engineering
            stage_start = time.time()
            self.progress.start_stage('Feature Engineering', len(data))
            features = self._calculate_features_parallel(data)
            self.progress.complete_stage('Feature Engineering', {
                'feature_sets': len(features),
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Features calculated in {time.time() - stage_start:.1f}s")
            
            # Stage 3: Data splitting
            stage_start = time.time()
            self.progress.start_stage('Data Splitting', len(data))
            train_data, val_data, test_data = self._split_data_fast(data, features)
            self.progress.complete_stage('Data Splitting', {
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Data split in {time.time() - stage_start:.1f}s")
            
            # Stage 4: ML training (for feature selection)
            if train_rl and not train_ml:
                ml_loaded = self._try_load_ml_model()
                if not ml_loaded:
                    logger.info("   Quick-training ML for feature selection...")
                    stage_start = time.time()
                    self.progress.start_stage('ML Feature Selection', 100)
                    self._train_ml_predictor_gpu(train_data, val_data)
                    self.progress.complete_stage('ML Feature Selection', {
                        'time_seconds': time.time() - stage_start
                    })
            elif train_ml:
                stage_start = time.time()
                self.progress.start_stage('ML Training', 100)
                self._train_ml_predictor_gpu(train_data, val_data)
                self.progress.complete_stage('ML Training', {
                    'time_seconds': time.time() - stage_start
                })
                logger.info(f" ML trained in {time.time() - stage_start:.1f}s")
            
            # Stage 5: Trade-based RL training
            if train_rl:
                stage_start = time.time()
                
                # Determine mode for stage name
                mode_str = "Multi-Objective" if self.config.get('use_multi_objective', False) and MO_AVAILABLE else "Standard"
                self.progress.start_stage(f'Trade-Based RL Training ({mode_str})', self.config['rl_episodes'])
                
                # Filter features
                train_data_filtered = self._filter_features_to_selected(train_data)
                val_data_filtered = self._filter_features_to_selected(val_data)
                test_data_filtered = self._filter_features_to_selected(test_data)
                
                # Train with trade-based episodes
                self._train_rl_agent_trade_based(train_data_filtered, val_data_filtered, test_data_filtered)
                
                self.progress.complete_stage(f'Trade-Based RL Training ({mode_str})', {
                    'time_seconds': time.time() - stage_start,
                    'episodes': self.config['rl_episodes'],
                    'multi_objective': self.use_multi_objective
                })
                logger.info(f" Trade-based RL trained in {time.time() - stage_start:.1f}s")
            
            # Stage 6: Save models
            self.training_results['end_time'] = datetime.now()
            stage_start = time.time()
            self.progress.start_stage('Saving Models', 2)
            self._save_models()
            self.progress.complete_stage('Saving Models', {
                'time_seconds': time.time() - stage_start
            })
            
            total_time = time.time() - overall_start
            
            logger.info("\n" + "="*80)
            logger.info(f" TRADE-BASED TRAINING COMPLETE")
            logger.info(f"   Total Time: {total_time/3600:.2f} hours")
            if self.use_multi_objective:
                logger.info(f"   Mode: MULTI-OBJECTIVE (5 objectives)")
            logger.info("="*80)
            
            self.training_results['speedup_metrics'] = {
                'total_time_hours': total_time / 3600,
                'multi_objective': self.use_multi_objective
            }
            
            print(self.progress.get_summary())
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    def _train_rl_agent_trade_based(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """
        Train RL agent with TRADE-BASED episodes
        
        KEY DIFFERENCE: Uses TradeBasedMultiTimeframeEnv
        - 2 actions (phase-dependent)
        - Episodes end when trade completes
        - Rewards only at episode end
        - OPTIONAL: Multi-objective rewards (v4.0)
        """
        from src.models.dqn_agent import DQNAgent, DQNConfig
        from src.data.data_manager import DataManager
        
        # Import the NEW trade-based environment
        from src.environment.trade_based_mtf_env import TradeBasedMultiTimeframeEnv, TradeConfig
        
        logger.info("\n" + "="*80)
        logger.info("TRADE-BASED RL TRAINING")
        logger.info("="*80)
        logger.info("  Episode = 1 complete trade")
        logger.info("  Actions: WAIT/ENTER (searching) or HOLD/EXIT (in trade)")
        logger.info("  Reward: Only at trade completion")
        
        # Check if multi-objective mode is enabled
        use_mo = self.config.get('use_multi_objective', False) and MO_AVAILABLE
        if use_mo:
            logger.info("  Mode: MULTI-OBJECTIVE (5 separate reward signals)")
            logger.info("    1. pnl_quality   - Maximize wins, minimize losses")
            logger.info("    2. hold_duration - Hold trades longer")
            logger.info("    3. win_achieved  - Win more trades")
            logger.info("    4. loss_control  - Cut losers early")
            logger.info("    5. risk_reward   - Good risk/reward ratios")
        else:
            logger.info("  Mode: STANDARD (single reward signal)")
        
        # [STEP 1: Prepare multi-timeframe data]
        logger.info("\n[1/5] Preparing multi-timeframe data...")
        
        data_manager = DataManager()
        asset_environments = {}
        
        if not hasattr(self, 'raw_data') or not self.raw_data:
            raise ValueError("raw_data must be stored for multi-timeframe training")
        
        for asset in train_data.keys():
            ohlcv_base, features_df = train_data[asset]
            
            if asset not in self.raw_data:
                continue
            
            raw_timeframe_data = self.raw_data[asset]
            
            logger.info(f"\n  Processing {asset}...")
            
            dataframes = {}
            features_dfs = {}
            
            for timeframe in self.config['timeframes']:
                try:
                    if timeframe not in raw_timeframe_data:
                        continue
                    
                    ohlcv_tf = raw_timeframe_data[timeframe]
                    
                    train_start = ohlcv_base.index.min()
                    train_end = ohlcv_base.index.max()
                    
                    ohlcv_tf_train = ohlcv_tf[
                        (ohlcv_tf.index >= train_start) & 
                        (ohlcv_tf.index <= train_end)
                    ]
                    
                    common_idx = ohlcv_tf_train.index.intersection(features_df.index)
                    
                    if len(common_idx) < 100:
                        continue
                    
                    dataframes[timeframe] = ohlcv_tf_train.loc[common_idx]
                    features_dfs[timeframe] = features_df.loc[common_idx]
                    
                    logger.info(f"    {timeframe}: {len(dataframes[timeframe]):,} candles")
                    
                except Exception as e:
                    logger.error(f"    Error loading {timeframe}: {e}")
                    continue
            
            if len(dataframes) != len(self.config['timeframes']):
                missing = set(self.config['timeframes']) - set(dataframes.keys())
                logger.warning(f"   Skipping {asset}: missing timeframes {missing}")
                continue
            
            asset_environments[asset] = {
                'dataframes': dataframes,
                'features_dfs': features_dfs
            }
        
        if not asset_environments:
            raise ValueError("No valid multi-timeframe environments!")
        
        logger.info(f"\n  Loaded {len(asset_environments)} assets")
        
        # [STEP 2: Create trade config and first environment]
        logger.info("\n[2/5] Creating trade-based environments...")
        
        # Create trade configuration with v3.0 asymmetric reward parameters
        tc = self.config.get('trade_config', {})
        trade_config = TradeConfig(
            # Timeouts
            max_wait_steps=tc.get('max_wait_steps', 200),
            max_hold_steps=tc.get('max_hold_steps', 300),
            
            # Penalties
            no_trade_penalty=tc.get('no_trade_penalty', -2.0),
            timeout_exit_penalty=tc.get('timeout_exit_penalty', -1.0),
            min_hold_before_penalty=tc.get('min_hold_before_penalty', 12),
            early_exit_penalty_max=tc.get('early_exit_penalty_max', -1.0),
            
            # ASYMMETRIC REWARD SYSTEM v3.0
            use_asymmetric_rewards=tc.get('use_asymmetric_rewards', True),
            constant_loss_reward=tc.get('constant_loss_reward', -1.0),
            
            # Hold Duration Bonus (only for wins)
            hold_duration_bonus=tc.get('hold_duration_bonus', 0.08),
            min_hold_for_bonus=tc.get('min_hold_for_bonus', 6),
            max_hold_bonus=tc.get('max_hold_bonus', 3.0),
            
            # Take-profit bonus
            take_profit_bonus=tc.get('take_profit_bonus', 2.0),
            
            # Agent exit bonus (disabled)
            agent_profitable_exit_bonus=tc.get('agent_profitable_exit_bonus', 0.0),
            
            # Early Exit Multipliers (only for wins now)
            early_winner_multiplier_min=tc.get('early_winner_multiplier_min', 0.05),
            early_winner_multiplier_max=tc.get('early_winner_multiplier_max', 1.0),
            early_loser_multiplier_min=tc.get('early_loser_multiplier_min', 1.0),
            early_loser_multiplier_max=tc.get('early_loser_multiplier_max', 1.5),
            
            # Trading costs
            fee_rate=self.config['fee_rate'],
            slippage=self.config.get('slippage', 0.0005)
        )
        
        # Create first environment to get dimensions
        first_asset = list(asset_environments.keys())[0]
        first_data = asset_environments[first_asset]
        
        temp_env = TradeBasedMultiTimeframeEnv(
            dataframes=first_data['dataframes'],
            features_dfs=first_data['features_dfs'],
            execution_timeframe=self.config.get('execution_timeframe', '5m'),
            initial_balance=self.config['initial_balance'],
            fee_rate=self.config['fee_rate'],
            slippage=self.config.get('slippage', 0.0005),
            stop_loss=self.config.get('stop_loss', 0.03),
            take_profit=self.config.get('take_profit', 0.03),  # 3% TP (was 6% - unreachable!)
            asset=first_asset,
            trade_config=trade_config
        )
        
        state_dim = temp_env.observation_space_shape[0]
        action_dim = temp_env.action_space_n  # Should be 2!
        
        logger.info(f"\n   Trade-based environment created")
        logger.info(f"      State dims: {state_dim}")
        logger.info(f"      Action dims: {action_dim} (phase-dependent)")
        logger.info(f"      Timeframes: {temp_env.timeframes}")
        
        # [STEP 3: Initialize agent]
        logger.info("\n[3/5] Initializing agent...")
        
        # ═══════════════════════════════════════════════════════════════════════
        # MULTI-OBJECTIVE vs STANDARD AGENT
        # ═══════════════════════════════════════════════════════════════════════
        if use_mo:
            logger.info("   Using MULTI-OBJECTIVE DQN Agent")
            
            mo_cfg = self.config.get('mo_reward_config', {})
            
            mo_dqn_config = MODQNConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=self.config['rl_hidden_dims'],
                head_hidden_dim=32,
                use_dueling=self.config['use_dueling_dqn'],
                use_double_dqn=self.config['use_double_dqn'],
                batch_size=self.config.get('rl_batch_size', 256),
                memory_size=self.config.get('rl_memory_size', 50000),
                min_memory_size=1000,
                update_every=4,
                epsilon_start=self.config.get('epsilon_start', 1.0),
                epsilon_end=self.config.get('epsilon_end', 0.05),
                epsilon_decay=self.config.get('epsilon_decay', 0.9998),
                objective_weights={
                    'pnl_quality': mo_cfg.get('weight_pnl_quality', 0.35),
                    'hold_duration': mo_cfg.get('weight_hold_duration', 0.25),
                    'win_achieved': mo_cfg.get('weight_win_achieved', 0.15),
                    'loss_control': mo_cfg.get('weight_loss_control', 0.15),
                    'risk_reward': mo_cfg.get('weight_risk_reward', 0.10),
                },
            )
            
            self.rl_agent = MultiObjectiveDQNAgent(config=mo_dqn_config)
            self.use_multi_objective = True
            
            # Store MO reward config for later use
            self.mo_reward_config = MORewardConfig(
                stop_loss_pct=self.config.get('stop_loss', 0.03),
                take_profit_pct=self.config.get('take_profit', 0.03),
                fee_rate=self.config.get('fee_rate', 0.001),
                weight_pnl_quality=mo_cfg.get('weight_pnl_quality', 0.35),
                weight_hold_duration=mo_cfg.get('weight_hold_duration', 0.25),
                weight_win_achieved=mo_cfg.get('weight_win_achieved', 0.15),
                weight_loss_control=mo_cfg.get('weight_loss_control', 0.15),
                weight_risk_reward=mo_cfg.get('weight_risk_reward', 0.10),
                min_hold_for_bonus=mo_cfg.get('min_hold_for_bonus', 12),
                target_hold_steps=mo_cfg.get('target_hold_steps', 48),
                pnl_scale=mo_cfg.get('pnl_scale', 50.0),
            )
            
            # Create reward calculator
            self.mo_reward_calculator = MultiObjectiveRewardCalculator(self.mo_reward_config)
            
            logger.info(f"   Multi-objective agent initialized")
            logger.info(f"      Objectives: {OBJECTIVES}")
            logger.info(f"      Weights: {mo_dqn_config.objective_weights}")
            
        else:
            logger.info("   Using STANDARD DQN Agent")
            
            rl_config = DQNConfig(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=self.config['rl_hidden_dims'],
                use_double_dqn=self.config['use_double_dqn'],
                use_dueling_dqn=self.config['use_dueling_dqn'],
                batch_size=self.config.get('rl_batch_size', 256),
                memory_size=self.config.get('rl_memory_size', 50000),
                epsilon_start=self.config.get('epsilon_start', 1.0),
                epsilon_end=self.config.get('epsilon_end', 0.05),
                epsilon_decay=self.config.get('epsilon_decay', 0.9998)
            )
            
            self.rl_agent = DQNAgent(config=rl_config)
            self.use_multi_objective = False
        
        logger.info(f"   Agent initialized")
        logger.info(f"      Network: {self.config['rl_hidden_dims']}")
        logger.info(f"      Device: {self.rl_agent.device}")
        logger.info(f"      Action space: {action_dim} (0=WAIT/HOLD, 1=ENTER/EXIT)")
        
        # [STEP 4: Create all environments]
        logger.info("\n[4/5] Creating environments for all assets...")
        
        environments = {}
        for asset, data in asset_environments.items():
            try:
                env = TradeBasedMultiTimeframeEnv(
                    dataframes=data['dataframes'],
                    features_dfs=data['features_dfs'],
                    execution_timeframe=self.config.get('execution_timeframe', '5m'),
                    initial_balance=self.config['initial_balance'],
                    fee_rate=self.config['fee_rate'],
                    slippage=self.config.get('slippage', 0.0005),
                    stop_loss=self.config.get('stop_loss', 0.03),
                    take_profit=self.config.get('take_profit', 0.03),  # 3% TP
                    asset=asset,
                    trade_config=trade_config
                )
                
                environments[asset] = {
                    'env': env,
                    'asset': asset,
                    'is_trade_based': True
                }
                
                logger.info(f"   {asset}: Trade-based environment created")
                
            except Exception as e:
                logger.error(f"  Error creating environment for {asset}: {e}")
                continue
        
        logger.info(f"\n   Created {len(environments)} trade-based environments")
        
        # [STEP 5: Training loop]
        logger.info("\n[5/5] Starting trade-based training...")
        logger.info("="*80)
        
        episode_results = self._train_trade_based_random(
            environments, 
            self.config['rl_episodes']
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STORE EPISODE RESULTS FOR PERCENTILE REPORT
        # ═══════════════════════════════════════════════════════════════════════
        self.all_episode_results = episode_results
        
        # Generate reports (both original and percentile)
        self._generate_trade_based_report(episode_results)
        
        logger.info("\n  Trade-based training complete!")
        logger.info(f"  Total episodes: {len(episode_results)}")
        logger.info(f"  Agent memory: {len(self.rl_agent.memory)} experiences")
        if self.use_multi_objective:
            logger.info(f"  Mode: MULTI-OBJECTIVE")

    def _train_trade_based_random(self, environments: Dict, total_episodes: int) -> list:
        """
        Train with random environment selection - TRADE-BASED VERSION
        
        Key differences from time-based:
        - Episodes end when trade completes (variable length)
        - No max_steps parameter (environment handles termination)
        - Tracks trade-specific metrics
        - OPTIONAL: Multi-objective rewards
        """
        import random
        
        episode_results = []
        env_names = list(environments.keys())
        
        # Validation tracking
        self.best_val_reward = float('-inf')
        self.patience_counter = 0
        
        mode_str = "MULTI-OBJECTIVE" if self.use_multi_objective else "STANDARD"
        logger.info(f"\n  TRADE-BASED RANDOM TRAINING MODE ({mode_str})")
        logger.info("  Each episode = 1 complete trade\n")
        
        with tqdm(total=total_episodes, desc="Trade-Based Training") as pbar:
            for episode in range(total_episodes):
                env_name = random.choice(env_names)
                env_info = environments[env_name]
                
                # Run trade-based episode
                stats = self._run_trade_based_episode(
                    env_info['env'],
                    env_info['asset'],
                    episode
                )
                episode_results.append(stats)
                
                # Update progress bar
                pbar.update(1)
                
                # Show trade result
                termination = stats.get('termination', 'unknown')
                pbar.set_postfix({
                    'asset': env_info['asset'][:4],
                    'result': termination[:8],
                    'reward': f"{stats['total_reward']:.1f}",
                    'ε': f"{stats['epsilon']:.3f}"
                })
                
                # Periodic logging
                if (episode + 1) % self.config.get('log_interval', 50) == 0:
                    self._log_trade_progress(episode_results[-50:], episode + 1, total_episodes)
                    
                    # ═══════════════════════════════════════════════════════════════
                    # INTERIM PERCENTILE SUMMARY (every 1000 episodes)
                    # ═══════════════════════════════════════════════════════════════
                    if (episode + 1) % 1000 == 0 and PERCENTILE_REPORTER_AVAILABLE and len(episode_results) > 100:
                        try:
                            quick_percentile_summary(episode_results)
                        except Exception as e:
                            logger.warning(f"Could not generate interim summary: {e}")
                
                # Validation
                if (episode + 1) % self.config.get('validation_frequency', 100) == 0:
                    val_results = self._run_trade_validation(environments, episode + 1)
                    
                    # Early stopping check (only if enabled)
                    if self.config.get('enable_early_stopping', False):
                        improvement = val_results['avg_reward'] - self.best_val_reward
                        if improvement > self.config.get('early_stopping_min_delta', 0.05):
                            self.best_val_reward = val_results['avg_reward']
                            self.patience_counter = 0
                            logger.info(f"   VALIDATION IMPROVED: {val_results['avg_reward']:.2f}")
                            
                            # Save best model
                            self._save_best_model(episode + 1)
                        else:
                            self.patience_counter += 1
                            patience = self.config.get('early_stopping_patience', 10)
                            logger.info(f"   No improvement (patience: {self.patience_counter}/{patience})")
                            
                            if self.patience_counter >= patience:
                                logger.info("  EARLY STOPPING: No improvement in validation")
                                break
                    else:
                        # Just log validation results, no early stopping
                        if val_results['avg_reward'] > self.best_val_reward:
                            self.best_val_reward = val_results['avg_reward']
                            self._save_best_model(episode + 1)
                
                # Save checkpoint
                if (episode + 1) % self.config.get('save_interval', 200) == 0:
                    self._save_checkpoint(episode + 1)
        
        return episode_results

    def _run_trade_based_episode(self, env, asset: str, episode_num: int) -> dict:
        """
        Run a single TRADE-BASED episode
        
        Key differences:
        - Episode ends when trade completes (not fixed steps)
        - Actions are phase-dependent (0=WAIT/HOLD, 1=ENTER/EXIT)
        - Reward comes only at the end
        - OPTIONAL: Multi-objective rewards stored separately
        """
        episode_start = time.time()
        
        # Reset environment (starts in SEARCHING phase)
        state = env.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        # ═══════════════════════════════════════════════════════════════
        # TRACKING FOR DIAGNOSTIC REPORT
        # ═══════════════════════════════════════════════════════════════
        search_steps = 0           # Steps spent searching for entry
        hold_steps = 0             # Steps spent holding
        search_actions = []        # Actions during search phase (0=WAIT, 1=ENTER)
        hold_actions = []          # Actions during hold phase (0=HOLD, 1=EXIT)
        entry_price = None
        exit_price = None
        in_trade = False
        
        # Multi-objective tracking
        episode_mo_rewards = {obj: 0.0 for obj in OBJECTIVES} if self.use_multi_objective else {}
        
        # Track MFE/MAE for risk-reward calculation
        max_favorable_excursion = 0.0
        max_adverse_excursion = 0.0
        
        # Safety limit (should never hit this with proper timeout config)
        max_safety_steps = 1000
        
        while not done and steps < max_safety_steps:
            # Track which phase we're in
            was_in_trade = in_trade
            
            # Get action from agent
            action = self.rl_agent.act(state, training=True)
            
            # Track actions by phase
            if not in_trade:
                search_actions.append(action)
                search_steps += 1
                if action == 1:  # ENTER
                    in_trade = True
            else:
                hold_actions.append(action)
                hold_steps += 1
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Track entry/exit prices
            if 'entry_price' in info and entry_price is None:
                entry_price = info.get('entry_price')
            if 'exit_price' in info:
                exit_price = info.get('exit_price')
            
            # Track MFE/MAE if in trade
            if in_trade and hasattr(env, 'unrealized_pnl'):
                pnl = env.unrealized_pnl / env.initial_balance if env.initial_balance > 0 else 0
                max_favorable_excursion = max(max_favorable_excursion, pnl)
                max_adverse_excursion = min(max_adverse_excursion, pnl)
            
            # ═══════════════════════════════════════════════════════════════
            # STORE EXPERIENCE (with multi-objective rewards if enabled)
            # ═══════════════════════════════════════════════════════════════
            bootstrap_done = done and not truncated
            
            if self.use_multi_objective:
                if done and env.trade_result is not None:
                    # TERMINAL STEP: Calculate proper multi-objective rewards from trade result
                    mo_rewards = self.mo_reward_calculator.calculate(
                        pnl_pct=env.trade_result.pnl_pct,
                        hold_duration=env.trade_result.hold_duration,
                        exit_reason=env.trade_result.exit_reason,
                        max_favorable_excursion=max_favorable_excursion,
                        max_adverse_excursion=max_adverse_excursion
                    )
                    
                    # Store with MO rewards
                    self.rl_agent.remember(state, action, reward, next_state, bootstrap_done, mo_rewards=mo_rewards)
                    
                    # Track for logging
                    for obj in OBJECTIVES:
                        episode_mo_rewards[obj] = mo_rewards.get(obj, 0.0)
                        if obj in self.objective_rewards_history:
                            self.objective_rewards_history[obj].append(mo_rewards.get(obj, 0.0))
                else:
                    # INTERMEDIATE STEPS: Use zero rewards for MO objectives
                    # This prevents the "same reward for all objectives" pollution
                    # The agent learns that intermediate steps are neutral,
                    # only the terminal outcome matters for each objective
                    zero_mo_rewards = {obj: 0.0 for obj in OBJECTIVES}
                    self.rl_agent.remember(state, action, reward, next_state, bootstrap_done, mo_rewards=zero_mo_rewards)
                        
            else:
                # Standard single reward (non-MO mode)
                self.rl_agent.remember(state, action, reward, next_state, bootstrap_done)
            
            # Increment step count
            self.rl_agent.step_count += 1
            
            # Training updates
            if self.rl_agent.step_count % self.rl_agent.config.update_every == 0:
                if len(self.rl_agent.memory) >= self.rl_agent.config.min_memory_size:
                    self.rl_agent.replay()
            
            # Target network update (for standard agent)
            if not self.use_multi_objective:
                if self.rl_agent.step_count % self.rl_agent.config.target_update_every == 0:
                    self.rl_agent.update_target_network(soft=True)
            
            # Epsilon decay
            if not self.use_multi_objective and not self.rl_agent.config.use_noisy_networks:
                self.rl_agent.epsilon = max(
                    self.rl_agent.config.epsilon_end,
                    self.rl_agent.epsilon * self.rl_agent.config.epsilon_decay
                )
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # Episode count (for standard agent)
        if hasattr(self.rl_agent, 'episode_count'):
            self.rl_agent.episode_count += 1
        
        # Build comprehensive stats
        stats = {
            'episode': episode_num + 1,
            'asset': asset,
            'total_reward': episode_reward,
            'steps': steps,
            'epsilon': self.rl_agent.epsilon,
            'episode_time': time.time() - episode_start,
            'termination': info.get('termination', 'unknown'),
            
            # Diagnostic data
            'search_steps': search_steps,
            'hold_steps': hold_steps,
            'search_wait_count': search_actions.count(0) if search_actions else 0,
            'search_enter_count': search_actions.count(1) if search_actions else 0,
            'hold_hold_count': hold_actions.count(0) if hold_actions else 0,
            'hold_exit_count': hold_actions.count(1) if hold_actions else 0,
            'entered_trade': in_trade or (entry_price is not None),
            
            # Multi-objective
            'multi_objective': self.use_multi_objective,
        }
        
        # Add MO rewards to stats
        if self.use_multi_objective:
            stats['mo_rewards'] = episode_mo_rewards
        
        # Add trade result if available
        if env.trade_result is not None:
            tr = env.trade_result
            stats['trade_pnl'] = tr.net_pnl
            stats['trade_pnl_pct'] = tr.pnl_pct
            stats['hold_duration'] = tr.hold_duration
            stats['exit_reason'] = tr.exit_reason
            stats['entry_price'] = tr.entry_price
            stats['exit_price'] = tr.exit_price
            stats['fees_paid'] = tr.fees_paid
            stats['gross_pnl'] = tr.gross_pnl
            
            # ═══════════════════════════════════════════════════════════════
            # ADD win_rate FOR PERCENTILE REPORTER
            # ═══════════════════════════════════════════════════════════════
            stats['win_rate'] = 1.0 if tr.pnl_pct > 0 else 0.0
            stats['num_trades'] = 1
            stats['pnl'] = tr.net_pnl
            
            # Add trade to trades list (for reporter compatibility)
            stats['trades'] = [{
                'pnl': tr.net_pnl,
                'pnl_pct': tr.pnl_pct,
                'hold_duration': tr.hold_duration,
                'entry_price': tr.entry_price,
                'exit_price': tr.exit_price,
                'exit_reason': tr.exit_reason,
                'fees': tr.fees_paid,
            }]
            
            # Track for analysis
            self.episode_trade_results.append({
                'episode': episode_num + 1,
                'asset': asset,
                'net_pnl': tr.net_pnl,
                'pnl_pct': tr.pnl_pct,
                'hold_duration': tr.hold_duration,
                'exit_reason': tr.exit_reason,
                'entry_price': tr.entry_price,
                'exit_price': tr.exit_price,
                'fees_paid': tr.fees_paid,
                'gross_pnl': tr.gross_pnl,
            })
        else:
            # No trade result - still add fields for percentile reporter
            stats['win_rate'] = 0.0
            stats['num_trades'] = 0
            stats['pnl'] = 0.0
            stats['trades'] = []
        
        return stats

    def _run_trade_validation(self, environments: Dict, episode_num: int) -> dict:
        """
        Run validation episodes (greedy, no exploration)
        """
        import random
        
        val_episodes = self.config.get('validation_episodes', 50)
        val_results = []
        trade_results = []
        mo_rewards_list = []
        
        # Save epsilon and set to 0
        original_epsilon = self.rl_agent.epsilon
        self.rl_agent.epsilon = 0.0
        
        logger.info(f"\n Running trade-based validation ({val_episodes} episodes)...")
        
        env_names = list(environments.keys())
        
        for _ in range(val_episodes):
            env_name = random.choice(env_names)
            env_info = environments[env_name]
            env = env_info['env']
            
            # Run episode without training
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            # Track MFE/MAE for MO rewards
            max_favorable_excursion = 0.0
            max_adverse_excursion = 0.0
            
            while not done and steps < 1000:
                action = self.rl_agent.act(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                
                # Track MFE/MAE
                if hasattr(env, 'unrealized_pnl') and env.position != 0:
                    pnl = env.unrealized_pnl / env.initial_balance if env.initial_balance > 0 else 0
                    max_favorable_excursion = max(max_favorable_excursion, pnl)
                    max_adverse_excursion = min(max_adverse_excursion, pnl)
                
                if done or truncated:
                    break
            
            val_results.append(episode_reward)
            
            # Track trade results AND calculate MO rewards
            if env.trade_result is not None:
                trade_results.append({
                    'pnl_pct': env.trade_result.pnl_pct,
                    'hold_duration': env.trade_result.hold_duration,
                    'exit_reason': env.trade_result.exit_reason,
                })
                
                # ═══════════════════════════════════════════════════════════════
                # CALCULATE MO REWARDS FOR VALIDATION
                # ═══════════════════════════════════════════════════════════════
                if self.use_multi_objective and self.mo_reward_calculator:
                    mo_rewards = self.mo_reward_calculator.calculate(
                        pnl_pct=env.trade_result.pnl_pct,
                        hold_duration=env.trade_result.hold_duration,
                        exit_reason=env.trade_result.exit_reason,
                        max_favorable_excursion=max_favorable_excursion,
                        max_adverse_excursion=max_adverse_excursion
                    )
                    mo_rewards_list.append(mo_rewards)
        
        # Restore epsilon
        self.rl_agent.epsilon = original_epsilon
        
        # Calculate stats
        avg_reward = np.mean(val_results)
        std_reward = np.std(val_results)
        
        # Trade-specific stats
        if trade_results:
            profitable = sum(1 for t in trade_results if t.get('pnl_pct', 0) > 0)
            win_rate = profitable / len(trade_results)
            avg_pnl = np.mean([t.get('pnl_pct', 0) for t in trade_results])
            avg_hold = np.mean([t.get('hold_duration', 0) for t in trade_results])
        else:
            win_rate = 0
            avg_pnl = 0
            avg_hold = 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE MO WEIGHTED REWARD
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_multi_objective and mo_rewards_list:
            weights = self.rl_agent.config.objective_weights
            weighted_totals = []
            for mo_r in mo_rewards_list:
                total = sum(mo_r.get(obj, 0) * weights.get(obj, 0) for obj in OBJECTIVES)
                weighted_totals.append(total)
            
            avg_mo_reward = np.mean(weighted_totals)
            std_mo_reward = np.std(weighted_totals)
            
            logger.info(f"  Validation Results (Episode {episode_num}):")
            logger.info(f"    MO Weighted Reward: {avg_mo_reward:+.3f} ± {std_mo_reward:.3f}")
            logger.info(f"    Win Rate: {win_rate:.1%}")
            logger.info(f"    Avg P&L: {avg_pnl*100:.2f}%")
            logger.info(f"    Avg Hold: {avg_hold:.0f} steps")
            
            # Per-objective breakdown
            logger.info(f"    PER-OBJECTIVE (validation):")
            for obj in OBJECTIVES:
                obj_rewards = [mo_r.get(obj, 0) for mo_r in mo_rewards_list]
                avg_obj = np.mean(obj_rewards)
                weight = weights.get(obj, 0)
                contribution = avg_obj * weight
                logger.info(f"      {obj:15s}: {avg_obj:+.3f} × {weight:.2f} = {contribution:+.4f}")
            
            # Return MO reward for early stopping comparison
            return {
                'avg_reward': avg_mo_reward,
                'std_reward': std_mo_reward,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'avg_hold_duration': avg_hold,
                'episode': episode_num
            }
        else:
            # Fallback for non-MO mode
            logger.info(f"  Validation Results (Episode {episode_num}):")
            logger.info(f"    Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
            logger.info(f"    Win Rate: {win_rate:.1%}")
            logger.info(f"    Avg P&L: {avg_pnl*100:.2f}%")
            logger.info(f"    Avg Hold: {avg_hold:.0f} steps")
            
            return {
                'avg_reward': avg_reward,
                'std_reward': std_reward,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'avg_hold_duration': avg_hold,
                'episode': episode_num
            }

    def _log_trade_progress(self, recent_episodes: list, current: int, total: int):
        """Log trade-based training progress"""
        
        # Trade-specific stats
        trades_with_results = [e for e in recent_episodes if 'trade_pnl_pct' in e]
        if trades_with_results:
            profitable = sum(1 for e in trades_with_results if e['trade_pnl_pct'] > 0)
            win_rate = profitable / len(trades_with_results)
            avg_pnl = np.mean([e['trade_pnl_pct'] for e in trades_with_results]) * 100
            avg_hold = np.mean([e.get('hold_duration', 0) for e in trades_with_results])
        else:
            win_rate = 0
            avg_pnl = 0
            avg_hold = 0
        
        # Count termination types
        terminations = {}
        for e in recent_episodes:
            term = e.get('termination', 'unknown')
            terminations[term] = terminations.get(term, 0) + 1
        
        logger.info(f"\n   Progress: Episode {current}/{total}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # FIXED: Show MO weighted reward instead of environment reward
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_multi_objective and OBJECTIVES:
            mo_episodes = [e for e in recent_episodes if e.get('mo_rewards')]
            if mo_episodes:
                weights = self.rl_agent.config.objective_weights
                weighted_totals = []
                for e in mo_episodes:
                    mo_r = e['mo_rewards']
                    total = sum(mo_r.get(obj, 0) * weights.get(obj, 0) for obj in OBJECTIVES)
                    weighted_totals.append(total)
                
                avg_mo_reward = np.mean(weighted_totals)
                logger.info(f"    MO Weighted Reward: {avg_mo_reward:+.3f}")
            else:
                avg_mo_reward = 0
                logger.info(f"    MO Weighted Reward: N/A (no MO data)")
        else:
            # Fallback to old environment reward for non-MO mode
            avg_reward = np.mean([e['total_reward'] for e in recent_episodes])
            logger.info(f"    Avg Reward: {avg_reward:.2f}")
        
        logger.info(f"    Win Rate: {win_rate:.1%}")
        logger.info(f"    Avg P&L: {avg_pnl:.2f}%")
        logger.info(f"    Avg Hold: {avg_hold:.0f} steps")
        logger.info(f"    Epsilon: {self.rl_agent.epsilon:.3f}")
        logger.info(f"    Memory: {len(self.rl_agent.memory):,}")
        logger.info(f"    Terminations: {terminations}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # MULTI-OBJECTIVE SPECIFIC LOGGING
        # ═══════════════════════════════════════════════════════════════════════
        if self.use_multi_objective and OBJECTIVES:
            mo_episodes = [e for e in recent_episodes if e.get('mo_rewards')]
            if mo_episodes:
                logger.info(f"    PER-OBJECTIVE REWARDS (last {len(mo_episodes)} episodes):")
                for obj in OBJECTIVES:
                    rewards = [e['mo_rewards'].get(obj, 0) for e in mo_episodes if 'mo_rewards' in e]
                    if rewards:
                        avg_obj = np.mean(rewards)
                        weight = self.rl_agent.config.objective_weights.get(obj, 0)
                        contribution = avg_obj * weight
                        logger.info(f"      {obj:15s}: {avg_obj:+.3f} × {weight:.2f} = {contribution:+.4f}")
            
            # Get loss stats from agent
            if hasattr(self.rl_agent, 'get_loss_stats'):
                loss_stats = self.rl_agent.get_loss_stats()
                if loss_stats and any(v > 0 for v in loss_stats.values()):
                    logger.info(f"    PER-OBJECTIVE LOSSES:")
                    for obj, loss in loss_stats.items():
                        logger.info(f"      {obj:15s}: {loss:.4f}")

    def _generate_trade_based_report(self, episode_results: list):
        """
        Generate COMPREHENSIVE trade-based training diagnostic report
        
        NOW INCLUDES:
        - Original diagnostic report
        - ⭐ NEW: Percentile analysis report (0-10%, 10-20%, ..., 90-100%)
        """
        from datetime import datetime
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("COMPREHENSIVE TRADE-BASED TRAINING DIAGNOSTIC REPORT")
        report_lines.append("="*100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Mode: {'MULTI-OBJECTIVE' if self.use_multi_objective else 'STANDARD'}")
        report_lines.append("")
        
        # Filter to trades with results
        trades = [e for e in episode_results if 'trade_pnl_pct' in e]
        total_episodes = len(episode_results)
        
        if not trades:
            report_lines.append("No completed trades to analyze.")
            self._save_report(report_lines)
            return
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 1: EXECUTIVE SUMMARY
        # ═══════════════════════════════════════════════════════════════════════════
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("| EXECUTIVE SUMMARY" + " "*80 + "|")
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("")
        
        profitable = sum(1 for t in trades if t['trade_pnl_pct'] > 0)
        win_rate = profitable / len(trades)
        avg_pnl = np.mean([t['trade_pnl_pct'] for t in trades]) * 100
        total_pnl = sum(t['trade_pnl_pct'] for t in trades) * 100
        avg_hold = np.mean([t.get('hold_duration', 0) for t in trades])
        
        # Calculate avg win and avg loss
        winners = [t['trade_pnl_pct'] * 100 for t in trades if t['trade_pnl_pct'] > 0]
        losers = [t['trade_pnl_pct'] * 100 for t in trades if t['trade_pnl_pct'] <= 0]
        
        avg_win = np.mean(winners) if winners else 0
        avg_loss = np.mean(losers) if losers else 0
        
        # Risk/Reward ratio
        rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Expectancy per trade
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        report_lines.append(f"  Total Trades:      {len(trades):,}")
        report_lines.append(f"  Win Rate:          {win_rate:.1%}")
        report_lines.append(f"  Avg P&L:           {avg_pnl:+.2f}%")
        report_lines.append(f"  Total P&L:         {total_pnl:+.2f}%")
        report_lines.append(f"  Avg Hold:          {avg_hold:.1f} steps ({avg_hold*5:.0f} min)")
        report_lines.append("")
        report_lines.append(f"  Avg Win:           {avg_win:+.2f}%")
        report_lines.append(f"  Avg Loss:          {avg_loss:+.2f}%")
        report_lines.append(f"  Risk/Reward:       {rr_ratio:.2f}")
        report_lines.append(f"  Expectancy:        {expectancy:+.3f}% per trade")
        report_lines.append("")
        
        if expectancy > 0.05:
            report_lines.append("    VERDICT: Positive expectancy - system is profitable!")
        elif expectancy > -0.05:
            report_lines.append("    VERDICT: Near breakeven - needs optimization")
        else:
            report_lines.append("    VERDICT: Negative expectancy - significant changes needed")
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION: MULTI-OBJECTIVE ANALYSIS (if enabled)
        # ═══════════════════════════════════════════════════════════════════════════
        if self.use_multi_objective and OBJECTIVES:
            report_lines.append("|" + "="*98 + "|")
            report_lines.append("| MULTI-OBJECTIVE ANALYSIS" + " "*73 + "|")
            report_lines.append("|" + "="*98 + "|")
            report_lines.append("")
            
            report_lines.append("  Objective Weights:")
            mo_cfg = self.config.get('mo_reward_config', {})
            for obj in OBJECTIVES:
                weight_key = f'weight_{obj}'
                weight = mo_cfg.get(weight_key, 0.2)
                report_lines.append(f"    {obj:15s}: {weight:.2f}")
            report_lines.append("")
            
            # Per-objective performance over training
            mo_episodes = [e for e in trades if e.get('mo_rewards')]
            if mo_episodes:
                report_lines.append("  Per-Objective Average Rewards:")
                for obj in OBJECTIVES:
                    rewards = [e['mo_rewards'].get(obj, 0) for e in mo_episodes]
                    avg = np.mean(rewards)
                    std = np.std(rewards)
                    report_lines.append(f"    {obj:15s}: {avg:+.3f} ± {std:.3f}")
                report_lines.append("")
            
            # Loss stats
            if hasattr(self.rl_agent, 'get_loss_stats'):
                loss_stats = self.rl_agent.get_loss_stats()
                if loss_stats:
                    report_lines.append("  Final Per-Objective Losses:")
                    for obj, loss in loss_stats.items():
                        report_lines.append(f"    {obj:15s}: {loss:.6f}")
            report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # REMAINING SECTIONS
        # ═══════════════════════════════════════════════════════════════════════════
        
        # P&L Distribution
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("| P&L DISTRIBUTION ANALYSIS" + " "*72 + "|")
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("")
        
        pnl_values = [t['trade_pnl_pct'] * 100 for t in trades]
        report_lines.append("  P&L Percentiles:")
        report_lines.append(f"    5th:   {np.percentile(pnl_values, 5):+.2f}%")
        report_lines.append(f"    25th:  {np.percentile(pnl_values, 25):+.2f}%")
        report_lines.append(f"    50th:  {np.percentile(pnl_values, 50):+.2f}%")
        report_lines.append(f"    75th:  {np.percentile(pnl_values, 75):+.2f}%")
        report_lines.append(f"    95th:  {np.percentile(pnl_values, 95):+.2f}%")
        report_lines.append("")
        
        # Hold Duration
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("| HOLD DURATION ANALYSIS" + " "*75 + "|")
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("")
        
        hold_durations = [t.get('hold_duration', 0) for t in trades]
        report_lines.append(f"  Min Hold:    {min(hold_durations)} steps")
        report_lines.append(f"  Max Hold:    {max(hold_durations)} steps")
        report_lines.append(f"  Avg Hold:    {np.mean(hold_durations):.1f} steps")
        report_lines.append(f"  Median Hold: {np.median(hold_durations):.1f} steps")
        report_lines.append("")
        
        # Winner vs Loser holds
        winner_holds = [t.get('hold_duration', 0) for t in trades if t['trade_pnl_pct'] > 0]
        loser_holds = [t.get('hold_duration', 0) for t in trades if t['trade_pnl_pct'] <= 0]
        if winner_holds and loser_holds:
            report_lines.append(f"  Avg Hold (Winners): {np.mean(winner_holds):.1f} steps")
            report_lines.append(f"  Avg Hold (Losers):  {np.mean(loser_holds):.1f} steps")
            if np.mean(winner_holds) > np.mean(loser_holds):
                report_lines.append("   GOOD: Letting winners run, cutting losers")
            else:
                report_lines.append("   WARNING: Holding losers longer than winners")
        report_lines.append("")
        
        # Exit Behavior
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("| EXIT BEHAVIOR ANALYSIS" + " "*75 + "|")
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("")
        
        exit_reasons = {}
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            if reason not in exit_reasons:
                exit_reasons[reason] = []
            exit_reasons[reason].append(t)
        
        report_lines.append("  Exit Reason Breakdown:")
        for reason in ['agent', 'stop_loss', 'take_profit', 'timeout', 'unknown']:
            if reason in exit_reasons:
                group = exit_reasons[reason]
                count = len(group)
                pct = count / len(trades) * 100
                avg_pnl_r = np.mean([t['trade_pnl_pct'] * 100 for t in group])
                report_lines.append(f"    {reason:14s}: {count:5d} ({pct:5.1f}%)  Avg P&L: {avg_pnl_r:+.2f}%")
        report_lines.append("")
        
        # Per-Asset
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("| PER-ASSET ANALYSIS" + " "*79 + "|")
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("")
        
        by_asset = {}
        for t in trades:
            asset = t.get('asset', 'unknown')
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(t)
        
        report_lines.append("  Asset         Trades    Win%   Avg P&L")
        report_lines.append("  " + "-"*45)
        for asset in sorted(by_asset.keys()):
            asset_trades = by_asset[asset]
            a_profitable = sum(1 for t in asset_trades if t['trade_pnl_pct'] > 0)
            a_win_rate = a_profitable / len(asset_trades)
            a_avg_pnl = np.mean([t['trade_pnl_pct'] * 100 for t in asset_trades])
            report_lines.append(f"  {asset:12s} {len(asset_trades):6d}  {a_win_rate:5.1%}   {a_avg_pnl:+.2f}%")
        report_lines.append("")
        
        # Final
        report_lines.append("="*100)
        report_lines.append("END OF DIAGNOSTIC REPORT")
        report_lines.append("="*100)
        
        # Save original report
        self._save_report(report_lines)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ⭐ GENERATE PERCENTILE ANALYSIS REPORT (NEW!)
        # ═══════════════════════════════════════════════════════════════════════════
        self._generate_percentile_report(episode_results)
    
    def _generate_percentile_report(self, episode_results: list):
        """
        Generate the NEW percentile analysis report
        
        Shows all metrics broken down by training progress:
        - 0-10%, 10-20%, ..., 90-100%
        
        Helps identify WHERE training starts to degrade
        """
        if not PERCENTILE_REPORTER_AVAILABLE:
            logger.warning("Percentile reporter not available. Copy rl_reporter_fast.py to src/")
            return
        
        if len(episode_results) < 10:
            logger.warning("Not enough episodes for percentile analysis (need at least 10)")
            return
        
        logger.info("\n" + "="*80)
        logger.info("GENERATING PERCENTILE ANALYSIS REPORT")
        logger.info("="*80)
        
        try:
            mode_suffix = "_MO" if self.use_multi_objective else ""
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            report_path = f"results/percentile_report{mode_suffix}_{timestamp}.txt"
            
            # Generate the report
            report = generate_percentile_report(
                episode_results=episode_results,
                env=None,
                agent=self.rl_agent,
                config=self.config,
                save_path=report_path,
                detail_level='full'
            )
            
            # Also print to console
            print("\n" + "="*120)
            print("PERCENTILE ANALYSIS - TRAINING PROGRESSION")
            print("="*120)
            print(report)
            
            logger.info(f"\n  📊 Percentile report saved to: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate percentile report: {e}", exc_info=True)
    
    def _save_report(self, report_lines: list):
        """Save report to console and file"""
        # Log to console
        for line in report_lines:
            logger.info(line)
        
        # Save to file
        mode_suffix = "_MO" if self.use_multi_objective else ""
        report_path = Path('results') / f'trade_based_diagnostic_report{mode_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"\n  📊 Report saved to: {report_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _try_load_ml_model(self) -> bool:
        """Try to load existing ML model"""
        try:
            import glob
            
            ml_models = glob.glob('models/ml/ml_predictor_*.pkl')
            if not ml_models:
                return False
            
            latest_model = max(ml_models, key=lambda x: Path(x).stat().st_mtime)
            
            from src.models.ml_predictor import MLPredictor
            from src.models.labeling import LabelingConfig
            
            labeling_config = LabelingConfig(
                method=self.config['labeling_method'],
                lookforward=self.config['lookforward'],
                pt_sl=self.config['pt_sl']
            )
            
            self.ml_predictor = MLPredictor(
                model_type=self.config['ml_model_type'],
                labeling_config=labeling_config
            )
            
            self.ml_predictor.load_model(latest_model)
            logger.info(f" Loaded ML model: {latest_model}")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
            return False
    
    def _fetch_data_parallel(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load data in parallel"""
        data = {}
        tasks = []
        
        for symbol in self.config['assets']:
            for timeframe in self.config['timeframes']:
                tasks.append((symbol, timeframe))
        
        logger.info(f"Loading {len(tasks)} datasets in parallel...")
        
        with ProcessPoolExecutor(max_workers=self.config['num_workers']) as executor:
            future_to_task = {
                executor.submit(self._load_single_dataset, symbol, timeframe): (symbol, timeframe)
                for symbol, timeframe in tasks
            }
            
            with tqdm(total=len(tasks), desc="Loading data") as pbar:
                for future in as_completed(future_to_task):
                    symbol, timeframe = future_to_task[future]
                    try:
                        df = future.result()
                        if df is not None and not df.empty:
                            if symbol not in data:
                                data[symbol] = {}
                            data[symbol][timeframe] = df
                    except Exception as e:
                        logger.error(f"Failed to load {symbol} {timeframe}: {e}")
                    
                    pbar.update(1)
        
        return data
    
    @staticmethod
    def _load_single_dataset(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load single dataset"""
        try:
            filepath = Path(f'data/raw/{timeframe}/{symbol}_{timeframe}.csv')
            if not filepath.exists():
                return None
            
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df[df.index >= pd.to_datetime('2021-01-01')]
            df.columns = df.columns.str.lower()
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None
            
            return df
            
        except Exception:
            return None
    
    def _calculate_features_parallel(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """Calculate features in parallel"""
        features = {}
        
        cache_dir = Path('data/features_cache')
        if self.config['cache_features'] and cache_dir.exists():
            for symbol in data.keys():
                cache_file = cache_dir / f'{symbol}_features.pkl'
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        features[symbol] = pickle.load(f)
        
        symbols_to_process = [s for s in data.keys() if s not in features]
        
        if symbols_to_process:
            with ProcessPoolExecutor(max_workers=self.config['num_workers']) as executor:
                future_to_symbol = {
                    executor.submit(self._calculate_features_for_symbol, symbol, data[symbol]): symbol
                    for symbol in symbols_to_process
                }
                
                with tqdm(total=len(symbols_to_process), desc="Calculating features") as pbar:
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            symbol_features = future.result()
                            features[symbol] = symbol_features
                            
                            if self.config['cache_features']:
                                cache_dir.mkdir(parents=True, exist_ok=True)
                                cache_file = cache_dir / f'{symbol}_features.pkl'
                                with open(cache_file, 'wb') as f:
                                    pickle.dump(symbol_features, f)
                                    
                        except Exception as e:
                            logger.error(f"Failed features for {symbol}: {e}")
                        
                        pbar.update(1)
        
        return features
    
    @staticmethod
    def _calculate_features_for_symbol(symbol: str, timeframe_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate features for single symbol"""
        from src.features.feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer()
        
        timeframes = list(timeframe_data.keys())
        base_tf = timeframes[0]
        base_df = timeframe_data[base_tf]
        
        features = feature_engineer.calculate_all_features(base_df, symbol)
        
        if len(timeframe_data) > 1:
            mtf_features = feature_engineer.calculate_multi_timeframe_features(timeframe_data)
            features = pd.concat([features, mtf_features], axis=1)
        
        return features
    
    def _split_data_fast(self, data: Dict, features: Dict) -> Tuple[Dict, Dict, Dict]:
        """Fast data splitting with buffers"""
        train_data, val_data, test_data = {}, {}, {}
        
        lookforward = self.config.get('lookforward', 10)
        buffer_size = self.config.get('buffer_size', 5)
        
        for symbol in self.config['assets']:
            if symbol not in features:
                continue
            
            base_tf = self.config['timeframes'][0]
            if base_tf not in data[symbol]:
                continue
            
            ohlcv = data[symbol][base_tf]
            feats = features[symbol]
            
            common_idx = ohlcv.index.intersection(feats.index)
            if len(common_idx) < 100:
                continue
            
            ohlcv = ohlcv.loc[common_idx]
            feats = feats.loc[common_idx]
            
            n = len(ohlcv)
            
            train_end = int(n * self.config['train_split']) - lookforward
            val_start = train_end + buffer_size
            val_end = val_start + int(n * self.config['validation_split']) - lookforward
            test_start = val_end + buffer_size
            
            if train_end < 100 or val_start >= val_end or test_start >= n:
                continue
            
            train_data[symbol] = (ohlcv.iloc[:train_end], feats.iloc[:train_end])
            val_data[symbol] = (ohlcv.iloc[val_start:val_end], feats.iloc[val_start:val_end])
            test_data[symbol] = (ohlcv.iloc[test_start:], feats.iloc[test_start:])
        
        return train_data, val_data, test_data
    
    def _train_ml_predictor_gpu(self, train_data: Dict, val_data: Dict):
        """Train ML model"""
        from src.train_system import SystemTrainer as StandardTrainer
        
        standard_trainer = StandardTrainer()
        standard_trainer.config = self.config
        
        standard_trainer._train_ml_predictor(train_data, val_data)
        
        self.ml_predictor = standard_trainer.ml_predictor
        self.training_results['ml_results'] = standard_trainer.training_results.get('ml_results', {})
    
    def _filter_features_to_selected(self, data_dict: Dict) -> Dict:
        """Filter features to ML-selected features"""
        if not hasattr(self, 'ml_predictor') or self.ml_predictor is None:
            return data_dict
        
        if not hasattr(self.ml_predictor, 'selected_features') or not self.ml_predictor.selected_features:
            return data_dict
        
        selected_features = self.ml_predictor.selected_features
        filtered_data = {}
        
        for asset, (ohlcv_df, features_df) in data_dict.items():
            available_features = [f for f in selected_features if f in features_df.columns]
            filtered_features = features_df[available_features].copy()
            filtered_data[asset] = (ohlcv_df, filtered_features)
        
        return filtered_data
    
    def _save_best_model(self, episode: int):
        """Save best model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "_MO" if self.use_multi_objective else ""
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based{mode_suffix}_best_{timestamp}_ep{episode}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
            logger.info(f"  Saved best model: {rl_path}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        mode_suffix = "_MO" if self.use_multi_objective else ""
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based{mode_suffix}_checkpoint_ep{episode}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
    
    def _save_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode_suffix = "_MO" if self.use_multi_objective else ""
        
        if self.ml_predictor:
            ml_path = Path(f'models/ml/ml_predictor_{timestamp}.pkl')
            ml_path.parent.mkdir(parents=True, exist_ok=True)
            self.ml_predictor.save_model(ml_path)
            logger.info(f"Saved ML model: {ml_path}")
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based{mode_suffix}_agent_{timestamp}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
            logger.info(f"Saved RL agent: {rl_path}")
        
        # Save trade history
        if self.episode_trade_results:
            trades_path = Path(f'results/trade_history{mode_suffix}_{timestamp}.json')
            trades_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trades_path, 'w') as f:
                json.dump(self.episode_trade_results, f, indent=2, default=str)
            logger.info(f"Saved trade history: {trades_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_trade_based(config_path: Optional[str] = None, multi_objective: bool = False) -> Dict:
    """
    Train with trade-based episodes (recommended)
    """
    trainer = TradeBasedTrainer(config_path)
    
    if multi_objective:
        trainer.config['use_multi_objective'] = True
    
    return trainer.train_complete_system(train_ml=True, train_rl=True)


def train_trade_based_rl_only(config_path: Optional[str] = None, multi_objective: bool = False) -> Dict:
    """
    Trade-based RL training only (uses existing ML model)
    """
    trainer = TradeBasedTrainer(config_path)
    
    if multi_objective:
        trainer.config['use_multi_objective'] = True
    
    return trainer.train_complete_system(train_ml=False, train_rl=True)


if __name__ == "__main__":
    import logging
    import argparse
    from src.utils.logger import setup_logger
    
    parser = argparse.ArgumentParser(description='Trade-Based Training System')
    parser.add_argument('--multi-objective', '-mo', action='store_true',
                       help='Enable multi-objective rewards (5 separate signals)')
    parser.add_argument('--rl-only', action='store_true',
                       help='Train RL only (use existing ML model)')
    args = parser.parse_args()
    
    logger = setup_logger('trade_based_training', level=logging.INFO)
    
    print("="*80)
    print("TRADE-BASED TRAINING SYSTEM")
    print("Each episode = 1 complete trade")
    print("Agent learns: WHEN to enter, WHEN to exit")
    if args.multi_objective:
        print("Mode: MULTI-OBJECTIVE (5 separate reward signals)")
    else:
        print("Mode: STANDARD (single reward signal)")
    print("="*80)
    
    # Run training
    if args.rl_only:
        results = train_trade_based_rl_only(multi_objective=args.multi_objective)
    else:
        results = train_trade_based(multi_objective=args.multi_objective)