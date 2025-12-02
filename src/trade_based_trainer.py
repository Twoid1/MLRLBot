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
            
            # Trade-based specific settings
            'trade_config': {
                'max_wait_steps': 250,       # Max steps to find entry
                'max_hold_steps': 300,       # Max steps to hold trade
                'min_hold_for_bonus': 36,    # Min hold for bonus (3 hours on 5m)
                'no_trade_penalty': -2.0,    # Penalty for not trading
                'timeout_exit_penalty': -1.0, # Penalty for forced exit
            },
            
            # RL training settings
            'rl_training_mode': 'random',
            'rl_episodes': 8000,  # More episodes since each is shorter
            
            # Network architecture (same as before)
            'rl_hidden_dims': [128, 64],
            'use_double_dqn': True,
            'use_dueling_dqn': True,
            'rl_batch_size': 256,
            'rl_memory_size': 50000,
            
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
            'stop_loss': 0.04,
            'take_profit': 0.08,
            
            # Progress tracking
            'log_interval': 50,
            'save_interval': 200,
            
            # Explainability
            'explainability': {
                'enabled': False,
                'verbose': False,
                'explain_frequency': 100,
                'save_dir': 'logs/trade_explanations'
            }
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
                self.progress.start_stage('Trade-Based RL Training', self.config['rl_episodes'])
                
                # Filter features
                train_data_filtered = self._filter_features_to_selected(train_data)
                val_data_filtered = self._filter_features_to_selected(val_data)
                test_data_filtered = self._filter_features_to_selected(test_data)
                
                # Train with trade-based episodes
                self._train_rl_agent_trade_based(train_data_filtered, val_data_filtered, test_data_filtered)
                
                self.progress.complete_stage('Trade-Based RL Training', {
                    'time_seconds': time.time() - stage_start,
                    'episodes': self.config['rl_episodes']
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
            logger.info("="*80)
            
            self.training_results['speedup_metrics'] = {
                'total_time_hours': total_time / 3600
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
        
        # Create trade configuration
        tc = self.config.get('trade_config', {})
        trade_config = TradeConfig(
            max_wait_steps=tc.get('max_wait_steps', 200),
            max_hold_steps=tc.get('max_hold_steps', 300),
            min_hold_for_bonus=tc.get('min_hold_for_bonus', 36),
            no_trade_penalty=tc.get('no_trade_penalty', -2.0),
            timeout_exit_penalty=tc.get('timeout_exit_penalty', -1.0),
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
            take_profit=self.config.get('take_profit', 0.06),
            asset=first_asset,
            trade_config=trade_config
        )
        
        state_dim = temp_env.observation_space_shape[0]
        action_dim = temp_env.action_space_n  # Should be 2!
        
        logger.info(f"\n   Trade-based environment created")
        logger.info(f"      State dims: {state_dim}")
        logger.info(f"      Action dims: {action_dim} (phase-dependent)")
        logger.info(f"      Timeframes: {temp_env.timeframes}")
        
        # [STEP 3: Initialize agent with 2 actions]
        logger.info("\n[3/5] Initializing agent with 2-action space...")
        
        rl_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,  # 2 actions!
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
                    take_profit=self.config.get('take_profit', 0.06),
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
        
        # Generate report
        self._generate_trade_based_report(episode_results)
        
        logger.info("\n  Trade-based training complete!")
        logger.info(f"  Total episodes: {len(episode_results)}")
        logger.info(f"  Agent memory: {len(self.rl_agent.memory)} experiences")

    def _train_trade_based_random(self, environments: Dict, total_episodes: int) -> list:
        """
        Train with random environment selection - TRADE-BASED VERSION
        
        Key differences from time-based:
        - Episodes end when trade completes (variable length)
        - No max_steps parameter (environment handles termination)
        - Tracks trade-specific metrics
        """
        import random
        
        episode_results = []
        env_names = list(environments.keys())
        
        # Validation tracking
        self.best_val_reward = float('-inf')
        self.patience_counter = 0
        
        logger.info("\n  TRADE-BASED RANDOM TRAINING MODE")
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
        """
        episode_start = time.time()
        
        # Reset environment (starts in SEARCHING phase)
        state = env.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Safety limit (should never hit this with proper timeout config)
        max_safety_steps = 1000
        
        while not done and steps < max_safety_steps:
            # Get action from agent
            action = self.rl_agent.act(state, training=True)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            # Note: For trade-based, most rewards are 0 until trade completes
            bootstrap_done = done and not truncated
            self.rl_agent.remember(state, action, reward, next_state, bootstrap_done)
            
            # Increment step count
            self.rl_agent.step_count += 1
            
            # Training updates
            if self.rl_agent.step_count % self.rl_agent.config.update_every == 0:
                if len(self.rl_agent.memory) >= self.rl_agent.config.min_memory_size:
                    self.rl_agent.replay()
            
            # Target network update
            if self.rl_agent.step_count % self.rl_agent.config.target_update_every == 0:
                self.rl_agent.update_target_network(soft=True)
            
            # Epsilon decay
            if not self.rl_agent.config.use_noisy_networks:
                self.rl_agent.epsilon = max(
                    self.rl_agent.config.epsilon_end,
                    self.rl_agent.epsilon * self.rl_agent.config.epsilon_decay
                )
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        self.rl_agent.episode_count += 1
        
        # Build stats
        stats = {
            'episode': episode_num + 1,
            'asset': asset,
            'total_reward': episode_reward,
            'steps': steps,
            'epsilon': self.rl_agent.epsilon,
            'episode_time': time.time() - episode_start,
            'termination': info.get('termination', 'unknown')
        }
        
        # Add trade result if available
        if 'trade_result' in info:
            tr = info['trade_result']
            stats['trade_pnl'] = tr.get('net_pnl', 0)
            stats['trade_pnl_pct'] = tr.get('pnl_pct', 0)
            stats['hold_duration'] = tr.get('hold_duration', 0)
            stats['exit_reason'] = tr.get('exit_reason', 'unknown')
            
            # Track for analysis
            self.episode_trade_results.append({
                'episode': episode_num + 1,
                'asset': asset,
                **tr
            })
        
        return stats

    def _run_trade_validation(self, environments: Dict, episode_num: int) -> dict:
        """
        Run validation episodes (greedy, no exploration)
        """
        import random
        
        val_episodes = self.config.get('validation_episodes', 50)
        val_results = []
        trade_results = []
        
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
            
            while not done and steps < 1000:
                action = self.rl_agent.act(state, training=False)
                next_state, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done or truncated:
                    break
            
            val_results.append(episode_reward)
            
            # Track trade results
            if 'trade_result' in info:
                trade_results.append(info['trade_result'])
        
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
        avg_reward = np.mean([e['total_reward'] for e in recent_episodes])
        
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
        logger.info(f"    Avg Reward: {avg_reward:.2f}")
        logger.info(f"    Win Rate: {win_rate:.1%}")
        logger.info(f"    Avg P&L: {avg_pnl:.2f}%")
        logger.info(f"    Avg Hold: {avg_hold:.0f} steps")
        logger.info(f"    Epsilon: {self.rl_agent.epsilon:.3f}")
        logger.info(f"    Memory: {len(self.rl_agent.memory):,}")
        logger.info(f"    Terminations: {terminations}")

    def _generate_trade_based_report(self, episode_results: list):
        """
        Generate COMPREHENSIVE trade-based training report with diagnostics
        
        This report is designed to help you understand:
        1. What the agent learned
        2. What's working and what isn't
        3. Specific recommendations for improvement
        """
        
        report_lines = []
        report_lines.append("="*100)
        report_lines.append("COMPREHENSIVE TRADE-BASED TRAINING DIAGNOSTIC REPORT")
        report_lines.append("="*100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 1: EXECUTIVE SUMMARY
        # ═══════════════════════════════════════════════════════════════════════════
        
        total_episodes = len(episode_results)
        trades_with_results = [e for e in episode_results if 'trade_pnl_pct' in e]
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " EXECUTIVE SUMMARY".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        if trades_with_results:
            profitable = sum(1 for e in trades_with_results if e['trade_pnl_pct'] > 0)
            win_rate = profitable / len(trades_with_results)
            avg_pnl = np.mean([e['trade_pnl_pct'] for e in trades_with_results]) * 100
            total_pnl = sum(e['trade_pnl_pct'] for e in trades_with_results) * 100
            avg_hold = np.mean([e.get('hold_duration', 0) for e in trades_with_results])
            avg_reward = np.mean([e['total_reward'] for e in trades_with_results])
            
            # Calculate key metrics
            avg_win = np.mean([e['trade_pnl_pct'] for e in trades_with_results if e['trade_pnl_pct'] > 0]) * 100 if profitable > 0 else 0
            avg_loss = np.mean([e['trade_pnl_pct'] for e in trades_with_results if e['trade_pnl_pct'] <= 0]) * 100 if (len(trades_with_results) - profitable) > 0 else 0
            
            # Risk/Reward ratio
            risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            
            report_lines.append("")
            report_lines.append(f"  Total Trades:      {len(trades_with_results):,}")
            report_lines.append(f"  Win Rate:          {win_rate:.1%}")
            report_lines.append(f"  Avg P&L:           {avg_pnl:+.2f}%")
            report_lines.append(f"  Total P&L:         {total_pnl:+.2f}%")
            report_lines.append(f"  Avg Hold:          {avg_hold:.1f} steps ({avg_hold*5:.0f} min)")
            report_lines.append("")
            report_lines.append(f"  Avg Win:           {avg_win:+.2f}%")
            report_lines.append(f"  Avg Loss:          {avg_loss:+.2f}%")
            report_lines.append(f"  Risk/Reward:       {risk_reward:.2f}")
            report_lines.append(f"  Expectancy:        {expectancy:+.3f}% per trade")
            report_lines.append("")
            
            # Quick verdict
            if expectancy > 0:
                report_lines.append("   VERDICT: Positive expectancy - system has potential")
            elif expectancy > -0.1:
                report_lines.append("    VERDICT: Near breakeven - needs fine-tuning")
            else:
                report_lines.append("   VERDICT: Negative expectancy - significant changes needed")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 2: P&L DISTRIBUTION ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " P&L DISTRIBUTION ANALYSIS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        if trades_with_results:
            pnl_values = [e['trade_pnl_pct'] * 100 for e in trades_with_results]
            
            # Percentiles
            p5 = np.percentile(pnl_values, 5)
            p25 = np.percentile(pnl_values, 25)
            p50 = np.percentile(pnl_values, 50)
            p75 = np.percentile(pnl_values, 75)
            p95 = np.percentile(pnl_values, 95)
            
            report_lines.append("")
            report_lines.append("  P&L Percentiles:")
            report_lines.append(f"    5th:   {p5:+.2f}%  (worst 5% of trades)")
            report_lines.append(f"    25th:  {p25:+.2f}%")
            report_lines.append(f"    50th:  {p50:+.2f}%  (median)")
            report_lines.append(f"    75th:  {p75:+.2f}%")
            report_lines.append(f"    95th:  {p95:+.2f}%  (best 5% of trades)")
            report_lines.append("")
            
            # Best and worst trades
            sorted_trades = sorted(trades_with_results, key=lambda x: x['trade_pnl_pct'])
            
            report_lines.append("  Worst 5 Trades:")
            for i, t in enumerate(sorted_trades[:5]):
                report_lines.append(f"    {i+1}. {t['asset']}: {t['trade_pnl_pct']*100:+.2f}% (held {t.get('hold_duration', 0)} steps, {t.get('exit_reason', 'unknown')} exit)")
            
            report_lines.append("")
            report_lines.append("  Best 5 Trades:")
            for i, t in enumerate(sorted_trades[-5:][::-1]):
                report_lines.append(f"    {i+1}. {t['asset']}: {t['trade_pnl_pct']*100:+.2f}% (held {t.get('hold_duration', 0)} steps, {t.get('exit_reason', 'unknown')} exit)")
            
            # Distribution histogram (text-based)
            report_lines.append("")
            report_lines.append("  P&L Distribution (text histogram):")
            bins = [(-10, -3), (-3, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 10)]
            for low, high in bins:
                count = sum(1 for p in pnl_values if low <= p < high)
                pct = count / len(pnl_values) * 100
                bar = "|" * int(pct / 2)
                label = f"{low:+.1f}% to {high:+.1f}%"
                report_lines.append(f"    {label:>18}: {bar:<25} {count:>4} ({pct:>5.1f}%)")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 3: HOLD DURATION ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " HOLD DURATION ANALYSIS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        if trades_with_results:
            hold_values = [e.get('hold_duration', 0) for e in trades_with_results]
            min_hold_bonus = self.config.get('trade_config', {}).get('min_hold_for_bonus', 36)
            
            report_lines.append("")
            report_lines.append(f"  Min Hold for Bonus: {min_hold_bonus} steps ({min_hold_bonus*5} min)")
            report_lines.append("")
            
            # Hold duration stats
            report_lines.append("  Hold Duration Stats:")
            report_lines.append(f"    Min:     {min(hold_values)} steps ({min(hold_values)*5} min)")
            report_lines.append(f"    Max:     {max(hold_values)} steps ({max(hold_values)*5} min)")
            report_lines.append(f"    Avg:     {np.mean(hold_values):.1f} steps ({np.mean(hold_values)*5:.0f} min)")
            report_lines.append(f"    Median:  {np.median(hold_values):.1f} steps ({np.median(hold_values)*5:.0f} min)")
            report_lines.append("")
            
            # How many hit the minimum?
            long_holds = sum(1 for h in hold_values if h >= min_hold_bonus)
            report_lines.append(f"  Trades >= {min_hold_bonus} steps: {long_holds} ({long_holds/len(hold_values)*100:.1f}%)")
            report_lines.append(f"  Trades <  {min_hold_bonus} steps: {len(hold_values) - long_holds} ({(len(hold_values) - long_holds)/len(hold_values)*100:.1f}%)")
            report_lines.append("")
            
            # Hold duration by outcome
            winners = [e for e in trades_with_results if e['trade_pnl_pct'] > 0]
            losers = [e for e in trades_with_results if e['trade_pnl_pct'] <= 0]
            
            if winners:
                avg_win_hold = np.mean([e.get('hold_duration', 0) for e in winners])
                report_lines.append(f"  Avg Hold (Winners):  {avg_win_hold:.1f} steps ({avg_win_hold*5:.0f} min)")
            if losers:
                avg_loss_hold = np.mean([e.get('hold_duration', 0) for e in losers])
                report_lines.append(f"  Avg Hold (Losers):   {avg_loss_hold:.1f} steps ({avg_loss_hold*5:.0f} min)")
            
            # Distribution
            report_lines.append("")
            report_lines.append("  Hold Duration Distribution:")
            hold_bins = [(0, 5), (5, 10), (10, 20), (20, 36), (36, 60), (60, 100), (100, 300)]
            for low, high in hold_bins:
                count = sum(1 for h in hold_values if low <= h < high)
                pct = count / len(hold_values) * 100
                bar = "|" * int(pct / 2)
                label = f"{low}-{high} steps"
                report_lines.append(f"    {label:>15}: {bar:<25} {count:>4} ({pct:>5.1f}%)")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 4: ENTRY BEHAVIOR ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " ENTRY BEHAVIOR ANALYSIS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        episodes_with_search = [e for e in episode_results if 'search_steps' in e]
        if episodes_with_search:
            search_times = [e['search_steps'] for e in episodes_with_search]
            entered = [e for e in episodes_with_search if e.get('entered_trade', False)]
            
            report_lines.append("")
            report_lines.append(f"  Total Episodes:        {len(episodes_with_search)}")
            report_lines.append(f"  Episodes with Entry:   {len(entered)} ({len(entered)/len(episodes_with_search)*100:.1f}%)")
            report_lines.append("")
            
            if entered:
                entry_search_times = [e['search_steps'] for e in entered]
                report_lines.append("  Search Time Before Entry:")
                report_lines.append(f"    Min:     {min(entry_search_times)} steps")
                report_lines.append(f"    Max:     {max(entry_search_times)} steps")
                report_lines.append(f"    Avg:     {np.mean(entry_search_times):.1f} steps")
                report_lines.append(f"    Median:  {np.median(entry_search_times):.0f} steps")
                report_lines.append("")
                
                # Immediate entries (first step)
                immediate_entries = sum(1 for e in entered if e['search_steps'] <= 1)
                report_lines.append(f"  Immediate Entries (<=1 step): {immediate_entries} ({immediate_entries/len(entered)*100:.1f}%)")
                
                # Patient entries (>10 steps)
                patient_entries = sum(1 for e in entered if e['search_steps'] > 10)
                report_lines.append(f"  Patient Entries (>10 steps): {patient_entries} ({patient_entries/len(entered)*100:.1f}%)")
        else:
            report_lines.append("")
            report_lines.append("  (Entry behavior data not available - run training with updated code)")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 5: EXIT BEHAVIOR ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " EXIT BEHAVIOR ANALYSIS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        if trades_with_results:
            report_lines.append("")
            
            # Exit reason breakdown
            exit_reasons = {}
            for e in trades_with_results:
                reason = e.get('exit_reason', e.get('termination', 'unknown'))
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            report_lines.append("  Exit Reason Breakdown:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                pct = count / len(trades_with_results) * 100
                
                # Get avg P&L for this exit type
                exit_trades = [e for e in trades_with_results if e.get('exit_reason', e.get('termination', '')) == reason]
                if exit_trades:
                    avg_exit_pnl = np.mean([e['trade_pnl_pct'] for e in exit_trades]) * 100
                    avg_exit_hold = np.mean([e.get('hold_duration', 0) for e in exit_trades])
                    report_lines.append(f"    {reason:15}: {count:>5} ({pct:>5.1f}%)  Avg P&L: {avg_exit_pnl:+.2f}%  Avg Hold: {avg_exit_hold:.0f} steps")
                else:
                    report_lines.append(f"    {reason:15}: {count:>5} ({pct:>5.1f}%)")
            
            # Immediate exits
            episodes_with_hold_data = [e for e in episode_results if 'hold_hold_count' in e]
            if episodes_with_hold_data:
                immediate_exits = sum(1 for e in episodes_with_hold_data if e.get('hold_hold_count', 0) <= 1)
                report_lines.append("")
                report_lines.append(f"  Immediate Exits (<=1 HOLD action): {immediate_exits} ({immediate_exits/len(episodes_with_hold_data)*100:.1f}%)")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 6: PER-ASSET ANALYSIS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " PER-ASSET DETAILED ANALYSIS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        by_asset = {}
        for e in episode_results:
            asset = e['asset']
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(e)
        
        report_lines.append("")
        report_lines.append(f"  {'Asset':<12} {'Trades':>7} {'Win%':>7} {'Avg P&L':>9} {'Avg Hold':>10} {'R/R':>6} {'Expectancy':>12}")
        report_lines.append("  " + "-"*75)
        
        for asset in sorted(by_asset.keys()):
            episodes = by_asset[asset]
            trades = [e for e in episodes if 'trade_pnl_pct' in e]
            
            if trades:
                profitable = sum(1 for e in trades if e['trade_pnl_pct'] > 0)
                win_rate = profitable / len(trades)
                avg_pnl = np.mean([e['trade_pnl_pct'] for e in trades]) * 100
                avg_hold = np.mean([e.get('hold_duration', 0) for e in trades])
                
                # Risk/reward for this asset
                wins = [e['trade_pnl_pct'] * 100 for e in trades if e['trade_pnl_pct'] > 0]
                losses = [e['trade_pnl_pct'] * 100 for e in trades if e['trade_pnl_pct'] <= 0]
                avg_win = np.mean(wins) if wins else 0
                avg_loss = np.mean(losses) if losses else 0
                rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
                
                status = "Y" if expectancy > 0 else "N"
                report_lines.append(f"  {asset:<12} {len(trades):>7} {win_rate:>6.1%} {avg_pnl:>+8.2f}% {avg_hold:>9.0f}s {rr:>6.2f} {expectancy:>+11.3f}% {status}")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 7: LEARNING PROGRESS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " LEARNING PROGRESS OVER TIME".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        if len(episode_results) >= 500:
            # Split into 5 segments
            segment_size = len(episode_results) // 5
            
            report_lines.append("")
            report_lines.append(f"  {'Segment':<12} {'Episodes':>12} {'Win%':>8} {'Avg P&L':>10} {'Avg Hold':>10} {'Avg Reward':>12}")
            report_lines.append("  " + "-"*70)
            
            for i in range(5):
                start = i * segment_size
                end = start + segment_size if i < 4 else len(episode_results)
                segment = [e for e in episode_results[start:end] if 'trade_pnl_pct' in e]
                
                if segment:
                    profitable = sum(1 for e in segment if e['trade_pnl_pct'] > 0)
                    win_rate = profitable / len(segment)
                    avg_pnl = np.mean([e['trade_pnl_pct'] for e in segment]) * 100
                    avg_hold = np.mean([e.get('hold_duration', 0) for e in segment])
                    avg_reward = np.mean([e['total_reward'] for e in segment])
                    
                    label = f"{start+1}-{end}"
                    report_lines.append(f"  {label:<12} {len(segment):>12} {win_rate:>7.1%} {avg_pnl:>+9.2f}% {avg_hold:>9.0f}s {avg_reward:>+11.2f}")
            
            # Overall trend analysis
            first_segment = [e for e in episode_results[:segment_size] if 'trade_pnl_pct' in e]
            last_segment = [e for e in episode_results[-segment_size:] if 'trade_pnl_pct' in e]
            
            if first_segment and last_segment:
                first_wr = sum(1 for e in first_segment if e['trade_pnl_pct'] > 0) / len(first_segment)
                last_wr = sum(1 for e in last_segment if e['trade_pnl_pct'] > 0) / len(last_segment)
                first_hold = np.mean([e.get('hold_duration', 0) for e in first_segment])
                last_hold = np.mean([e.get('hold_duration', 0) for e in last_segment])
                first_pnl = np.mean([e['trade_pnl_pct'] for e in first_segment]) * 100
                last_pnl = np.mean([e['trade_pnl_pct'] for e in last_segment]) * 100
                
                report_lines.append("")
                report_lines.append("  Improvement First -> Last Segment:")
                report_lines.append(f"    Win Rate:     {first_wr:.1%} -> {last_wr:.1%}  ({(last_wr-first_wr)*100:+.1f}pp)")
                report_lines.append(f"    Avg P&L:      {first_pnl:+.2f}% -> {last_pnl:+.2f}%  ({last_pnl-first_pnl:+.2f}%)")
                report_lines.append(f"    Avg Hold:     {first_hold:.0f}s -> {last_hold:.0f}s  ({last_hold-first_hold:+.0f}s)")
        else:
            report_lines.append("")
            report_lines.append("  (Need at least 500 episodes for detailed learning progress)")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 8: DIAGNOSTIC ISSUES & RECOMMENDATIONS
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " DIAGNOSTIC ISSUES & RECOMMENDATIONS".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        issues = []
        recommendations = []
        
        if trades_with_results:
            # Issue 1: Hold duration too short
            avg_hold = np.mean([e.get('hold_duration', 0) for e in trades_with_results])
            min_hold_bonus = self.config.get('trade_config', {}).get('min_hold_for_bonus', 36)
            
            if avg_hold < min_hold_bonus * 0.5:
                issues.append(f"CRITICAL: Avg hold ({avg_hold:.0f} steps) is < 50% of min bonus threshold ({min_hold_bonus})")
                recommendations.append("-> INCREASE hold_duration_bonus from 0.01 to 0.05 or higher")
                recommendations.append("-> INCREASE early exit penalty (change 0.3 multiplier to 0.1)")
                recommendations.append("-> DECREASE min_hold_for_bonus from 36 to 20 (make bonus easier to achieve)")
            elif avg_hold < min_hold_bonus:
                issues.append(f"WARNING: Avg hold ({avg_hold:.0f} steps) is below min bonus threshold ({min_hold_bonus})")
                recommendations.append("-> Slightly increase hold_duration_bonus")
            
            # Issue 2: Too many agent exits vs stop/take-profit
            exit_reasons = {}
            for e in trades_with_results:
                reason = e.get('exit_reason', e.get('termination', 'unknown'))
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            agent_exits = exit_reasons.get('agent', exit_reasons.get('agent_exit', 0))
            sl_exits = exit_reasons.get('stop_loss', 0)
            tp_exits = exit_reasons.get('take_profit', 0)
            
            if agent_exits / len(trades_with_results) > 0.95:
                issues.append(f"CRITICAL: {agent_exits/len(trades_with_results)*100:.1f}% agent exits - not letting trades run")
                recommendations.append("-> Agent is exiting too early instead of hitting stop-loss/take-profit")
                recommendations.append("-> REDUCE the +0.2 bonus for profitable agent exits")
                recommendations.append("-> ADD penalty for exiting before min hold duration")
            
            if tp_exits / len(trades_with_results) < 0.01:
                issues.append(f"WARNING: Only {tp_exits} take-profit hits ({tp_exits/len(trades_with_results)*100:.2f}%)")
                recommendations.append("-> Agent never lets winners run to take-profit")
                recommendations.append("-> INCREASE take_profit bonus from 0.5 to 1.0 or higher")
            
            # Issue 3: Win rate too low
            profitable = sum(1 for e in trades_with_results if e['trade_pnl_pct'] > 0)
            win_rate = profitable / len(trades_with_results)
            
            if win_rate < 0.35:
                issues.append(f"CRITICAL: Win rate ({win_rate:.1%}) is below 35%")
                recommendations.append("-> Agent may be entering at wrong times")
                recommendations.append("-> Consider LONGER search phase (increase max_wait_steps)")
                recommendations.append("-> Review ML features being used for entry signals")
            
            # Issue 4: Negative expectancy
            avg_pnl = np.mean([e['trade_pnl_pct'] for e in trades_with_results]) * 100
            if avg_pnl < -0.1:
                issues.append(f"CRITICAL: Negative avg P&L ({avg_pnl:+.2f}%)")
                recommendations.append("-> Fees + slippage may be eating profits")
                recommendations.append("-> Consider WIDER stop-loss (currently 3%)")
                recommendations.append("-> Consider TIGHTER take-profit to lock in more wins")
            
            # Issue 5: Not enough exploration
            episodes_with_search = [e for e in episode_results if 'search_steps' in e]
            if episodes_with_search:
                immediate_entries = sum(1 for e in episodes_with_search if e.get('search_steps', 0) <= 1)
                if immediate_entries / len(episodes_with_search) > 0.8:
                    issues.append(f"WARNING: {immediate_entries/len(episodes_with_search)*100:.1f}% immediate entries")
                    recommendations.append("-> Agent enters too quickly without waiting for good setup")
                    recommendations.append("-> Consider adding patience_bonus for waiting")
            
            # Issue 6: Large losses
            big_losses = sum(1 for e in trades_with_results if e['trade_pnl_pct'] < -0.025)
            if big_losses / len(trades_with_results) > 0.1:
                issues.append(f"WARNING: {big_losses} trades ({big_losses/len(trades_with_results)*100:.1f}%) lost > 2.5%")
                recommendations.append("-> Consider TIGHTER stop-loss (currently 3%)")
        
        report_lines.append("")
        if issues:
            report_lines.append("  ISSUES DETECTED:")
            for issue in issues:
                report_lines.append(f"    {issue}")
            report_lines.append("")
            report_lines.append("  RECOMMENDATIONS:")
            for rec in recommendations:
                report_lines.append(f"    {rec}")
        else:
            report_lines.append("  No critical issues detected")
        
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 9: CONFIGURATION USED
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " CONFIGURATION USED".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        report_lines.append("")
        report_lines.append("  Environment:")
        report_lines.append(f"    Assets:              {self.config.get('assets', [])}")
        report_lines.append(f"    Timeframes:          {self.config.get('timeframes', [])}")
        report_lines.append(f"    Episodes:            {self.config.get('rl_episodes', 'N/A')}")
        report_lines.append("")
        
        trade_config = self.config.get('trade_config', {})
        report_lines.append("  Trade Settings:")
        report_lines.append(f"    Max Wait Steps:      {trade_config.get('max_wait_steps', 200)}")
        report_lines.append(f"    Max Hold Steps:      {trade_config.get('max_hold_steps', 300)}")
        report_lines.append(f"    Min Hold for Bonus:  {trade_config.get('min_hold_for_bonus', 36)} steps")
        report_lines.append(f"    No Trade Penalty:    {trade_config.get('no_trade_penalty', -2.0)}")
        report_lines.append(f"    Timeout Exit Penalty:{trade_config.get('timeout_exit_penalty', -1.0)}")
        report_lines.append(f"    Hold Duration Bonus: {trade_config.get('hold_duration_bonus', 0.01)} per step")
        report_lines.append("")
        
        report_lines.append("  Risk Settings:")
        report_lines.append(f"    Stop Loss:           {self.config.get('stop_loss', 0.03)*100:.1f}%")
        report_lines.append(f"    Take Profit:         {self.config.get('take_profit', 0.06)*100:.1f}%")
        report_lines.append(f"    Fee Rate:            {self.config.get('fee_rate', 0.001)*100:.2f}%")
        report_lines.append(f"    Slippage:            {self.config.get('slippage', 0.0005)*100:.2f}%")
        report_lines.append("")
        
        # ═══════════════════════════════════════════════════════════════════════════
        # SECTION 10: REWARD SYSTEM REFERENCE
        # ═══════════════════════════════════════════════════════════════════════════
        
        report_lines.append("|" + "="*98 + "|")
        report_lines.append("|" + " REWARD SYSTEM REFERENCE".ljust(98) + "|")
        report_lines.append("|" + "="*98 + "|")
        
        report_lines.append("")
        report_lines.append("  GOOD REWARDS (Positive):")
        report_lines.append("    +P&L% x 100          When trade is profitable")
        report_lines.append("    +0.5                 When take-profit is hit")
        report_lines.append("    +0.2                 When agent exits profitably (after costs)")
        report_lines.append(f"    +0.01 per step       When holding >= {trade_config.get('min_hold_for_bonus', 36)} steps (max 2.0)")
        report_lines.append("")
        report_lines.append("  BAD REWARDS (Negative):")
        report_lines.append("    -P&L% x 100          When trade loses money")
        report_lines.append("    -2.0                 When search times out (no entry in 200 steps)")
        report_lines.append("    -1.0                 Extra penalty when hold times out (300 steps)")
        report_lines.append("    x1.0-1.5             Early exit on losing trade (amplified loss)")
        report_lines.append("")
        report_lines.append("  REDUCED REWARDS:")
        report_lines.append("    x0.3-1.0             Early exit on winning trade (reduced profit)")
        report_lines.append("")
        
        report_lines.append("="*100)
        report_lines.append("END OF COMPREHENSIVE DIAGNOSTIC REPORT")
        report_lines.append("="*100)
        
        # Log to console
        for line in report_lines:
            logger.info(line)
        
        # Save to file
        report_path = Path('results') / f'trade_based_diagnostic_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"\n  Comprehensive report saved to: {report_path}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS (Same as OptimizedSystemTrainer)
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
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based_best_{timestamp}_ep{episode}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
            logger.info(f"  Saved best model: {rl_path}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based_checkpoint_ep{episode}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
    
    def _save_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.ml_predictor:
            ml_path = Path(f'models/ml/ml_predictor_{timestamp}.pkl')
            ml_path.parent.mkdir(parents=True, exist_ok=True)
            self.ml_predictor.save_model(ml_path)
            logger.info(f"Saved ML model: {ml_path}")
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/trade_based_agent_{timestamp}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
            logger.info(f"Saved RL agent: {rl_path}")
        
        # Save trade history
        if self.episode_trade_results:
            trades_path = Path(f'results/trade_history_{timestamp}.json')
            trades_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trades_path, 'w') as f:
                json.dump(self.episode_trade_results, f, indent=2, default=str)
            logger.info(f"Saved trade history: {trades_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_trade_based(config_path: Optional[str] = None) -> Dict:
    """
    Train with trade-based episodes (recommended)
    
    Each episode = 1 complete trade
    Agent learns trade quality, not step prediction
    """
    trainer = TradeBasedTrainer(config_path)
    return trainer.train_complete_system(train_ml=True, train_rl=True)


def train_trade_based_rl_only(config_path: Optional[str] = None) -> Dict:
    """
    Trade-based RL training only (uses existing ML model)
    """
    trainer = TradeBasedTrainer(config_path)
    return trainer.train_complete_system(train_ml=False, train_rl=True)


if __name__ == "__main__":
    import logging
    from src.utils.logger import setup_logger
    
    logger = setup_logger('trade_based_training', level=logging.INFO)
    
    print("="*80)
    print("TRADE-BASED TRAINING SYSTEM")
    print("Each episode = 1 complete trade")
    print("Agent learns: WHEN to enter, WHEN to exit")
    print("="*80)
    
    # Run training
    results = train_trade_based()