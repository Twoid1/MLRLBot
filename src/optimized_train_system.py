"""
OPTIMIZED Training System - 10-20x Speed Improvement + EXPLAINABILITY
- GPU acceleration for PyTorch and XGBoost
- Parallel processing for multi-asset operations
- Vectorized feature calculations
- Progress tracking with live updates
- Memory optimization
- â­ NEW: Explainability system to understand agent decisions
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
torch.set_num_threads(8)  # Optimize CPU threading

# Fast operations
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# â­ NEW: Explainability import
from src.explainability_integration import ExplainableRL

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Real-time progress tracking for training"""
    
    def __init__(self, log_file: str = 'training_progress.json'):
        self.log_file = Path(log_file)
        # Create directory if it doesn't exist
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


class OptimizedSystemTrainer:
    """
    Ultra-fast training system with parallel processing and GPU acceleration
    Target: 4-6 hour total training time (down from 2 days)
    â­ NEW: With explainability to understand agent decisions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._validate_config()
        self.progress = ProgressTracker('logs/training_progress.json')
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Import components (lazy loading)
        from src.features.feature_engineer import FeatureEngineer
        from src.features.selector import FeatureSelector
        from src.models.ml_predictor import MLPredictor
        from src.models.dqn_agent import DQNAgent, DQNConfig
        from src.environment.trading_env import TradingEnvironment
        # In optimized_train_system.py:
        from src.environment.multi_timeframe_env import MultiTimeframeEnvironment
        
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()

        self.ml_predictor = None  # Initialize as None
        self.rl_agent = None 
        
        # â­ NEW: Explainability
        self.explainer = None
        
        # Training results
        self.training_results = {
            'ml_results': None,
            'rl_results': None,
            'start_time': None,
            'end_time': None,
            'speedup_metrics': {}
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load optimized configuration"""
        config = {
            # Assets and timeframes
            'assets': ['SOL_USDT', 'AVAX_USDT', 'ADA_USDT', 'ETH_USDT', 'DOT_USDT'],
            'timeframes': ['5m', '15m', '1h'],  # ✓ Day trading timeframes
            'execution_timeframe': '5m',  # ✓ Execute trades on 5m
            
            # Data settings
            'start_date': '2021-01-01',
            'end_date': '2025-10-01',
            'train_split': 0.8,
            'validation_split': 0.1,
            
            # Optimization settings
            'use_gpu': True,
            'num_workers': 8,  # CPU cores for parallel processing
            'batch_size': 256,  # Larger batch size for GPU
            'cache_features': True,  # Cache computed features
            'preload_data': True,  # Load all data into memory
            
            # ML settings
            'ml_model_type': 'xgboost',
            'n_features': 50,
            'feature_selection': True,
            'ml_gpu_params': {
                'tree_method': 'gpu_hist',  # GPU acceleration
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            },

            # Labeling settings
            'labeling_method': 'triple_barrier',
            'lookforward': 10,
            'pt_sl': [1.5, 1.0],  # Profit-take / Stop-loss multipliers
            
            # Feature engineering
            'n_features': 50,
            'feature_selection': True,
            'optimize_ml_params': False,
            'use_sample_weights': False,
            'walk_forward_splits': 5,
            
            # RL settings
            'rl_training_mode': 'random',
            'max_steps_per_episode': {
                '5m': 700,   # ✅ NEW: More steps since 5m is faster
                '15m': 500,  # ✅ NEW: Medium timeframe
                '1h': 400    # ✅ KEEP: Matches current '1h' 
            },
            # âš¡ OVERFITTING FIX: Variable episode lengths prevent temporal memorization
            'variable_episode_length': True,
            'episode_length_range': {
                '5m': [400, 700],   # ✅ NEW: Variable length for 5m
                '15m': [250, 500],  # ✅ NEW: Variable length for 15m
                '1h': [150, 400]    # ✅ KEEP: Current range for 1h
            },
            'rl_episodes': 3000,
            # âš¡ OVERFITTING FIX: Smaller network (was [256, 256, 128] = 99k params)
            'rl_hidden_dims': [128, 64],  # ~15k parameters - prevents memorization
            'use_double_dqn': True,
            'use_dueling_dqn': True,
            # âš¡ OVERFITTING FIX: Smaller batches for better generalization  
            'rl_batch_size': 512,  # Was 1024 - more frequent, stochastic updates
            'rl_update_frequency': 4,  # Update every N steps (faster)
            # âš¡ OVERFITTING FIX: Smaller memory (was 100k)
            'rl_memory_size': 20000,  # Focus on recent, relevant experiences
            
            # Progress tracking
            'log_interval': 10,  # Log every N episodes
            'save_interval': 50,  # Save checkpoint every N episodes
            
            # âš¡ OVERFITTING FIX: Validation monitoring and early stopping
            'validation_frequency': 50,  # Validate every N episodes
            'early_stopping_patience': 99999,  # Stop if no improvement for N validations
            'early_stopping_min_delta': 0.01,  # Minimum improvement threshold
            'validation_episodes': 20,  # Number of episodes to run for validation
            
            # Environment
            'initial_balance': 10000,
            'fee_rate': 0.001,
            'slippage': 0.001,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            
            # â­ NEW: Explainability settings
            'explainability': {
                'enabled': False,
                'verbose': False,
                'explain_frequency': 100,
                'save_dir': 'logs/explanations'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
        
        return config
    
    def _validate_config(self):
        """
        Validate configuration consistency
        
        Ensures all timeframes have proper episode length configurations
        to prevent runtime errors during training.
        """
        logger.info("\n" + "="*60)
        logger.info("VALIDATING CONFIGURATION")
        logger.info("="*60)
        
        # Check 1: Timeframes have episode configurations
        logger.info("\n[1/3] Checking timeframe configurations...")
        for tf in self.config['timeframes']:
            # Check max_steps_per_episode
            if tf not in self.config['max_steps_per_episode']:
                raise ValueError(
                    f" Timeframe '{tf}' in config['timeframes'] but NOT in "
                    f"config['max_steps_per_episode'].\n"
                    f"Available keys: {list(self.config['max_steps_per_episode'].keys())}\n"
                    f"Please add episode length configuration for '{tf}'."
                )
            
            # Check episode_length_range if variable lengths enabled
            if self.config.get('variable_episode_length'):
                if tf not in self.config['episode_length_range']:
                    raise ValueError(
                        f" Timeframe '{tf}' missing from config['episode_length_range'].\n"
                        f"Available keys: {list(self.config['episode_length_range'].keys())}\n"
                        f"Please add variable episode range for '{tf}'."
                    )
            
            logger.info(f"   {tf}: max_steps={self.config['max_steps_per_episode'][tf]}")
        
        # Check 2: Execution timeframe is in timeframes list
        logger.info("\n[2/3] Checking execution timeframe...")
        exec_tf = self.config.get('execution_timeframe', '5m')
        if exec_tf not in self.config['timeframes']:
            raise ValueError(
                f" Execution timeframe '{exec_tf}' not in timeframes list: "
                f"{self.config['timeframes']}"
            )
        logger.info(f"   Execution timeframe '{exec_tf}' is valid")
        
        # Check 3: Episode lengths are reasonable
        logger.info("\n[3/3] Checking episode length ranges...")
        for tf, max_steps in self.config['max_steps_per_episode'].items():
            if max_steps < 50:
                logger.warning(f"    {tf}: max_steps={max_steps} is very short (<50)")
            elif max_steps > 2000:
                logger.warning(f"    {tf}: max_steps={max_steps} is very long (>2000)")
            else:
                logger.info(f"   {tf}: max_steps={max_steps} is reasonable")
        
        logger.info("\n" + "="*60)
        logger.info(" CONFIGURATION VALIDATION PASSED")
        logger.info("="*60 + "\n")
    
    def train_complete_system(self, train_ml: bool = True, train_rl: bool = True) -> Dict:
        """
        Optimized complete training pipeline
        Target: 4-6 hours (vs 2 days original)
        
        âœ… FIXED: When train_rl=True but train_ml=False, will either:
        1. Load existing ML model, OR
        2. Quick train ML just for feature selection
        """
        logger.info("="*80)
        logger.info("OPTIMIZED HYBRID ML/RL SYSTEM TRAINING")
        if self.config['explainability']['enabled']:
            logger.info(" EXPLAINABILITY ENABLED")
        logger.info(f"Device: {self.device}")
        logger.info(f"CPU Workers: {self.config['num_workers']}")
        logger.info("="*80)
        
        self.training_results['start_time'] = datetime.now()
        overall_start = time.time()
        
        try:
            # Stage 1: Parallel data loading (2-5 min)
            stage_start = time.time()
            self.progress.start_stage('Data Loading', len(self.config['assets']) * len(self.config['timeframes']))
            data = self._fetch_data_parallel()
            
            # ✅ CRITICAL FIX: Store original multi-timeframe data for RL training
            self.raw_data = data
            
            self.progress.complete_stage('Data Loading', {
                'datasets_loaded': sum(len(v) for v in data.values()),
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Data loaded in {time.time() - stage_start:.1f}s")
            
            # Stage 2: Parallel feature calculation (5-10 min)
            stage_start = time.time()
            self.progress.start_stage('Feature Engineering', len(data))
            features = self._calculate_features_parallel(data)
            self.progress.complete_stage('Feature Engineering', {
                'feature_sets': len(features),
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Features calculated in {time.time() - stage_start:.1f}s")
            
            # Stage 3: Fast data splitting (< 1 min)
            stage_start = time.time()
            self.progress.start_stage('Data Splitting', len(data))
            train_data, val_data, test_data = self._split_data_fast(data, features)
            self.progress.complete_stage('Data Splitting', {
                'time_seconds': time.time() - stage_start
            })
            logger.info(f" Data split in {time.time() - stage_start:.1f}s")
            
            # Handle ML training for feature selection
            if train_rl and not train_ml:
                logger.info("\n  RL training requested without ML training")
                logger.info("   Attempting to load existing ML model for feature selection...")
                
                ml_loaded = self._try_load_ml_model()
                
                if not ml_loaded:
                    logger.warning("   No existing ML model found")
                    logger.info("   Quick-training ML for feature selection...")
                    
                    stage_start = time.time()
                    self.progress.start_stage('ML Feature Selection', 100)
                    self._train_ml_predictor_gpu(train_data, val_data)
                    self.progress.complete_stage('ML Feature Selection', {
                        'time_seconds': time.time() - stage_start,
                        'purpose': 'feature_selection_only'
                    })
                    logger.info(f" ML trained for feature selection in {time.time() - stage_start:.1f}s")
            
            # Stage 4: GPU-accelerated ML training (30-60 min)
            elif train_ml:
                stage_start = time.time()
                self.progress.start_stage('ML Training', 100)
                self._train_ml_predictor_gpu(train_data, val_data)
                self.progress.complete_stage('ML Training', {
                    'time_seconds': time.time() - stage_start,
                    'accuracy': self.training_results['ml_results'].get('train_accuracy', 0)
                })
                logger.info(f" ML trained in {time.time() - stage_start:.1f}s")
            
            # Stage 5: GPU-accelerated RL training (2-4 hours)
            if train_rl:
                stage_start = time.time()
                self.progress.start_stage('RL Training', self.config['rl_episodes'])

                logger.info("\n[Pre-RL] Filtering features to ML-selected features...")
                train_data_filtered = self._filter_features_to_selected(train_data)
                val_data_filtered = self._filter_features_to_selected(val_data)
                test_data_filtered = self._filter_features_to_selected(test_data)

                self._train_rl_agent_gpu(train_data_filtered, val_data_filtered, test_data_filtered)
                self.progress.complete_stage('RL Training', {
                    'time_seconds': time.time() - stage_start,
                    'episodes': self.config['rl_episodes']
                })
                logger.info(f" RL trained in {time.time() - stage_start:.1f}s")
            
            # Stage 6: Save models (< 1 min)
            self.training_results['end_time'] = datetime.now()
            stage_start = time.time()
            self.progress.start_stage('Saving Models', 2)
            self._save_models()
            self.progress.complete_stage('Saving Models', {
                'time_seconds': time.time() - stage_start
            })
            
            self.training_results['end_time'] = datetime.now()
            total_time = time.time() - overall_start
            
            logger.info("\n" + "="*80)
            logger.info(f" TRAINING COMPLETE - Total Time: {total_time/3600:.2f} hours")
            logger.info(f"  Speedup: ~{48/total_time:.1f}x faster (vs 2 days)")
            logger.info("="*80)
            
            self.training_results['speedup_metrics'] = {
                'total_time_hours': total_time / 3600,
                'estimated_speedup': 48 / (total_time / 3600)
            }
            
            print(self.progress.get_summary())
            
            return self.training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

    # ... [Keep all existing methods: _try_load_ml_model, _fetch_data_parallel, etc.] ...
    # I'll only show the modified methods below:

    def _try_load_ml_model(self) -> bool:
        """Try to load the most recent ML model"""
        try:
            from pathlib import Path
            import glob
            
            ml_models = glob.glob('models/ml/ml_predictor_*.pkl')
            
            if not ml_models:
                return False
            
            latest_model = max(ml_models, key=lambda x: Path(x).stat().st_mtime)
            
            logger.info(f"   Loading ML model: {latest_model}")
            
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
            
            logger.info(f" Loaded ML model with {len(self.ml_predictor.selected_features)} selected features")
            
            return True
            
        except Exception as e:
            logger.warning(f"   Failed to load ML model: {e}")
            return False
    
    def _fetch_data_parallel(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Load data in parallel using ProcessPoolExecutor"""
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
                    self.progress.update('Data Loading', pbar.n)
        
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
    
    def _calculate_features_parallel(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Calculate features in parallel for all assets"""
        features = {}
        
        cache_dir = Path('data/features_cache')
        if self.config['cache_features'] and cache_dir.exists():
            logger.info("Checking for cached features...")
            cached_count = 0
            for symbol in data.keys():
                cache_file = cache_dir / f'{symbol}_features.pkl'
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        features[symbol] = pickle.load(f)
                        cached_count += 1
            
            if cached_count > 0:
                logger.info(f"Loaded {cached_count} cached feature sets")
        
        symbols_to_process = [s for s in data.keys() if s not in features]
        
        if symbols_to_process:
            logger.info(f"Calculating features for {len(symbols_to_process)} assets...")
            
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
                            logger.error(f"Failed to calculate features for {symbol}: {e}")
                        
                        pbar.update(1)
                        self.progress.update('Feature Engineering', pbar.n)
        
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
        """Fast vectorized data splitting WITH proper buffers"""
        train_data, val_data, test_data = {}, {}, {}
        
        lookforward = self.config.get('lookforward', 10)
        buffer_size = self.config.get('buffer_size', 5)
        
        for symbol in tqdm(self.config['assets'], desc="Splitting data"):
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
                logger.warning(f"{symbol}: Insufficient data after buffers, skipping")
                continue
            
            train_data[symbol] = (ohlcv.iloc[:train_end], feats.iloc[:train_end])
            val_data[symbol] = (ohlcv.iloc[val_start:val_end], feats.iloc[val_start:val_end])
            test_data[symbol] = (ohlcv.iloc[test_start:], feats.iloc[test_start:])
            
            self.progress.update('Data Splitting', len(train_data))
        
        return train_data, val_data, test_data
    
    def _train_ml_predictor_gpu(self, train_data: Dict, val_data: Dict):
        """Train ML model"""
        logger.info("Training ML predictor...")
        
        from src.train_system import SystemTrainer as StandardTrainer
        
        standard_trainer = StandardTrainer()
        standard_trainer.config = self.config
        
        logger.info(f"  Training on {len(train_data)} assets...")
        
        try:
            standard_trainer._train_ml_predictor(train_data, val_data)
            
            self.ml_predictor = standard_trainer.ml_predictor
            self.training_results['ml_results'] = standard_trainer.training_results.get('ml_results', {})
            
            if self.training_results['ml_results']:
                self.progress.update('ML Training', 100, {
                    'accuracy': self.training_results['ml_results'].get('train_accuracy', 0),
                    'f1_score': self.training_results['ml_results'].get('train_f1', 0)
                })
                
                logger.info(f"  Train Accuracy: {self.training_results['ml_results'].get('train_accuracy', 0):.4f}")
                logger.info(f"  Train F1 Score: {self.training_results['ml_results'].get('train_f1', 0):.4f}")
            
        except Exception as e:
            logger.error(f"ML training failed: {e}")
            raise

    def _filter_features_to_selected(self, data_dict: Dict) -> Dict:
        """Filter features to only include selected features from ML training"""
        if not hasattr(self, 'ml_predictor') or self.ml_predictor is None:
            logger.warning("  No ML predictor found - using all features")
            return data_dict
        
        if not hasattr(self.ml_predictor, 'selected_features') or not self.ml_predictor.selected_features:
            logger.warning("  No selected features found - using all features")
            return data_dict
        
        selected_features = self.ml_predictor.selected_features
        logger.info(f"\n Filtering features to {len(selected_features)} selected features...")
        
        filtered_data = {}
        
        for asset, (ohlcv_df, features_df) in data_dict.items():
            available_features = [f for f in selected_features if f in features_df.columns]
            
            if len(available_features) < len(selected_features):
                missing = len(selected_features) - len(available_features)
                logger.warning(f"  {asset}: Only {len(available_features)}/{len(selected_features)} features available ({missing} missing)")
            
            filtered_features = features_df[available_features].copy()
            
            logger.info(f"  {asset}: {features_df.shape[1]} -> {filtered_features.shape[1]} features")
            
            filtered_data[asset] = (ohlcv_df, filtered_features)
        
        return filtered_data
    
    def _train_rl_agent_gpu(self, train_data: Dict, val_data: Dict, test_data: Dict):
        """
        Train RL agent with MULTI-TIMEFRAME ENVIRONMENTS + EXPLAINABILITY
        
        ✓ NEW: Uses multi-timeframe analysis for professional trading
        ✓ Agent sees ALL timeframes simultaneously (5m, 15m, 1h)
        ✓ Keeps all optimization features (GPU, parallel, explainability, etc.)
        """
        from src.models.dqn_agent import DQNAgent, DQNConfig
        from src.data.data_manager import DataManager
        # ✓ NEW: Use multi-timeframe environment
        from src.environment.multi_timeframe_env import MultiTimeframeEnvironment
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-TIMEFRAME RL TRAINING")
        if self.config['explainability']['enabled']:
            logger.info(" WITH EXPLAINABILITY")
        logger.info("="*80)
        logger.info("   Agent will see ALL timeframes simultaneously")
        logger.info("   Professional-grade multi-timeframe analysis")
        
        # [STEP 1: Load and group data by asset]
        logger.info("\n[1/5] Loading multi-timeframe data (grouped by asset)...")
        logger.info("   OPTIMIZATION: Using pre-calculated features from ML stage")
        
        feature_calc_start = time.time()
        data_manager = DataManager()
        
        # ✓ NEW: Group data by asset instead of asset-timeframe
        asset_environments = {}
        
        # ✅ CHECK: Ensure we have raw data
        if not hasattr(self, 'raw_data') or not self.raw_data:
            logger.error(" ERROR: raw_data not available!")
            logger.error("   Cannot perform proper multi-timeframe training.")
            logger.error("   Make sure you added 'self.raw_data = data' in train_complete_system()")
            raise ValueError("raw_data must be stored for multi-timeframe training")
        
        for asset in train_data.keys():
            # Get the features from train_data (old format - base TF only)
            ohlcv_base, features_df = train_data[asset]
            
            # ✅ NEW: Get multi-timeframe OHLCV from raw_data
            if asset not in self.raw_data:
                logger.warning(f"  {asset}: Not in raw_data, skipping")
                continue
            
            raw_timeframe_data = self.raw_data[asset]
            
            logger.info(f"\n  Processing {asset}...")
            
            # ✅ NEW: Use actual timeframe-specific data from raw_data
            dataframes = {}
            features_dfs = {}
            
            for timeframe in self.config['timeframes']:
                try:
                    if timeframe not in raw_timeframe_data:
                        logger.warning(f"    {timeframe}: Not available in raw_data")
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
                        logger.warning(f"    {timeframe}: Insufficient aligned data ({len(common_idx)} rows)")
                        continue
                    
                    dataframes[timeframe] = ohlcv_tf_train.loc[common_idx]
                    features_dfs[timeframe] = features_df.loc[common_idx]
                    
                    logger.info(f"    {timeframe}: {len(dataframes[timeframe]):,} candles "
                               f"({features_df.shape[1]} features)")
                    
                except Exception as e:
                    logger.error(f"    Error loading {timeframe}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # ← VALIDATION CHECKS (after timeframe loop ends)
            if len(dataframes) != len(self.config['timeframes']):
                missing = set(self.config['timeframes']) - set(dataframes.keys())
                logger.warning(f"   Skipping {asset}: missing timeframes {missing}")
                continue
            
            if len(self.config['timeframes']) >= 2:
                tf1 = self.config['timeframes'][0]
                tf2 = self.config['timeframes'][1]
                
                len1, len2 = len(dataframes[tf1]), len(dataframes[tf2])
                ratio = len1 / len2 if len2 > 0 else 0
                
                logger.info(f"    Validation: {tf1}={len1} rows, {tf2}={len2} rows (ratio: {ratio:.2f})")
                
                if abs(ratio - 1.0) < 0.01:
                    logger.warning(f"      WARNING: {tf1} and {tf2} have nearly identical length!")
                
                if len1 >= 5 and len2 >= 5:
                    ts1 = list(dataframes[tf1].index[:5])
                    ts2 = list(dataframes[tf2].index[:5])
                    
                    if ts1 == ts2:
                        logger.error(f"     ERROR: {tf1} and {tf2} have IDENTICAL timestamps!")
                        logger.error(f"    Multi-timeframe training will NOT work properly!")
            
            # ✅✅✅ THIS LINE MUST BE HERE - OUTSIDE ALL IF BLOCKS! ✅✅✅
            asset_environments[asset] = {
                'dataframes': dataframes,
                'features_dfs': features_dfs
            }
        
        # After ALL assets are processed:
        if not asset_environments:
            logger.error("  No valid multi-timeframe environments!")
            raise ValueError("No training data available")
        
        
        feature_calc_time = time.time() - feature_calc_start
        
        logger.info(f"\n  Loaded {len(asset_environments)} assets with multi-timeframe data")
        logger.info(f"  Feature calculation time: {feature_calc_time:.2f}s (reused from ML!) ")
        
        # [STEP 2: Create first environment to get dimensions]
        logger.info("\n[2/5] Initializing universal multi-timeframe agent...")
        
        first_asset = list(asset_environments.keys())[0]
        first_data = asset_environments[first_asset]
        
        temp_env = MultiTimeframeEnvironment(
            dataframes=first_data['dataframes'],
            features_dfs=first_data['features_dfs'],
            execution_timeframe=self.config.get('execution_timeframe', '5m'),
            initial_balance=self.config['initial_balance'],
            fee_rate=self.config['fee_rate'],
            slippage=self.config.get('slippage', 0.001),
            stop_loss=self.config.get('stop_loss'),
            take_profit=self.config.get('take_profit'),
            asset=first_asset,
            selected_features=self.ml_predictor.selected_features if hasattr(self.ml_predictor, 'selected_features') else None
        )
        
        state_dim = temp_env.observation_space_shape[0]
        action_dim = temp_env.action_space_n
        
        logger.info(f"   Multi-timeframe environment created")
        logger.info(f"      State dims: {state_dim} (includes ALL timeframes!)")
        logger.info(f"      Timeframes: {temp_env.timeframes}")
        logger.info(f"      Execution TF: {temp_env.execution_timeframe}")
        
        # Initialize RL agent
        rl_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['rl_hidden_dims'],
            use_double_dqn=self.config['use_double_dqn'],
            use_dueling_dqn=self.config['use_dueling_dqn'],
            batch_size=self.config.get('rl_batch_size', 512),
            use_prioritized_replay=self.config.get('use_prioritized_replay', True)
        )
        
        self.rl_agent = DQNAgent(config=rl_config)
        
        logger.info(f"\n   Agent initialized")
        logger.info(f"      Network: {self.config['rl_hidden_dims']}")
        logger.info(f"      Device: {self.rl_agent.device}")
        
        # [STEP 3: Initialize explainer if enabled] - UNCHANGED
        if self.config['explainability']['enabled']:
            logger.info("\n  Setting up explainability system...")
            
            if hasattr(temp_env, 'get_feature_names'):
                state_feature_names = temp_env.get_feature_names()
            else:
                state_feature_names = [f'feature_{i}' for i in range(state_dim)]
            
            self.explainer = ExplainableRL(
                agent=self.rl_agent,
                state_feature_names=state_feature_names,
                action_names=['Hold', 'Buy', 'Sell'],
                explain_frequency=self.config['explainability']['explain_frequency'],
                verbose=self.config['explainability']['verbose'],
                save_dir=self.config['explainability']['save_dir']
            )
            
            logger.info(f"   Explainability enabled")
            logger.info(f"      Tracking {len(state_feature_names)} features")
            logger.info(f"      Explain frequency: every {self.config['explainability']['explain_frequency']} steps")
        
        # [STEP 4: Create all multi-timeframe environments]
        logger.info("\n[3/5] Creating multi-timeframe environments for all assets...")
        
        environments = {}
        for asset, data in asset_environments.items():
            try:
                env = MultiTimeframeEnvironment(
                    dataframes=data['dataframes'],
                    features_dfs=data['features_dfs'],
                    execution_timeframe=self.config.get('execution_timeframe', '5m'),
                    initial_balance=self.config['initial_balance'],
                    fee_rate=self.config['fee_rate'],
                    slippage=self.config.get('slippage', 0.001),
                    stop_loss=self.config.get('stop_loss'),
                    take_profit=self.config.get('take_profit'),
                    asset=asset,
                    selected_features=self.ml_predictor.selected_features if hasattr(self.ml_predictor, 'selected_features') else None
                )
                
                environments[asset] = {
                    'env': env,
                    'asset': asset,
                    'timeframe': self.config.get('execution_timeframe', '5m'),  # For compatibility
                    'is_multi_timeframe': True,  # NEW: Flag for special handling
                    'execution_timeframe': env.execution_timeframe,  # NEW: Actual execution TF
                    'all_timeframes': env.timeframes  # NEW: All timeframes available
                }
                
                logger.info(f"   {asset}: Multi-TF environment created")
                
            except Exception as e:
                logger.error(f"  Error creating environment for {asset}: {e}")
                continue
        
        logger.info(f"\n   Created {len(environments)} multi-timeframe environments")
        
        # [STEP 5: Training strategy] - UNCHANGED
        logger.info("\n[4/5] Setting up training strategy...")
        
        total_episodes = self.config['rl_episodes']
        episodes_per_asset = max(1, total_episodes // len(environments))
        training_mode = self.config.get('rl_training_mode', 'random')
        
        logger.info(f"  Training mode: {training_mode}")
        logger.info(f"  Total episodes: {total_episodes}")
        logger.info(f"  Assets: {len(environments)}")
        logger.info(f"  Episodes per asset: ~{episodes_per_asset}")
        
        # [STEP 6: Execute training] - UNCHANGED ROUTING
        logger.info("\n[5/5] Starting multi-timeframe training...")
        logger.info("="*80)
        
        if training_mode == 'sequential':
            episode_results = self._train_sequential(
                environments, episodes_per_asset, total_episodes
            )
        elif training_mode == 'interleaved':
            episode_results = self._train_interleaved(
                environments, total_episodes
            )
        elif training_mode == 'random':
            episode_results = self._train_random(
                environments, total_episodes
            )
        else:
            logger.warning(f"Unknown mode '{training_mode}', using random")
            episode_results = self._train_random(
                environments, total_episodes
            )
        
        # Generate explainability report if enabled - UNCHANGED
        if self.explainer:
            logger.info("\n" + "="*80)
            logger.info("GENERATING EXPLAINABILITY REPORT")
            logger.info("="*80)
            
            final_report = self.explainer.generate_final_report()
            print("\n" + final_report)
        
        # Generate training report - UNCHANGED
        logger.info("\n" + "="*80)
        logger.info("GENERATING MULTI-TIMEFRAME TRAINING REPORT")
        logger.info("="*80)
        
        from src.utils.rl_reporter_fast import FastRLTrainingReporter
        
        reporter = FastRLTrainingReporter()
        report = reporter.generate_full_report(
            episode_results=episode_results,
            env=list(environments.values())[0]['env'],
            agent=self.rl_agent,
            config=self.config,
            save_path=f'results/rl_multi_timeframe_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        
        print("\n" + report)
        
        self._generate_detailed_analysis(episode_results)
        
        logger.info("\n  Multi-timeframe training complete!")
        logger.info(f"  Agent trained on {len(set(e['asset'] for e in episode_results))} assets")
        logger.info(f"  Each asset had {len(self.config['timeframes'])} timeframes")
        logger.info(f"  Total episodes: {len(episode_results)}")
        logger.info(f"  Agent memory: {len(self.rl_agent.memory)} experiences")

    def _train_sequential(self, environments: Dict, episodes_per_combo: int, 
                        total_episodes: int) -> list:
        """Train sequentially - UNCHANGED"""
        episode_results = []
        global_episode = 0
        
        logger.info("\n  SEQUENTIAL TRAINING MODE")
        logger.info("  Training each asset-timeframe combo in order\n")
        
        for env_name, env_info in environments.items():
            env = env_info['env']
            asset = env_info['asset']
            timeframe = env_info['timeframe']
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Training: {asset} @ {timeframe}")
            logger.info(f"{'='*80}")
            
            with tqdm(total=episodes_per_combo, desc=f"{asset} {timeframe}") as pbar:
                for episode in range(episodes_per_combo):
                    stats = self._run_training_episode(
                        env, asset, timeframe, global_episode
                    )
                    episode_results.append(stats)
                    global_episode += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'reward': f"{stats['total_reward']:.2f}",
                        'epsilon': f"{stats['epsilon']:.3f}"
                    })
                    
                    if global_episode >= total_episodes:
                        break
            
            if global_episode >= total_episodes:
                break
            
            recent = episode_results[-episodes_per_combo:]
            logger.info(f"  Completed {asset} {timeframe}")
            logger.info(f"    Avg Reward: {np.mean([e['total_reward'] for e in recent]):.2f}")
            logger.info(f"    Memory Size: {len(self.rl_agent.memory)}")
        
        return episode_results

    def _train_interleaved(self, environments: Dict, total_episodes: int) -> list:
        """Train interleaved - UNCHANGED"""
        import itertools
        
        episode_results = []
        
        logger.info("\n  INTERLEAVED TRAINING MODE")
        logger.info("  Rotating through all asset-timeframe combinations\n")
        
        env_names = list(environments.keys())
        env_cycle = itertools.cycle(env_names)
        
        with tqdm(total=total_episodes, desc="Multi-Dimensional Training") as pbar:
            for episode in range(total_episodes):
                env_name = next(env_cycle)
                env_info = environments[env_name]
                
                stats = self._run_training_episode(
                    env_info['env'],
                    env_info['asset'],
                    env_info['timeframe'],
                    episode
                )
                episode_results.append(stats)
                
                pbar.update(1)
                pbar.set_postfix({
                    'combo': f"{env_info['asset']}_{env_info['timeframe']}",
                    'reward': f"{stats['total_reward']:.2f}",
                    'epsilon': f"{stats['epsilon']:.3f}"
                })
                
                if (episode + 1) % 20 == 0:
                    self._log_progress(episode_results[-20:], episode + 1, total_episodes)

                # âš¡ OVERFITTING FIX: Validation monitoring and early stopping
                if (episode + 1) % self.config.get('validation_frequency', 50) == 0:
                    val_results = self._run_validation(environments, episode + 1)
                    
                    # Check for improvement (early stopping)
                    if hasattr(self, 'best_val_reward'):
                        improvement = val_results['avg_reward'] - self.best_val_reward
                        if improvement > self.config.get('early_stopping_min_delta', 0.01):
                            self.best_val_reward = val_results['avg_reward']
                            self.patience_counter = 0
                            logger.info(f"   VALIDATION IMPROVED: {val_results['avg_reward']:.2f} (+{improvement:.2f})")
                        else:
                            self.patience_counter += 1
                            logger.info(f"   No significant improvement (patience: {self.patience_counter}/{self.config.get('early_stopping_patience', 5)})")
                            
                            if self.patience_counter >= self.config.get('early_stopping_patience', 5):
                                logger.info("  EARLY STOPPING: Validation performance not improving")
                                break
                    else:
                        self.best_val_reward = val_results['avg_reward']
                        self.patience_counter = 0
                        logger.info(f"  Initial validation reward: {val_results['avg_reward']:.2f}")

        
        return episode_results

    def _train_random(self, environments: Dict, total_episodes: int) -> list:
        """Train with random selection - UNCHANGED"""
        import random
        
        episode_results = []
        env_names = list(environments.keys())
        
        logger.info("\n  RANDOM TRAINING MODE")
        logger.info("  Randomly selecting asset-timeframe each episode\n")
        
        with tqdm(total=total_episodes, desc="Multi-Dimensional Training") as pbar:
            for episode in range(total_episodes):
                env_name = random.choice(env_names)
                env_info = environments[env_name]
                
                stats = self._run_training_episode(
                    env_info['env'],
                    env_info['asset'],
                    env_info['timeframe'],
                    episode
                )
                episode_results.append(stats)
                
                pbar.update(1)
                pbar.set_postfix({
                    'combo': f"{env_info['asset']}_{env_info['timeframe']}",
                    'reward': f"{stats['total_reward']:.2f}",
                    'epsilon': f"{stats['epsilon']:.3f}"
                })
                
                if (episode + 1) % 20 == 0:
                    self._log_progress(episode_results[-20:], episode + 1, total_episodes)

                # âš¡ OVERFITTING FIX: Validation monitoring and early stopping
                if (episode + 1) % self.config.get('validation_frequency', 50) == 0:
                    val_results = self._run_validation(environments, episode + 1)
                    
                    # Check for improvement (early stopping)
                    if hasattr(self, 'best_val_reward'):
                        improvement = val_results['avg_reward'] - self.best_val_reward
                        if improvement > self.config.get('early_stopping_min_delta', 0.01):
                            self.best_val_reward = val_results['avg_reward']
                            self.patience_counter = 0
                            logger.info(f"   VALIDATION IMPROVED: {val_results['avg_reward']:.2f} (+{improvement:.2f})")
                        else:
                            self.patience_counter += 1
                            logger.info(f"   No significant improvement (patience: {self.patience_counter}/{self.config.get('early_stopping_patience', 5)})")
                            
                            if self.patience_counter >= self.config.get('early_stopping_patience', 5):
                                logger.info("  EARLY STOPPING: Validation performance not improving")
                                break
                    else:
                        self.best_val_reward = val_results['avg_reward']
                        self.patience_counter = 0
                        logger.info(f"  Initial validation reward: {val_results['avg_reward']:.2f}")

        
        return episode_results


    def _run_validation(self, environments: Dict, episode_num: int) -> dict:
        """
        âš¡ OVERFITTING FIX: Run validation episodes to monitor generalization
        
        Runs N validation episodes across all environments without exploration
        to measure how well the agent generalizes.
        """
        import random
        
        val_episodes = self.config.get('validation_episodes', 20)
        val_results = []
        
        # Save current epsilon and set to 0 for greedy validation
        original_epsilon = self.rl_agent.epsilon
        self.rl_agent.epsilon = 0.0
        
        logger.info(f"\n Running validation ({val_episodes} episodes, greedy)...")
        
        env_names = list(environments.keys())
        for _ in range(val_episodes):
            env_name = random.choice(env_names)
            env_info = environments[env_name]
            
            # Run episode without training
            state = env_info['env'].reset()
            episode_reward = 0
            done = False
            steps = 0
            if env_info.get('is_multi_timeframe', False):
                exec_tf = self.config.get('execution_timeframe', '5m')
                max_steps = self.config['max_steps_per_episode'].get(exec_tf, 500)
            else:
                max_steps = self.config['max_steps_per_episode'].get(env_info['timeframe'], 500)
            
            while not done and steps < max_steps:
                action = self.rl_agent.act(state, training=False)
                next_state, reward, done, truncated, _ = env_info['env'].step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                
                if done or truncated:
                    break
            
            val_results.append(episode_reward)
        
        # Restore original epsilon
        self.rl_agent.epsilon = original_epsilon
        
        avg_reward = np.mean(val_results)
        std_reward = np.std(val_results)
        max_reward = np.max(val_results)
        min_reward = np.min(val_results)
        
        logger.info(f"  Validation Results (Episode {episode_num}):")
        logger.info(f"    Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"    Range: [{min_reward:.2f}, {max_reward:.2f}]")
        
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'episode': episode_num
        }

    def _run_training_episode(self, env, asset: str, timeframe: str, 
                            episode_num: int) -> dict:
        """
        Run a single training episode WITH OPTIONAL EXPLAINABILITY
        
        â­ MODIFIED: Now uses explainer when enabled
        """
        episode_start = time.time()
        
        # âš¡ OVERFITTING FIX: Variable episode lengths
        if self.config.get('variable_episode_length', False):
            min_steps, max_steps = self.config['episode_length_range'][timeframe]
            MAX_STEPS = np.random.randint(min_steps, max_steps + 1)
        else:
            MAX_STEPS = self.config['max_steps_per_episode'].get(timeframe, 500)
        
        # â­ NEW: Use explainer if enabled
        if self.explainer:
            # Build context for explainer
            context = {
                'asset': asset,
                'timeframe': timeframe,
                'episode': episode_num + 1
            }
            
            # Train with explainability
            state = env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done and steps < MAX_STEPS:
                # âœ… FIX: Get context for this step with proper attribute names and bounds checking
                try:
                    # Multi-timeframe environment
                    if hasattr(env, 'synchronized_data') and hasattr(env, 'current_step'):
                        if 0 <= env.current_step < len(env.synchronized_data):
                            exec_index = env.synchronized_data.index[env.current_step]
                            current_price = float(env.dataframes[env.execution_timeframe].loc[exec_index, 'close'])
                        else:
                            current_price = 0.0
                    # Single-timeframe environment (fallback)
                    elif hasattr(env, 'df') and hasattr(env, 'current_step'):
                        if 0 <= env.current_step < len(env.df):
                            current_price = float(env.df.iloc[env.current_step]['close'])
                        else:
                            current_price = 0.0
                    else:
                        current_price = 0.0
                except (KeyError, IndexError, TypeError, AttributeError):
                    current_price = 0.0

                step_context = {
                    'price': current_price,
                    'position': env.position if hasattr(env, 'position') else 0,
                    'balance': env.balance if hasattr(env, 'balance') else 0,
                    'asset': asset,
                    'timeframe': timeframe
                }
                
                # Act with explanation
                action, explanation = self.explainer.act_with_explanation(
                    state=state,
                    context=step_context,
                    training=True
                )
                
                # Environment step
                next_state, reward, done, truncated, info = env.step(action)
                
                bootstrap_done = done and not truncated

                self.rl_agent.remember(state, action, reward, next_state, bootstrap_done)
                
                # Increment step_count
                self.rl_agent.step_count += 1
                
                # Use correct replay frequency (every 4 steps, not 10)
                if self.rl_agent.step_count % self.rl_agent.config.update_every == 0:
                    if len(self.rl_agent.memory) >= self.rl_agent.config.min_memory_size:
                        self.rl_agent.replay()
                
                # Update target network periodically
                if self.rl_agent.step_count % self.rl_agent.config.target_update_every == 0:
                    self.rl_agent.update_target_network(soft=True)
                
                # Decay epsilon
                if not self.rl_agent.config.use_noisy_networks:
                    self.rl_agent.epsilon = max(
                        self.rl_agent.config.epsilon_end,
                        self.rl_agent.epsilon * self.rl_agent.config.epsilon_decay
                    )
                
                state = next_state
                episode_reward += reward
                steps += 1
            
            self.rl_agent.episode_count += 1
            # Episode summary
            self.explainer.episode_summary(episode_num, episode_reward, steps)
            
            # Save periodic reports
            if (episode_num + 1) % 10 == 0:
                self.explainer.save_episode_report(episode_num)
            
            stats = {
                'total_reward': episode_reward,
                'steps': steps,
                'epsilon': self.rl_agent.epsilon,
                'num_trades': info.get('num_trades', 0),
                'portfolio_value': info.get('portfolio_value', 0),
                'win_rate': info.get('win_rate', 0),
                'sharpe_ratio': info.get('sharpe_ratio', 0),
                'max_drawdown': info.get('max_drawdown', 0),
                'winning_trades': sum(1 for t in env.trades if (t.get('pnl') or 0) > 0) if hasattr(env, 'trades') else 0,
                'losing_trades': sum(1 for t in env.trades if (t.get('pnl') or 0) < 0) if hasattr(env, 'trades') else 0,
                'trades': [dict(t) for t in env.trades]
            }
        else:
            # Standard training (no explainability)
            stats = self.rl_agent.train_episode(env, max_steps=MAX_STEPS)
        
        # Add metadata
        stats['episode_time'] = time.time() - episode_start
        stats['asset'] = asset
        stats['symbol'] = asset
        stats['timeframe'] = timeframe
        stats['episode'] = episode_num + 1
        stats['combination'] = f"{asset}_{timeframe}"
        
        return stats

    def _log_progress(self, recent_episodes: list, current: int, total: int):
        """Log training progress - UNCHANGED"""
        avg_reward = np.mean([e['total_reward'] for e in recent_episodes])
        
        by_asset = {}
        for e in recent_episodes:
            asset = e['asset']
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(e['total_reward'])
        
        by_timeframe = {}
        for e in recent_episodes:
            tf = e['timeframe']
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(e['total_reward'])
        
        logger.info(f"\n   Progress: Episode {current}/{total}")
        logger.info(f"    Overall Avg Reward: {avg_reward:.2f}")
        logger.info(f"    Epsilon: {self.rl_agent.epsilon:.3f}")
        logger.info(f"    Memory: {len(self.rl_agent.memory):,} experiences")
        
        if len(by_asset) > 1:
            logger.info(f"\n    Per-Asset (last 20 episodes):")
            for asset in sorted(by_asset.keys()):
                rewards = by_asset[asset]
                logger.info(f"      {asset:10s}: {np.mean(rewards):6.2f} avg ({len(rewards)} eps)")
        
        if len(by_timeframe) > 1:
            logger.info(f"\n    Per-Timeframe (last 20 episodes):")
            for tf in sorted(by_timeframe.keys()):
                rewards = by_timeframe[tf]
                logger.info(f"      {tf:4s}: {np.mean(rewards):6.2f} avg ({len(rewards)} eps)")

    def _generate_detailed_analysis(self, episode_results: list):
        """Generate detailed analysis - UNCHANGED"""
        logger.info("\n" + "="*80)
        logger.info("DETAILED PERFORMANCE ANALYSIS")
        logger.info("="*80)
        
        by_asset = {}
        for e in episode_results:
            asset = e['asset']
            if asset not in by_asset:
                by_asset[asset] = []
            by_asset[asset].append(e)
        
        logger.info("\n  Per-Asset Performance:")
        for asset in sorted(by_asset.keys()):
            episodes = by_asset[asset]
            rewards = [e['total_reward'] for e in episodes]
            trades = sum(e.get('num_trades', 0) for e in episodes)
            
            logger.info(f"\n  {asset}:")
            logger.info(f"    Episodes: {len(episodes)}")
            logger.info(f"    Avg Reward: {np.mean(rewards):.2f}")
            logger.info(f"    Best Reward: {max(rewards):.2f}")
            logger.info(f"    Total Trades: {trades}")
        
        by_timeframe = {}
        for e in episode_results:
            tf = e['timeframe']
            if tf not in by_timeframe:
                by_timeframe[tf] = []
            by_timeframe[tf].append(e)
        
        logger.info("\n  Per-Timeframe Performance:")
        for tf in sorted(by_timeframe.keys()):
            episodes = by_timeframe[tf]
            rewards = [e['total_reward'] for e in episodes]
            trades = sum(e.get('num_trades', 0) for e in episodes)
            
            logger.info(f"\n  {tf}:")
            logger.info(f"    Episodes: {len(episodes)}")
            logger.info(f"    Avg Reward: {np.mean(rewards):.2f}")
            logger.info(f"    Best Reward: {max(rewards):.2f}")
            logger.info(f"    Total Trades: {trades}")
        
        logger.info("\n  Top 5 Asset-Timeframe Combinations:")
        combo_performance = {}
        for e in episode_results:
            combo = f"{e['asset']}_{e['timeframe']}"
            if combo not in combo_performance:
                combo_performance[combo] = []
            combo_performance[combo].append(e['total_reward'])
        
        combo_avgs = {k: np.mean(v) for k, v in combo_performance.items()}
        top_combos = sorted(combo_avgs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (combo, avg_reward) in enumerate(top_combos, 1):
            num_episodes = len(combo_performance[combo])
            logger.info(f"  {i}. {combo:20s}: {avg_reward:7.2f} avg ({num_episodes} episodes)")
    
    def _save_models(self):
        """Save trained models - UNCHANGED"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if self.ml_predictor:
            ml_path = Path(f'models/ml/ml_predictor_{timestamp}.pkl')
            ml_path.parent.mkdir(parents=True, exist_ok=True)
            self.ml_predictor.save_model(ml_path)
            logger.info(f"Saved ML model to {ml_path}")
            self.progress.update('Saving Models', 1)
        
        if self.rl_agent:
            rl_path = Path(f'models/rl/dqn_agent_{timestamp}.pth')
            rl_path.parent.mkdir(parents=True, exist_ok=True)
            self.rl_agent.save(rl_path)
            logger.info(f"Saved RL agent to {rl_path}")
            self.progress.update('Saving Models', 2)

        if self.feature_engineer:
            import joblib
            fe_path = Path('models/feature_engineer.pkl')
            fe_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.feature_engineer, fe_path)
            logger.info(f"Saved feature engineer to {fe_path}")
            self.progress.update('Saving Models', 3)
        
        results_path = Path(f'results/training_results_{timestamp}.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_to_save = {
            'start_time': self.training_results.get('start_time').isoformat() if self.training_results.get('start_time') else None,
            'end_time': self.training_results.get('end_time').isoformat() if self.training_results.get('end_time') else datetime.now().isoformat(),
            'speedup_metrics': self.training_results.get('speedup_metrics', {}),
            'ml_results': self.training_results.get('ml_results'),
            'rl_results': self.training_results.get('rl_results')
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")


# â­ MODIFIED: Convenience functions now accept explainability params
def train_ml_only_fast(config_path: Optional[str] = None) -> Dict:
    """Fast ML-only training"""
    trainer = OptimizedSystemTrainer(config_path)
    return trainer.train_complete_system(train_ml=True, train_rl=False)


def train_rl_only_fast(config_path: Optional[str] = None,
                      explain: bool = False,
                      explain_freq: int = 100,
                      verbose: bool = False,
                      explain_dir: str = 'logs/explanations') -> Dict:
    """
    Fast RL-only training WITH OPTIONAL EXPLAINABILITY
    
    Args:
        config_path: Path to config file
        explain: Enable explainability system
        explain_freq: How often to print explanations (every N steps)
        verbose: Explain EVERY decision (warning: lots of output)
        explain_dir: Directory to save explanations
    """
    trainer = OptimizedSystemTrainer(config_path)
    
    # â­ NEW: Set explainability config
    if explain:
        trainer.config['explainability'] = {
            'enabled': True,
            'verbose': verbose,
            'explain_frequency': explain_freq,
            'save_dir': explain_dir
        }
    
    return trainer.train_complete_system(train_ml=False, train_rl=True)


def train_both_fast(config_path: Optional[str] = None,
                   explain: bool = False,
                   explain_freq: int = 100,
                   verbose: bool = False,
                   explain_dir: str = 'logs/explanations') -> Dict:
    """
    Fast complete training WITH OPTIONAL EXPLAINABILITY
    
    Args:
        config_path: Path to config file
        explain: Enable explainability system
        explain_freq: How often to print explanations
        verbose: Explain every decision
        explain_dir: Directory to save explanations
    """
    trainer = OptimizedSystemTrainer(config_path)
    
    # â­ NEW: Set explainability config
    if explain:
        trainer.config['explainability'] = {
            'enabled': True,
            'verbose': verbose,
            'explain_frequency': explain_freq,
            'save_dir': explain_dir
        }
    
    return trainer.train_complete_system(train_ml=True, train_rl=True)


if __name__ == "__main__":
    import logging
    from src.utils.logger import setup_logger
    
    logger = setup_logger('optimized_training', level=logging.INFO)
    
    print("="*80)
    print("OPTIMIZED TRAINING SYSTEM + EXPLAINABILITY")
    print("Expected speedup: 10-20x faster")
    print("With agent decision insights")
    print("="*80)