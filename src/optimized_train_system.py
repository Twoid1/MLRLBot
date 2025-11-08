"""
OPTIMIZED Training System - 10-20x Speed Improvement + EXPLAINABILITY
- GPU acceleration for PyTorch and XGBoost
- Parallel processing for multi-asset operations
- Vectorized feature calculations
- Progress tracking with live updates
- Memory optimization
- ⭐ NEW: Explainability system to understand agent decisions
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

# ⭐ NEW: Explainability import
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
    ⭐ NEW: With explainability to understand agent decisions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
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
        
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()

        self.ml_predictor = None  # Initialize as None
        self.rl_agent = None 
        
        # ⭐ NEW: Explainability
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
            'timeframes': ['1h', '4h', '1d'],
            
            # Data settings
            'start_date': '2021-01-01',
            'end_date': '2025-01-01',
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
                '1h': 500,
                '4h': 300,
                '1d': 100
            },
            'rl_episodes': 2500,
            'rl_hidden_dims': [256, 256, 128],
            'use_double_dqn': True,
            'use_dueling_dqn': True,
            'rl_batch_size': 1024,  # Larger for GPU
            'rl_update_frequency': 4,  # Update every N steps (faster)
            'rl_memory_size': 100000,
            
            # Progress tracking
            'log_interval': 10,  # Log every N episodes
            'save_interval': 50,  # Save checkpoint every N episodes
            
            # Environment
            'initial_balance': 10000,
            'fee_rate': 0.0026,
            'slippage': 0.001,
            'stop_loss': 0.03,
            'take_profit': 0.05,
            
            # ⭐ NEW: Explainability settings
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
    
    def train_complete_system(self, train_ml: bool = True, train_rl: bool = True) -> Dict:
        """
        Optimized complete training pipeline
        Target: 4-6 hours (vs 2 days original)
        
        ✅ FIXED: When train_rl=True but train_ml=False, will either:
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
        Train RL agent on multiple assets and timeframes WITH EXPLAINABILITY
        
        ⭐ MODIFIED: Now includes explainability system
        """
        from src.models.dqn_agent import DQNAgent, DQNConfig
        from src.environment.trading_env import TradingEnvironment
        
        logger.info("\n" + "="*80)
        logger.info("MULTI-ASSET & MULTI-TIMEFRAME RL TRAINING")
        if self.config['explainability']['enabled']:
            logger.info(" WITH EXPLAINABILITY")
        logger.info("="*80)
        
        # [STEP 1: Load data - UNCHANGED]
        logger.info("\n[1/5] Loading multi-asset, multi-timeframe data...")
        logger.info("   OPTIMIZATION: Using pre-calculated features from ML stage")
        
        feature_calc_start = time.time()
        
        training_combinations = []
        
        for asset in train_data.keys():
            ohlcv_df, features_df = train_data[asset]
            
            for timeframe in self.config['timeframes']:
                try:
                    ohlcv_train = ohlcv_df
                    features_train = features_df
                    
                    common_idx = ohlcv_train.index.intersection(features_train.index)
                    ohlcv_train = ohlcv_train.loc[common_idx]
                    features_train = features_train.loc[common_idx]
                    
                    if len(ohlcv_train) < 100:
                        logger.warning(f"   {asset} {timeframe} insufficient data")
                        continue
                    
                    training_combinations.append({
                        'asset': asset,
                        'timeframe': timeframe,
                        'ohlcv': ohlcv_train,
                        'features': features_train,
                        'name': f"{asset}_{timeframe}"
                    })
                    
                    logger.info(f"   {asset:10s} {timeframe:4s}: {len(ohlcv_train):,} candles "
                               f"({features_train.shape[1]} features, pre-calculated)")
                    
                except Exception as e:
                    logger.error(f"   Error loading {asset} {timeframe}: {e}")
                    continue
        
        if not training_combinations:
            logger.error("  No valid training combinations found!")
            raise ValueError("No training data available")
        
        feature_calc_time = time.time() - feature_calc_start
        
        logger.info(f"\n Loaded {len(training_combinations)} asset-timeframe combinations")
        logger.info(f"  Total training data: {sum(len(c['ohlcv']) for c in training_combinations):,} candles")
        logger.info(f"  Feature calculation time: {feature_calc_time:.2f}s (reused from ML stage!) 🚀")
        
        # [STEP 2: Initialize agent - UNCHANGED]
        logger.info("\n[2/5] Initializing universal trading agent...")
        
        first_combo = training_combinations[0]
        temp_env = TradingEnvironment(
            df=first_combo['ohlcv'],
            initial_balance=self.config['initial_balance'],
            fee_rate=self.config['fee_rate'],
            features_df=first_combo['features'],
            selected_features=self.ml_predictor.selected_features if hasattr(self.ml_predictor, 'selected_features') else None,
            precompute_observations=True
        )
        
        state_dim = temp_env.observation_space_shape[0]
        action_dim = temp_env.action_space_n
        
        rl_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config['rl_hidden_dims'],
            use_double_dqn=self.config['use_double_dqn'],
            use_dueling_dqn=self.config['use_dueling_dqn'],
            batch_size=self.config.get('rl_batch_size', 64),
            use_prioritized_replay=self.config.get('use_prioritized_replay', False)
        )
        
        self.rl_agent = DQNAgent(config=rl_config)
        
        logger.info(f" Agent initialized")
        logger.info(f"  State dimensions: {state_dim}")
        logger.info(f"  Action dimensions: {action_dim}")
        logger.info(f"  Network: {self.config['rl_hidden_dims']}")
        logger.info(f"  Device: {self.rl_agent.device}")
        
        # ⭐ NEW: Initialize explainer if enabled
        if self.config['explainability']['enabled']:
            logger.info("\n  Setting up explainability system...")
            
            # Get feature names
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
            
            logger.info(f" Explainability enabled")
            logger.info(f"  Tracking {len(state_feature_names)} features")
            logger.info(f"  Explain frequency: every {self.config['explainability']['explain_frequency']} steps")
            logger.info(f"  Verbose mode: {'ON' if self.config['explainability']['verbose'] else 'OFF'}")
            logger.info(f"  Save directory: {self.config['explainability']['save_dir']}")
        
        # [STEP 3: Create environments - UNCHANGED]
        logger.info("\n[3/5] Creating trading environments...")
        
        environments = {}
        for combo in training_combinations:
            name = combo['name']
            asset = combo['asset']
            timeframe = combo['timeframe']
            
            env = TradingEnvironment(
                df=combo['ohlcv'],
                initial_balance=self.config['initial_balance'],
                fee_rate=self.config['fee_rate'],
                slippage=self.config['slippage'],
                features_df=combo['features'],
                selected_features=self.ml_predictor.selected_features if hasattr(self.ml_predictor, 'selected_features') else None,
                stop_loss=self.config.get('stop_loss'),
                take_profit=self.config.get('take_profit'),
                precompute_observations=True,
                asset=asset,
                timeframe=timeframe  
            )
            
            environments[name] = {
                'env': env,
                'asset': combo['asset'],
                'timeframe': combo['timeframe']
            }
        
        logger.info(f" Created {len(environments)} environments")
        
        # [STEP 4: Training strategy - UNCHANGED]
        logger.info("\n[4/5] Setting up training strategy...")
        
        total_episodes = self.config['rl_episodes']
        episodes_per_combo = max(1, total_episodes // len(training_combinations))
        training_mode = self.config.get('rl_training_mode', 'interleaved')
        
        logger.info(f"  Training mode: {training_mode}")
        logger.info(f"  Total episodes: {total_episodes}")
        logger.info(f"  Combinations: {len(training_combinations)}")
        logger.info(f"  Episodes per combo: ~{episodes_per_combo}")
        
        # [STEP 5: Execute training - UNCHANGED ROUTING]
        logger.info("\n[5/5] Starting multi-dimensional training...")
        logger.info("="*80)
        
        if training_mode == 'sequential':
            episode_results = self._train_sequential(
                environments, episodes_per_combo, total_episodes
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
            logger.warning(f"Unknown mode '{training_mode}', using interleaved")
            episode_results = self._train_interleaved(
                environments, total_episodes
            )
        
        # ⭐ NEW: Generate explainability report if enabled
        if self.explainer:
            logger.info("\n" + "="*80)
            logger.info("GENERATING EXPLAINABILITY REPORT")
            logger.info("="*80)
            
            final_report = self.explainer.generate_final_report()
            print("\n" + final_report)
        
        # [STEP 6: Generate report - UNCHANGED]
        logger.info("\n" + "="*80)
        logger.info("GENERATING MULTI-DIMENSIONAL TRAINING REPORT")
        logger.info("="*80)
        
        from src.utils.rl_reporter import RLTrainingReporter
        
        reporter = RLTrainingReporter()
        report = reporter.generate_full_report(
            episode_results=episode_results,
            env=list(environments.values())[0]['env'],
            agent=self.rl_agent,
            config=self.config,
            save_path=f'results/rl_multi_dimensional_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        )
        
        print("\n" + report)
        
        self._generate_detailed_analysis(episode_results)
        
        logger.info("\n Multi-dimensional training complete!")
        logger.info(f"  Agent trained on {len(set(e['asset'] for e in episode_results))} assets")
        logger.info(f"  Agent trained on {len(set(e['timeframe'] for e in episode_results))} timeframes")
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
        
        return episode_results

    def _run_training_episode(self, env, asset: str, timeframe: str, 
                            episode_num: int) -> dict:
        """
        Run a single training episode WITH OPTIONAL EXPLAINABILITY
        
        ⭐ MODIFIED: Now uses explainer when enabled
        """
        episode_start = time.time()
        
        MAX_STEPS = self.config['max_steps_per_episode'].get(timeframe, 500)
        
        # ⭐ NEW: Use explainer if enabled
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
                # ✅ FIX: Get context for this step with proper attribute names and bounds checking
                try:
                    if hasattr(env, 'df') and hasattr(env, 'current_step') and 0 <= env.current_step < len(env.df):
                        current_price = float(env.df.iloc[env.current_step]['close'])
                    else:
                        current_price = 0.0
                except (KeyError, IndexError, TypeError):
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
                
                # Remember and train
                self.rl_agent.remember(state, action, reward, next_state, done)
                
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
                'trades': env.trades if hasattr(env, 'trades') else []
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


# ⭐ MODIFIED: Convenience functions now accept explainability params
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
    
    # ⭐ NEW: Set explainability config
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
    
    # ⭐ NEW: Set explainability config
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