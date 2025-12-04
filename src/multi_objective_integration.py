"""
Multi-Objective RL Integration with Existing Trading System

This module bridges the Multi-Objective RL system with your existing
data pipeline, feature engineering, and training infrastructure.

Usage:
    python main.py train --rl --multi-objective
    
    or in code:
    
    from multi_objective_integration import run_multi_objective_training
    results = run_multi_objective_training(data_manager, config)

Author: Claude
Version: 1.0
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Import existing project modules (adjust paths as needed)
try:
    from src.data.data_manager import DataManager
    from src.feature_engineer import FeatureEngineer
except ImportError:
    # Running standalone
    DataManager = None
    FeatureEngineer = None

# Import multi-objective modules
from multi_objective_rewards import MultiObjectiveConfig, MultiObjectiveRewardCalculator
from multi_objective_dqn import MultiObjectiveAgent, MODQNConfig, OBJECTIVES
from multi_objective_trainer import (
    MultiObjectiveTrainer, 
    MOTradeConfig,
    MultiObjectiveTradingEnv,
    convert_existing_data_format
)

logger = logging.getLogger(__name__)


def prepare_data_for_mo_training(
    data_manager,
    feature_engineer,
    symbols: List[str],
    execution_timeframe: str = '5m',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[Dict, Dict, Dict]:
    """
    Prepare data from existing pipeline for Multi-Objective training
    
    Args:
        data_manager: Your DataManager instance
        feature_engineer: Your FeatureEngineer instance
        symbols: List of symbols to train on
        execution_timeframe: Primary timeframe for execution
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        
    Returns:
        train_data, val_data, test_data dictionaries
    """
    
    train_data = {}
    val_data = {}
    test_data = {}
    
    for symbol in symbols:
        logger.info(f"Preparing data for {symbol}...")
        
        # Load OHLCV data
        ohlcv = data_manager.load_data(symbol, execution_timeframe)
        
        if ohlcv is None or len(ohlcv) < 1000:
            logger.warning(f"Skipping {symbol} - insufficient data")
            continue
        
        # Calculate features
        features = feature_engineer.calculate_all_features(ohlcv)
        
        # Drop NaN rows
        valid_idx = features.dropna().index
        ohlcv = ohlcv.loc[valid_idx]
        features = features.loc[valid_idx]
        
        # Split data
        n = len(ohlcv)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Training data
        train_data[symbol] = {
            'prices': ohlcv.iloc[:train_end][['close', 'high', 'low', 'open', 'volume']].copy(),
            'features': features.iloc[:train_end].copy()
        }
        
        # Validation data
        val_data[symbol] = {
            'prices': ohlcv.iloc[train_end:val_end][['close', 'high', 'low', 'open', 'volume']].copy(),
            'features': features.iloc[train_end:val_end].copy()
        }
        
        # Test data
        test_data[symbol] = {
            'prices': ohlcv.iloc[val_end:][['close', 'high', 'low', 'open', 'volume']].copy(),
            'features': features.iloc[val_end:].copy()
        }
        
        logger.info(f"  {symbol}: train={len(train_data[symbol]['prices'])}, "
                   f"val={len(val_data[symbol]['prices'])}, "
                   f"test={len(test_data[symbol]['prices'])}")
    
    return train_data, val_data, test_data


def run_multi_objective_training(
    train_data: Dict[str, Dict[str, pd.DataFrame]],
    val_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "models/multi_objective"
) -> Dict[str, Any]:
    """
    Run Multi-Objective RL training
    
    Args:
        train_data: {symbol: {'prices': df, 'features': df}}
        val_data: Same format for validation
        config: Configuration overrides
        output_dir: Where to save models
        
    Returns:
        Training results
    """
    
    # Default config
    default_config = {
        # Episode structure
        'max_wait_steps': 200,
        'max_hold_steps': 300,
        'min_hold_steps': 12,
        
        # Trading parameters
        'stop_loss_pct': 0.03,
        'take_profit_pct': 0.03,
        'fee_rate': 0.001,
        
        # Multi-objective weights
        'weight_pnl_quality': 0.35,
        'weight_hold_duration': 0.25,
        'weight_win_achieved': 0.15,
        'weight_loss_control': 0.15,
        'weight_risk_reward': 0.10,
        
        # Training
        'num_episodes': 8000,
        'batch_size': 64,
        'learning_rate': 0.001,
        'log_interval': 100,
        'save_interval': 1000,
    }
    
    if config:
        default_config.update(config)
    
    # Create MOTradeConfig
    mo_config = MOTradeConfig(**{
        k: v for k, v in default_config.items()
        if hasattr(MOTradeConfig, k) or k in MOTradeConfig.__dataclass_fields__
    })
    
    # Create trainer
    trainer = MultiObjectiveTrainer(
        train_data=train_data,
        val_data=val_data,
        config=mo_config,
        output_dir=output_dir
    )
    
    # Train
    results = trainer.train()
    
    return results


def evaluate_mo_agent(
    agent: MultiObjectiveAgent,
    test_data: Dict[str, Dict[str, pd.DataFrame]],
    config: MOTradeConfig
) -> Dict[str, Any]:
    """
    Evaluate trained Multi-Objective agent on test data
    
    Returns per-objective and overall metrics
    """
    
    all_trades = []
    
    for symbol, data in test_data.items():
        env = MultiObjectiveTradingEnv(
            price_data=data['prices'],
            feature_data=data['features'],
            config=config,
            asset=symbol
        )
        
        # Run through test data
        num_test_episodes = min(500, len(data['prices']) // 400)
        
        for _ in range(num_test_episodes):
            state = env.reset()
            done = False
            episode_rewards = {obj: 0.0 for obj in OBJECTIVES}
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, rewards, done, info = env.step(action)
                
                for obj in OBJECTIVES:
                    episode_rewards[obj] += rewards[obj]
                
                state = next_state
            
            if env.episode_info['exit_reason'] and env.episode_info['exit_reason'] != 'search_timeout':
                all_trades.append({
                    'symbol': symbol,
                    'pnl_pct': env.episode_info['pnl_pct'],
                    'hold_duration': env.episode_info['hold_duration'],
                    'exit_reason': env.episode_info['exit_reason'],
                    **episode_rewards
                })
    
    if not all_trades:
        return {'error': 'No trades executed'}
    
    # Calculate metrics
    trades_df = pd.DataFrame(all_trades)
    
    results = {
        'num_trades': len(trades_df),
        'overall': {
            'avg_pnl_pct': trades_df['pnl_pct'].mean() * 100,
            'total_pnl_pct': trades_df['pnl_pct'].sum() * 100,
            'win_rate': (trades_df['pnl_pct'] > 0).mean() * 100,
            'avg_hold': trades_df['hold_duration'].mean(),
        },
        'by_exit_reason': trades_df.groupby('exit_reason')['pnl_pct'].agg(['count', 'mean', 'sum']).to_dict(),
        'by_objective': {
            obj: {
                'mean': trades_df[obj].mean(),
                'std': trades_df[obj].std(),
            }
            for obj in OBJECTIVES
        }
    }
    
    return results


class MultiObjectiveTrainingCallback:
    """
    Callback for integration with main.py training loop
    """
    
    def __init__(self, trainer: MultiObjectiveTrainer):
        self.trainer = trainer
        self.start_time = None
    
    def on_training_start(self):
        self.start_time = datetime.now()
        logger.info("Multi-Objective training started")
    
    def on_episode_end(self, episode: int, info: Dict):
        pass  # Logging handled by trainer
    
    def on_training_end(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Multi-Objective training complete in {duration:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Standalone training script
    
    Usage:
        python multi_objective_integration.py --symbols ETH_USDT,SOL_USDT --episodes 5000
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Objective RL Training')
    parser.add_argument('--symbols', type=str, default='ETH_USDT,SOL_USDT,ADA_USDT',
                       help='Comma-separated list of symbols')
    parser.add_argument('--episodes', type=int, default=8000,
                       help='Number of training episodes')
    parser.add_argument('--output-dir', type=str, default='models/multi_objective',
                       help='Output directory for models')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    print("=" * 70)
    print("MULTI-OBJECTIVE RL TRAINING")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)
    
    # Check if we can import the data manager
    if DataManager is None:
        print("\nDataManager not available - using synthetic data for demo")
        
        # Create synthetic data
        train_data = {}
        for symbol in symbols:
            n = 50000
            np.random.seed(hash(symbol) % 2**32)
            
            returns = np.random.randn(n) * 0.001
            prices = 100 * np.exp(np.cumsum(returns))
            
            train_data[symbol] = {
                'prices': pd.DataFrame({
                    'close': prices,
                    'high': prices * 1.001,
                    'low': prices * 0.999,
                    'open': prices,
                    'volume': np.random.randint(1000, 10000, n)
                }),
                'features': pd.DataFrame(
                    np.random.randn(n, 50),
                    columns=[f'feature_{i}' for i in range(50)]
                )
            }
        
        val_data = None
    else:
        # Use real data pipeline
        data_manager = DataManager(data_dir=args.data_dir)
        feature_engineer = FeatureEngineer()
        
        train_data, val_data, test_data = prepare_data_for_mo_training(
            data_manager,
            feature_engineer,
            symbols=symbols,
        )
    
    # Run training
    results = run_multi_objective_training(
        train_data=train_data,
        val_data=val_data,
        config={'num_episodes': args.episodes},
        output_dir=args.output_dir
    )
    
    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()