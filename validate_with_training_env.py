"""
Validate Agent Using EXACT Training Environment
================================================

This script uses the TradeBasedMultiTimeframeEnv (same as training)
to evaluate the agent on 2025 data.

If this shows good results but walk_forward_runner shows bad results,
then there's a state construction mismatch in the backtest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import logging
from datetime import datetime
import inspect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_selected_features():
    """Load selected features from ML model - handles multiple formats"""
    import joblib
    
    # Try loading from separate features file first (if extract script was run)
    features_file = Path('models/features/selected_features.pkl')
    if features_file.exists():
        selected = joblib.load(features_file)
        logger.info(f"  ✓ Loaded {len(selected)} features from {features_file}")
        return list(selected)
    
    # Load from ML model dict
    ml_path = Path('models/ml/ml_predictor.pkl')
    ml_dict = joblib.load(ml_path)
    
    logger.info(f"  ML model type: {type(ml_dict)}")
    
    if isinstance(ml_dict, dict):
        logger.info(f"  ML model keys: {list(ml_dict.keys())}")
        if 'selected_features' in ml_dict:
            selected = ml_dict['selected_features']
            logger.info(f"  ✓ Found 'selected_features': {len(selected)} features")
            return list(selected)
    
    logger.error("  ✗ Could not find selected_features!")
    return None


def load_agent():
    """Load the trained MO DQN agent"""
    from src.multi_objective_extension import MultiObjectiveDQNAgent, MODQNConfig
    
    checkpoint_path = Path('models/rl/dqn_agent.pth')
    # weights_only=False is safe here since we trust our own checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    
    # Extract config
    config_obj = checkpoint['config']
    
    config = MODQNConfig(
        state_dim=config_obj.state_dim,
        action_dim=config_obj.action_dim,
        hidden_dims=list(config_obj.hidden_dims),
        objective_weights=dict(config_obj.objective_weights)
    )
    
    agent = MultiObjectiveDQNAgent(config)
    agent.q_network.load_state_dict(checkpoint['q_network'])
    agent.target_network.load_state_dict(checkpoint['target_network'])
    agent.epsilon = 0.0  # Greedy for evaluation
    
    logger.info(f"Loaded agent with weights: {config.objective_weights}")
    
    return agent


def load_2025_data(symbol='SOL_USDT'):
    """Load 2025 data for validation"""
    from src.features.feature_engineer import FeatureEngineer
    
    timeframes = ['5m', '15m', '1h']
    data_path = Path('data/raw')
    
    dataframes = {}
    for tf in timeframes:
        file_path = data_path / tf / f"{symbol}_{tf}.csv"
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Filter to 2025
        df = df[df.index >= '2024-01-01']
        df = df[df.index < '2024-06-01']
        
        dataframes[tf] = df
        logger.info(f"  Loaded {len(df)} rows for {tf}")
    
    # Calculate features from 5m (like training)
    fe = FeatureEngineer()
    base_features = fe.calculate_all_features(
        df=dataframes['5m'],
        symbol=symbol.replace('_', '/')
    )
    
    logger.info(f"  Raw features calculated: {base_features.shape[1]} columns")
    
    # Load selected features using helper
    selected_features = load_selected_features()
    
    if selected_features:
        available = [f for f in selected_features if f in base_features.columns]
        if available:
            base_features = base_features[available]
            logger.info(f"  Filtered to {len(available)} selected features")
        else:
            logger.warning(f"  No selected features found in base_features columns!")
    else:
        logger.warning(f"  ✗ No selected_features - using ALL {base_features.shape[1]} features")
    
    # Create features_dfs for all timeframes (same features, different alignment)
    features_dfs = {}
    for tf in timeframes:
        common_idx = base_features.index.intersection(dataframes[tf].index)
        features_dfs[tf] = base_features.loc[common_idx]
        logger.info(f"  {tf}: {len(features_dfs[tf])} rows, {features_dfs[tf].shape[1]} features")
    
    return dataframes, features_dfs, selected_features


def inspect_environment():
    """Inspect the TradeBasedMultiTimeframeEnv to see its actual signature"""
    try:
        from src.environment.trade_based_mtf_env import TradeBasedMultiTimeframeEnv, TradeConfig
        
        # Get __init__ signature
        sig = inspect.signature(TradeBasedMultiTimeframeEnv.__init__)
        
        logger.info("\n" + "="*60)
        logger.info("TradeBasedMultiTimeframeEnv.__init__ signature:")
        logger.info("="*60)
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            default = param.default if param.default is not inspect.Parameter.empty else "REQUIRED"
            logger.info(f"  {name}: {default}")
        
        # Also check TradeConfig
        if hasattr(TradeConfig, '__dataclass_fields__'):
            logger.info("\nTradeConfig fields:")
            for name, field in TradeConfig.__dataclass_fields__.items():
                logger.info(f"  {name}: {field.default if field.default is not field.default_factory else 'factory'}")
        
        return TradeBasedMultiTimeframeEnv, TradeConfig
        
    except Exception as e:
        logger.error(f"Failed to import environment: {e}")
        return None, None


def create_environment(dataframes, features_dfs, EnvClass, ConfigClass, selected_features=None):
    """Create environment by inspecting what parameters it accepts"""
    
    sig = inspect.signature(EnvClass.__init__)
    param_names = [p for p in sig.parameters.keys() if p != 'self']
    
    logger.info(f"  selected_features passed: {selected_features is not None}, count: {len(selected_features) if selected_features else 0}")
    logger.info(f"  'selected_features' in param_names: {'selected_features' in param_names}")
    
    # Build kwargs based on what the environment accepts
    kwargs = {}
    
    # Core data - try different parameter names
    if 'dataframes' in param_names:
        kwargs['dataframes'] = dataframes
    if 'features_dfs' in param_names:
        kwargs['features_dfs'] = features_dfs
    
    # Timeframes
    if 'timeframes' in param_names:
        kwargs['timeframes'] = ['5m', '15m', '1h']
    if 'execution_timeframe' in param_names:
        kwargs['execution_timeframe'] = '5m'
    
    # Asset
    if 'asset' in param_names:
        kwargs['asset'] = 'SOL_USDT'
    
    # Balance
    if 'initial_balance' in param_names:
        kwargs['initial_balance'] = 10000
    
    # Fees/Slippage
    if 'fee_rate' in param_names:
        kwargs['fee_rate'] = 0.001
    if 'slippage' in param_names:
        kwargs['slippage'] = 0.001
    
    # Stop/Take
    if 'stop_loss' in param_names:
        kwargs['stop_loss'] = 0.03
    if 'take_profit' in param_names:
        kwargs['take_profit'] = 0.06
    
    # CRITICAL: Pass selected_features to filter from 191 to 50 features!
    if 'selected_features' in param_names:
        if selected_features is not None:
            kwargs['selected_features'] = selected_features
            logger.info(f"  ✓ Passing {len(selected_features)} selected features to environment")
        else:
            logger.warning("  ✗ selected_features is None, environment will use ALL features!")
    
    # Trade config
    if 'trade_config' in param_names and ConfigClass is not None:
        kwargs['trade_config'] = ConfigClass()
    
    logger.info(f"\nCreating environment with kwargs: {list(kwargs.keys())}")
    
    return EnvClass(**kwargs)


def validate_with_training_env(agent, dataframes, features_dfs, selected_features, num_episodes=100):
    """
    Run validation using the EXACT same environment as training
    """
    EnvClass, ConfigClass = inspect_environment()
    
    if EnvClass is None:
        logger.error("Could not load environment class!")
        return None
    
    logger.info(f"  Using {len(selected_features) if selected_features else 0} selected features")
    
    # Create environment with 2025 data AND selected_features
    env = create_environment(dataframes, features_dfs, EnvClass, ConfigClass, selected_features)
    
    logger.info(f"\nEnvironment created:")
    logger.info(f"  Observation shape: {env.observation_space_shape}")
    logger.info(f"  Action space: {env.action_space_n}")
    
    # Run episodes
    episode_results = []
    
    for ep in range(num_episodes):
        # Handle both old (state) and new (state, info) reset API
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            
            # Handle both old (4-value) and new (5-value) Gym API
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            state = next_state
        
        # Get trade result
        trade_result = env.trade_result
        if trade_result:
            # Debug: print available attributes
            if ep == 0:
                logger.info(f"  TradeResult attributes: {[a for a in dir(trade_result) if not a.startswith('_')]}")
            
            # Try different attribute names
            pnl_pct = getattr(trade_result, 'pnl_pct', None) or getattr(trade_result, 'pnl', 0)
            hold_steps = getattr(trade_result, 'hold_steps', None) or getattr(trade_result, 'hold_duration', 0) or getattr(trade_result, 'steps_held', 0)
            exit_reason = getattr(trade_result, 'exit_reason', 'unknown')
            
            episode_results.append({
                'episode': ep,
                'pnl_pct': pnl_pct * 100 if pnl_pct else 0,
                'hold_steps': hold_steps,
                'exit_reason': exit_reason,
                'win': (pnl_pct or 0) > 0,
            })
            
            if (ep + 1) % 20 == 0:
                recent = episode_results[-20:]
                win_rate = sum(1 for r in recent if r['win']) / len(recent)
                avg_pnl = np.mean([r['pnl_pct'] for r in recent])
                logger.info(f"  Episode {ep+1}: Last 20 avg P&L: {avg_pnl:+.3f}%, Win rate: {win_rate*100:.1f}%")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS (Using Training Environment)")
    logger.info("="*60)
    
    if episode_results:
        df = pd.DataFrame(episode_results)
        
        win_rate = df['win'].mean()
        avg_pnl = df['pnl_pct'].mean()
        avg_hold = df['hold_steps'].mean()
        
        logger.info(f"  Episodes: {len(df)}")
        logger.info(f"  Win Rate: {win_rate*100:.1f}%")
        logger.info(f"  Avg P&L: {avg_pnl:+.3f}%")
        logger.info(f"  Avg Hold: {avg_hold:.1f} steps ({avg_hold*5:.1f} min)")
        
        # Exit reasons
        logger.info("\n  Exit Reasons:")
        for reason, count in df['exit_reason'].value_counts().items():
            logger.info(f"    {reason}: {count} ({count/len(df)*100:.1f}%)")
        
        # Compare first few states with backtest
        logger.info("\n" + "-"*60)
        logger.info("SAMPLE STATES (for comparison with backtest)")
        logger.info("-"*60)
        
        reset_result = env.reset()
        state = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        logger.info(f"  State shape: {state.shape}")
        logger.info(f"  First 5 values: {state[:5]}")
        logger.info(f"  Last 5 values: {state[-5:]}")
        logger.info(f"  Market features (150): sum={state[:150].sum():.4f}, mean={state[:150].mean():.4f}")
        
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_hold': avg_hold,
            'results': df
        }
    
    return None


def compare_states(agent, dataframes, features_dfs, selected_features):
    """
    Compare state construction between training env and backtest
    """
    logger.info("\n" + "="*60)
    logger.info("STATE COMPARISON: Training Env vs Backtest")
    logger.info("="*60)
    
    EnvClass, ConfigClass = inspect_environment()
    
    if EnvClass is None:
        return
    
    env = create_environment(dataframes, features_dfs, EnvClass, ConfigClass, selected_features)
    
    # Get state from training env - handle both old and new Gym API
    reset_result = env.reset()
    state_train = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    
    # Get action from agent
    action_train = agent.act(state_train, training=False)
    
    logger.info(f"\nTraining Environment State:")
    logger.info(f"  Shape: {state_train.shape}")
    logger.info(f"  First 10: {state_train[:10]}")
    logger.info(f"  Position info (150-155): {state_train[150:155]}")
    logger.info(f"  Account info (155-160): {state_train[155:160]}")
    logger.info(f"  Asset encoding (160-165): {state_train[160:165]}")
    logger.info(f"  TF encoding (165-183): {state_train[165:183]}")
    logger.info(f"  Phase encoding (183-185): {state_train[183:185]}")
    logger.info(f"  Phase info (185-188): {state_train[185:188]}")
    logger.info(f"  Action chosen: {action_train}")
    
    # Get Q-values
    state_tensor = torch.FloatTensor(state_train).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_dict = agent.q_network(state_tensor)
        combined = agent.q_network.get_combined_q_values(state_tensor, agent.config.objective_weights)
        
    logger.info(f"\nQ-Values from Training State:")
    for obj, q in q_dict.items():
        logger.info(f"  {obj}: {q.cpu().numpy().flatten()}")
    logger.info(f"  Combined: {combined.cpu().numpy().flatten()}")


def main():
    logger.info("="*60)
    logger.info("VALIDATION USING TRAINING ENVIRONMENT")
    logger.info("="*60)
    
    # First, inspect environment to see what parameters it takes
    EnvClass, ConfigClass = inspect_environment()
    
    if EnvClass is None:
        logger.error("Cannot proceed without environment class")
        return
    
    # Load agent
    logger.info("\nLoading agent...")
    agent = load_agent()
    
    # Load 2025 data (now returns selected_features too)
    logger.info("\nLoading 2025 data...")
    dataframes, features_dfs, selected_features = load_2025_data()
    
    if selected_features is None:
        logger.error("Cannot proceed without selected_features!")
        return
    
    # Run validation
    logger.info("\nRunning validation with training environment...")
    results = validate_with_training_env(agent, dataframes, features_dfs, selected_features, num_episodes=100)
    
    # Compare states
    compare_states(agent, dataframes, features_dfs, selected_features)
    
    if results:
        logger.info("\n" + "="*60)
        logger.info("CONCLUSION")
        logger.info("="*60)
        
        if results['avg_pnl'] > -0.1:
            logger.info("   Training env validation shows REASONABLE performance")
            logger.info("   If backtest shows much worse, there's a STATE MISMATCH")
        else:
            logger.info("   Training env validation also shows POOR performance")
            logger.info("   Agent genuinely doesn't generalize to 2025 data")


if __name__ == '__main__':
    main()