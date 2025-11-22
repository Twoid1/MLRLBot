"""
Walk-Forward Validation Runner WITH EXPLAINABILITY
Complete pipeline for running walk-forward validation with proper asset identification
⭐ NEW: Explainability to understand agent decisions on test data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib
import torch
from typing import Dict, List, Optional, Tuple

# ⭐ NEW: Import explainability
from src.explainability_integration import ExplainableRL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardRunner:
    """
    Complete walk-forward validation runner WITH EXPLAINABILITY
    Handles data loading, feature engineering with symbols, model loading, and backtesting
    
    ⭐ NEW: Can explain WHY the agent makes each decision during validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize walk-forward runner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.data_path = Path(self.config['data_path'])
        self.models_path = Path(self.config['models_path'])
        
        # Models will be loaded
        self.ml_model = None
        self.rl_agent = None
        self.feature_engineer = None
        self.feature_selector = None
        
        # ⭐ NEW: Explainability
        self.explainer = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_path': 'data/raw/',  # Base path for all timeframes
            'models_path': 'models/',
            'symbols': ['ETH_USDT', 'DOT_USDT', 'SOL_USDT', 'ADA_USDT', 'AVAX_USDT'],
            'timeframes': ['5m', '15m', '1h'],  # ✅ FIXED: Day trading timeframes
            'execution_timeframe': '5m',  # ✅ NEW: Execution timeframe for trading
            'test_start_date': '2022-04-01',
            'test_end_date': '2023-01-01',
            'initial_balance': 100,
            'fee': 0.0026,  # ✅ FIXED: Match training
            'slippage': 0.001,  # ✅ FIXED: Match training
            # Explainability settings
            'explain': False,
            'explain_frequency': 100,
            'verbose': False,
            'explain_dir': 'logs/backtest_explanations'
        }
    
    def load_models(self):
        """
        Load trained ML model, RL agent, and feature engineer
        """
        logger.info("Loading trained models...")
        
        try:
            # Load ML model
            ml_path = self.models_path / 'ml' / 'ml_predictor.pkl'
            if ml_path.exists():
                from src.models.ml_predictor import MLPredictor
                
                self.ml_model = MLPredictor()
                self.ml_model.load_model(str(ml_path))
                logger.info(f" ML model loaded from {ml_path}")
            else:
                logger.warning(f" ML model not found at {ml_path}")
                
            # Load RL agent
            rl_path = self.models_path / 'rl' / 'dqn_agent.pth'
            if rl_path.exists():
                from src.models.dqn_agent import DQNAgent, DQNConfig
                
                checkpoint = torch.load(rl_path, weights_only=False, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                
                if 'config_dict' in checkpoint:
                    config_dict = checkpoint['config_dict']
                    
                    state_dim = int(config_dict.get('state_dim', 81))
                    action_dim = int(config_dict.get('action_dim', 3))
                    
                    hidden_dims_raw = config_dict.get('hidden_dims', [256, 256, 128])
                    if isinstance(hidden_dims_raw, list):
                        hidden_dims = [int(x) for x in hidden_dims_raw]
                    else:
                        hidden_dims = [256, 256, 128]
                    
                    use_dueling = bool(config_dict.get('use_dueling_dqn', True))
                    use_double = bool(config_dict.get('use_double_dqn', True))
                    use_noisy = bool(config_dict.get('use_noisy_networks', False))
                    
                    logger.info(f"   Extracted config: state_dim={state_dim}, action_dim={action_dim}")
                    logger.info(f"   Hidden dims: {hidden_dims}")
                    logger.info(f"   Dueling: {use_dueling}, Double: {use_double}, Noisy: {use_noisy}")
                    
                    rl_config = DQNConfig(
                        state_dim=state_dim,
                        action_dim=action_dim,
                        hidden_dims=hidden_dims,
                        use_dueling_dqn=use_dueling,
                        use_double_dqn=use_double,
                        use_noisy_networks=use_noisy
                    )
                    
                    self.rl_agent = DQNAgent(config=rl_config)
                    
                    self.rl_agent.q_network.load_state_dict(checkpoint['q_network_state'])
                    self.rl_agent.target_network.load_state_dict(checkpoint['target_network_state'])
                    self.rl_agent.q_network.eval()
                    
                    logger.info(f" RL agent loaded from {rl_path}")
                    logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
                else:
                    logger.error(" Checkpoint missing 'config_dict'")
            else:
                logger.warning(f" RL agent not found at {rl_path}")
                
            # Load feature engineer
            feature_path = self.models_path / 'feature_engineer.pkl'
            if feature_path.exists():
                self.feature_engineer = joblib.load(feature_path)
                logger.info(f" Feature engineer loaded from {feature_path}")
            else:
                logger.warning(f" Feature engineer not found, creating new one")
                from src.features.feature_engineer import FeatureEngineer
                self.feature_engineer = FeatureEngineer()
                
        except Exception as e:
            logger.error(f" Error loading models: {str(e)}")
            raise
    
    def load_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes
        
        Args:
            symbol: Symbol to load (e.g., 'ETH_USDT')
            
        Returns:
            Dictionary {timeframe: DataFrame}
        """
        data_dict = {}
        
        for timeframe in self.config['timeframes']:
            filepath = self.data_path / timeframe / f"{symbol}_{timeframe}.csv"
            
            if not filepath.exists():
                logger.warning(f" Data file not found: {filepath}")
                continue
                
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            data_dict[timeframe] = df
            logger.info(f" Loaded {len(df)} rows for {symbol} {timeframe}")
        
        return data_dict
    
    def prepare_features(self, data_dict: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """
        Calculate features with multi-timeframe data
        
        Args:
            data_dict: Dictionary of {timeframe: DataFrame}
            symbol: Symbol name (e.g., 'ETH_USDT')
            
        Returns:
            DataFrame with features
        """
        logger.info(f"Calculating features for {symbol}...")
        
        # Convert symbol format (ETH_USDT -> ETH/USDT)
        symbol_formatted = symbol.replace('_', '/')
        
        # ✅ FIXED: Get main timeframe data dynamically
        execution_tf = self.config.get('execution_timeframe', '5m')
        main_df = data_dict.get(execution_tf)
        if main_df is None:
            raise ValueError(f"No {execution_tf} data found for {symbol}")
        
        # Calculate features with multi-timeframe support
        features = self.feature_engineer.calculate_all_features(
            df=main_df,
            symbol=symbol_formatted
        )
        
        logger.info(f" Calculated {len(features.columns)} features for {symbol}")

        if len(data_dict) > 1:
            logger.info(f" Calculating multi-timeframe features from {list(data_dict.keys())}...")
            mtf_features = self.feature_engineer.calculate_multi_timeframe_features(data_dict)
            
            # Merge multi-timeframe features with base features
            features = pd.concat([features, mtf_features], axis=1)
            logger.info(f" Total features after adding multi-timeframe: {len(features.columns)}")
        
        # Verify symbol encoding
        if 'symbol_encoded' in features.columns:
            symbol_value = features['symbol_encoded'].iloc[0]
            logger.info(f" Symbol encoded as: {symbol_value}")
        else:
            logger.warning(f" WARNING: symbol_encoded not found in features!")
        
        return features
    
    def get_ml_predictions(self, symbol: str, features: pd.DataFrame) -> np.ndarray:
        """
        Get ML model predictions (returns numpy array)
        
        Args:
            symbol: Trading symbol (e.g., 'ETH_USDT')
            features: DataFrame with all calculated features
            
        Returns:
            Numpy array of predictions with shape (n_samples, 3) for [down, flat, up] probabilities
        """
        try:
            if self.ml_model is None:
                logger.warning(" ML model not loaded, returning neutral predictions")
                return np.array([[0.33, 0.34, 0.33]] * len(features))
            
            if not hasattr(self.ml_model, 'selected_features') or self.ml_model.selected_features is None:
                logger.error(" ML model doesn't have selected_features attribute")
                return np.array([[0.33, 0.34, 0.33]] * len(features))
            
            selected_features = self.ml_model.selected_features
            logger.info(f"Using {len(selected_features)} selected features for prediction")
            
            # Check for missing features
            missing_features = [f for f in selected_features if f not in features.columns]
            if missing_features:
                logger.error(f" Missing features: {missing_features[:5]}...")
                logger.error(f"   Total missing: {len(missing_features)} features")
                return np.array([[0.33, 0.34, 0.33]] * len(features))
            
            # Select features
            X = features[selected_features].copy()
            
            # Handle NaN values
            if X.isnull().any().any():
                logger.warning(f" Found NaN values in features, filling with 0")
                X = X.ffill().fillna(0)

            nan_count = X.isnull().sum().sum()
            if nan_count > 0:
                nan_pct = (nan_count / (len(X) * len(X.columns))) * 100
                logger.warning(f" Filled {nan_count} NaN values ({nan_pct:.2f}% of data)")
            
            # Scale features
            X_scaled = self.ml_model.scaler.transform(X)
            
            # Get predictions
            predictions = self.ml_model.best_model.predict_proba(X_scaled)
            
            # Verify shape
            if predictions.shape[1] != 3:
                logger.error(f" Invalid prediction shape: {predictions.shape}")
                return np.array([[0.33, 0.34, 0.33]] * len(features))
            
            logger.info(f" Generated {len(predictions)} predictions successfully")
            return predictions  # Returns numpy array
            
        except Exception as e:
            logger.error(f" Error getting ML predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([[0.33, 0.34, 0.33]] * len(features))
    
    def run_backtest_for_symbol(self, symbol: str) -> Dict:
        """
        Run backtest with multi-timeframe data
        
        ✅ UPDATED: Now passes multi-timeframe data to RL agent
        
        Args:
            symbol: Symbol to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running walk-forward validation for {symbol}")
        if self.config.get('explain', False):
            logger.info(" WITH EXPLAINABILITY")
        logger.info(f"{'='*60}")
        
        try:
            # 1. Load multi-timeframe data
            data_dict = self.load_multi_timeframe_data(symbol)
            
            execution_tf = self.config.get('execution_timeframe', '5m')
            if execution_tf not in data_dict:
                logger.error(f"   No {execution_tf} data for {symbol}, skipping")
                return {
                    'symbol': symbol,
                    'total_return': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'error': f'No {execution_tf} data available'
                }
            
            # 2. Filter to test period
            test_start = pd.to_datetime(self.config['test_start_date'])
            test_end = pd.to_datetime(self.config['test_end_date'])
            
            # Filter all timeframes
            filtered_data_dict = {}
            for tf, df in data_dict.items():
                filtered_data_dict[tf] = df.loc[test_start:test_end]
            
            test_data = filtered_data_dict[execution_tf]
            
            logger.info(f"Test period: {test_start} to {test_end}")
            logger.info(f"Test data points ({execution_tf}): {len(test_data)}")
            
            # 3. Calculate features for EACH timeframe separately
            logger.info("Calculating features for each timeframe...")
            features_dict = {}
            
            # Convert symbol format (ETH_USDT -> ETH/USDT)
            symbol_formatted = symbol.replace('_', '/')
            
            for tf in self.config['timeframes']:
                if tf not in filtered_data_dict:
                    logger.warning(f"   Skipping {tf} - no data")
                    continue
                
                # ✅ FIXED: Use calculate_all_features() with symbol parameter
                tf_features = self.feature_engineer.calculate_all_features(
                    df=filtered_data_dict[tf],
                    symbol=symbol_formatted
                )
                
                # Filter to selected features if available
                if hasattr(self.ml_model, 'selected_features') and self.ml_model.selected_features:
                    selected_features = self.ml_model.selected_features
                    available_features = [f for f in selected_features if f in tf_features.columns]
                    if available_features:
                        tf_features = tf_features[available_features].copy()
                        logger.info(f"   {tf}: {len(available_features)} features")
                    else:
                        logger.warning(f"   {tf}: No selected features found, using all")
                
                features_dict[tf] = tf_features
            
            # 4. Get ML predictions (use execution timeframe features)
            execution_features = features_dict[execution_tf]
            predictions = self.get_ml_predictions(symbol, execution_features)
            
            # 5. Run backtest - USE RL AGENT IF AVAILABLE!
            if self.rl_agent is not None:
                logger.info(f" Using trained RL agent for backtest")
                
                # ✅ FIXED: Pass multi-timeframe data
                results = self._run_backtest_with_rl_agent(
                    data_dict=filtered_data_dict,      # ← Multi-timeframe
                    features_dict=features_dict,        # ← Multi-timeframe  
                    predictions=predictions,
                    symbol=symbol
                )
            else:
                logger.warning(" RL agent not loaded, falling back to ML-only backtest")
                # Fallback still uses single timeframe
                results = self._run_backtest(
                    test_data, 
                    execution_features, 
                    predictions, 
                    symbol
                )
            
            return results
            
        except Exception as e:
            logger.error(f" Error processing {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'symbol': symbol, 'error': str(e)}
    
    def _run_backtest(self, 
                     data: pd.DataFrame, 
                     features: pd.DataFrame, 
                     predictions: np.ndarray,
                     symbol: str) -> Dict:
        """
        Run backtest with ML predictions only (fallback method)
        """
        # Initialize
        balance = self.config['initial_balance']
        initial_balance = self.config['initial_balance']
        max_position_value = initial_balance * 0.10
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Align data
        common_index = data.index.intersection(features.index)
        data = data.loc[common_index]
        features = features.loc[common_index]
        predictions = predictions[:len(common_index)]
        
        logger.info(f"Backtesting on {len(data)} periods...")
        
        for i in range(len(data)):
            timestamp = data.index[i]
            current_price = data.iloc[i]['open']
            
            if i < len(predictions):
                pred_down = predictions[i, 0]
                pred_flat = predictions[i, 1]
                pred_up = predictions[i, 2]
            else:
                pred_down, pred_flat, pred_up = 0.33, 0.34, 0.33
            
            # Calculate equity
            if position > 0:
                equity = balance + (position * current_price)
            else:
                equity = balance
            equity_curve.append({'timestamp': timestamp, 'equity': equity})
            
            # Trading logic
            if position == 0 and pred_up > 0.6:  # Buy signal
                reference_capital = min(balance, initial_balance * 2.0)
                max_position_value = reference_capital * 0.95
                position = max_position_value / current_price
                cost = position * current_price * (1 + self.config['fee'])
                balance -= cost
                entry_price = current_price
                trades.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'size': position,
                    'balance': balance
                })
                
            elif position > 0 and pred_down > 0.6:  # Sell signal
                balance += position * current_price * (1 - self.config['fee'])
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': current_price,
                    'size': position,
                    'balance': balance,
                    'pnl': (current_price - entry_price) * position
                })
                position = 0
        
        # Close any open position
        if position > 0:
            final_price = data.iloc[-1]['open']
            balance += position * final_price * (1 - self.config['fee'])
            trades.append({
                'timestamp': data.index[-1],
                'type': 'SELL',
                'price': final_price,
                'size': position,
                'balance': balance,
                'pnl': (final_price - entry_price) * position
            })
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        trades_df = pd.DataFrame(trades)
        
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (balance - self.config['initial_balance']) / self.config['initial_balance']
        
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            total_closed = len(trades_df[trades_df['pnl'].notna()])
            win_rate = winning_trades / total_closed if total_closed > 0 else 0
        else:
            win_rate = 0
        
        results = {
            'symbol': symbol,
            'initial_balance': self.config['initial_balance'],
            'final_balance': balance,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades_df[trades_df['pnl'].notna()]) if len(trades_df) > 0 else 0,
            'win_rate': win_rate,
            'equity_curve': equity_df,
            'trades': trades_df
        }
        
        return results
    
    def _run_backtest_with_rl_agent(self, 
                                    data_dict: Dict[str, pd.DataFrame],
                                    features_dict: Dict[str, pd.DataFrame],
                                    predictions: np.ndarray,
                                    symbol: str) -> Dict:
        """
        Run backtest using the ACTUAL trained RL agent WITH EXPLAINABILITY
        
        ⭐ MODIFIED: Now tracks trade duration
        """
        # Initialize state
        balance = self.config['initial_balance']
        initial_balance = self.config['initial_balance']
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Get execution timeframe
        execution_tf = self.config.get('execution_timeframe', '5m')
        execution_data = data_dict[execution_tf]
        execution_features = features_dict[execution_tf]
        
        # Align indices
        common_idx = execution_data.index.intersection(execution_features.index)
        
        # Filter all timeframes to common indices
        aligned_data_dict = {}
        aligned_features_dict = {}
        
        for tf in data_dict.keys():
            aligned_data_dict[tf] = data_dict[tf].loc[data_dict[tf].index.isin(common_idx)]
            aligned_features_dict[tf] = features_dict[tf].loc[features_dict[tf].index.isin(common_idx)]
        
        # Use execution timeframe for main loop
        execution_data = aligned_data_dict[execution_tf]
        execution_features = aligned_features_dict[execution_tf]
        predictions = predictions[:len(execution_data)]

        logger.info(f"   Backtesting {symbol} using RL agent...")
        logger.info(f"   Data points ({execution_tf}): {len(execution_data)}")
        logger.info(f"   Features ({execution_tf}): {execution_features.shape[1]}")
        logger.info(f"   Total timeframes: {len(data_dict)}")
        
        # Set agent to evaluation mode
        self.rl_agent.epsilon = 0.0
        
        # Setup explainability if enabled
        if self.config.get('explain', False):
            logger.info("   Setting up explainability for backtest...")
            state_feature_names = self._get_feature_names(execution_features)
            self.explainer = ExplainableRL(
                agent=self.rl_agent,
                state_feature_names=state_feature_names,
                action_names=['Hold', 'Buy', 'Sell'],
                explain_frequency=self.config.get('explain_frequency', 100),
                verbose=self.config.get('verbose', False),
                save_dir=f"{self.config.get('explain_dir', 'logs/backtest_explanations')}/{symbol}"
            )
            logger.info(f"     Explainability enabled")
            logger.info(f"      Tracking {len(state_feature_names)} features")
        
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        realized_pnl = 0.0
        total_fees_paid = 0.0
        position_opened_step = 0
        position_opened_timestamp = None  # ✅ NEW: Track opening timestamp
    
        # Main trading loop
        for i in range(len(execution_data)):
            timestamp = execution_data.index[i]
            current_price = execution_data.iloc[i]['open']
            
            if i < len(predictions):
                pred_probs = predictions[i]
            else:
                pred_probs = np.array([0.33, 0.34, 0.33])
            
            # Calculate current equity
            if position > 0:
                equity = balance + (position * current_price)
            else:
                equity = balance
            equity_curve.append({'timestamp': timestamp, 'equity': equity})
            
            # Construct state
            state = self._construct_state(
                i=i,
                data_dict=aligned_data_dict,
                features_dict=aligned_features_dict,
                execution_tf=execution_tf,
                current_price=current_price,
                balance=balance,
                equity=equity,
                position=position,
                entry_price=entry_price,
                initial_balance=initial_balance,
                symbol=symbol,
                realized_pnl=realized_pnl,
                total_fees_paid=total_fees_paid,
                position_opened_step=position_opened_step
            )
            
            # Get action
            if self.explainer:
                context = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'price': current_price,
                    'position': position,
                    'position_type': 'LONG' if position > 0 else 'FLAT',
                    'balance': balance,
                    'equity': equity,
                    'ml_prediction': pred_probs,
                    'step': i
                }
                action, explanation = self.explainer.act_with_explanation(
                    state=state,
                    context=context,
                    training=False
                )
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rl_agent.device)
                with torch.no_grad():
                    q_values = self.rl_agent.q_network(state_tensor)
                    action = q_values.argmax().item()
            
            # Execute action
            if action == 1 and position == 0:  # BUY
                execution_price = current_price * (1 + self.config['slippage'])
                
                reference_capital = min(balance, initial_balance * 2.0)
                position = (reference_capital * 0.95) / execution_price
                cost = position * execution_price
                fee = cost * self.config['fee']

                if (cost + fee) > balance:
                    position = (balance * 0.95) / execution_price
                    cost = position * execution_price
                    fee = cost * self.config['fee']
                
                if (position * execution_price * (1 + self.config['fee'])) > balance:
                    position = (balance * 0.95) / execution_price
                    cost = position * execution_price
                    fee = cost * self.config['fee']
                
                total_fees_paid += fee
                balance -= (cost + fee)
                entry_price = execution_price
                position_opened_step = i  # ✅ Track step
                position_opened_timestamp = timestamp  # ✅ NEW: Track timestamp
                action_counts['BUY'] += 1
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': execution_price,
                    'size': position,
                    'balance': balance
                })
                
            elif action == 2 and position > 0:  # SELL
                execution_price = current_price * (1 - self.config['slippage'])
                
                proceeds = position * execution_price
                fee = proceeds * self.config['fee']
                total_fees_paid += fee
                net_proceeds = proceeds - fee
                pnl = (execution_price - entry_price) * position
                realized_pnl += pnl
                balance += net_proceeds
                action_counts['SELL'] += 1
                
                # ✅ NEW: Calculate duration
                duration_steps = i - position_opened_step
                duration_time = timestamp - position_opened_timestamp if position_opened_timestamp else pd.Timedelta(0)
                
                trades.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': execution_price,
                    'size': position,
                    'balance': balance,
                    'pnl': pnl,
                    'duration_steps': duration_steps,  # ✅ NEW: Duration in steps
                    'duration_seconds': duration_time.total_seconds(),  # ✅ NEW: Duration in seconds
                    'entry_timestamp': position_opened_timestamp,  # ✅ NEW: When position opened
                    'exit_timestamp': timestamp  # ✅ NEW: When position closed
                })
                
                position = 0
                entry_price = 0
                position_opened_step = 0
                position_opened_timestamp = None  # ✅ Reset
            
            else:
                action_counts['HOLD'] += 1
        
        # Close any remaining position
        if position > 0:
            final_price = execution_data.iloc[-1]['open']
            execution_price = final_price * (1 - self.config['slippage'])
            
            proceeds = position * execution_price
            fee = proceeds * self.config['fee']
            total_fees_paid += fee
            net_proceeds = proceeds - fee
            pnl = (execution_price - entry_price) * position
            realized_pnl += pnl
            balance += net_proceeds
            
            # ✅ NEW: Calculate duration for final trade
            final_timestamp = execution_data.index[-1]
            duration_steps = len(execution_data) - 1 - position_opened_step
            duration_time = final_timestamp - position_opened_timestamp if position_opened_timestamp else pd.Timedelta(0)
            
            trades.append({
                'timestamp': final_timestamp,
                'type': 'SELL',
                'price': execution_price,
                'size': position,
                'balance': balance,
                'pnl': pnl,
                'duration_steps': duration_steps,
                'duration_seconds': duration_time.total_seconds(),
                'entry_timestamp': position_opened_timestamp,
                'exit_timestamp': final_timestamp
            })
        
        # Generate explainability report if enabled
        if self.explainer:
            logger.info("\n   " + "="*60)
            logger.info(f"   GENERATING EXPLAINABILITY REPORT FOR {symbol}")
            logger.info("   " + "="*60)
            report = self.explainer.generate_final_report()
            print("\n" + report)
            self.explainer.save_decision_history()
            logger.info(f"     Saved decision history to {self.explainer.save_dir}")
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve).set_index('timestamp')
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Calculate performance metrics
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (balance - initial_balance) / initial_balance
        
        bars_per_year = 12 * 24 * 252  # 72,576 for 5m bars
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(bars_per_year) if len(returns) > 0 and returns.std() > 0 else 0
        
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            total_closed = len(trades_df[trades_df['pnl'].notna()])
            win_rate = winning_trades / total_closed if total_closed > 0 else 0
        else:
            win_rate = 0
        
        # ✅ NEW: Calculate duration statistics
        closed_trades = trades_df[trades_df['type'] == 'SELL'].copy()
        if len(closed_trades) > 0:
            avg_duration_steps = closed_trades['duration_steps'].mean()
            avg_duration_seconds = closed_trades['duration_seconds'].mean()
            
            # Convert to human-readable format based on timeframe
            if execution_tf == '5m':
                avg_duration_minutes = avg_duration_steps * 5
                duration_display = f"{avg_duration_minutes:.1f} minutes"
            elif execution_tf == '15m':
                avg_duration_minutes = avg_duration_steps * 15
                duration_display = f"{avg_duration_minutes/60:.1f} hours"
            elif execution_tf == '1h':
                avg_duration_hours = avg_duration_steps * 1
                duration_display = f"{avg_duration_hours:.1f} hours"
            elif execution_tf == '4h':
                avg_duration_hours = avg_duration_steps * 4
                duration_display = f"{avg_duration_hours:.1f} hours"
            else:
                duration_display = f"{avg_duration_steps:.1f} bars"
            
            median_duration_steps = closed_trades['duration_steps'].median()
            min_duration_steps = closed_trades['duration_steps'].min()
            max_duration_steps = closed_trades['duration_steps'].max()
        else:
            avg_duration_steps = 0
            avg_duration_seconds = 0
            duration_display = "N/A"
            median_duration_steps = 0
            min_duration_steps = 0
            max_duration_steps = 0
        
        # Log action distribution
        total_actions = sum(action_counts.values())
        logger.info(f"   Action Distribution:")
        logger.info(f"     HOLD:  {action_counts['HOLD']:5d} ({action_counts['HOLD']/total_actions*100:5.1f}%)")
        logger.info(f"     BUY:   {action_counts['BUY']:5d} ({action_counts['BUY']/total_actions*100:5.1f}%)")
        logger.info(f"     SELL:  {action_counts['SELL']:5d} ({action_counts['SELL']/total_actions*100:5.1f}%)")
        
        # ✅ NEW: Log duration statistics
        if len(closed_trades) > 0:
            logger.info(f"\n   Trade Duration Statistics:")
            logger.info(f"     Average: {duration_display}")
            logger.info(f"     Median:  {median_duration_steps:.0f} bars")
            logger.info(f"     Min:     {min_duration_steps:.0f} bars")
            logger.info(f"     Max:     {max_duration_steps:.0f} bars")
        
        results = {
            'symbol': symbol,
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'num_trades': len(trades_df[trades_df['pnl'].notna()]) if len(trades_df) > 0 else 0,
            'win_rate': win_rate,
            'equity_curve': equity_df,
            'trades': trades_df,
            'action_distribution': action_counts,
            # ✅ NEW: Duration metrics
            'avg_duration_steps': avg_duration_steps,
            'avg_duration_display': duration_display,
            'median_duration_steps': median_duration_steps,
            'min_duration_steps': min_duration_steps,
            'max_duration_steps': max_duration_steps
        }
        
        logger.info(f"   Results:")
        logger.info(f"     Total Return: {total_return*100:.2f}%")
        logger.info(f"     Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"     Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"     Trades: {len(trades_df)}")
        logger.info(f"     Win Rate: {win_rate*100:.1f}%")
        logger.info(f"     Avg Duration: {duration_display}")  # ✅ NEW
        
        return results
    
    def _get_feature_names(self, features: pd.DataFrame) -> List[str]:
        """
        Get feature names for explainability
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of 81 feature names
        """
        feature_names = []
        
        # Market features (50)
        if hasattr(self.ml_model, 'selected_features') and self.ml_model.selected_features:
            market_features = self.ml_model.selected_features[:50]
            if len(market_features) < 50:
                market_features = list(market_features) + [f'market_{i}' for i in range(len(market_features), 50)]
            feature_names.extend(market_features)
        else:
            feature_names.extend([f'market_feature_{i}' for i in range(50)])
        
        # Technical features (10)
        feature_names.extend([
            'recent_open_norm', 'recent_high_norm', 'recent_low_norm', 'recent_close_norm',
            'prev_close_norm', 'candle_body_ratio', 'candle_range_ratio',
            'volume_ratio', 'volume_ratio_capped', 'price_direction'
        ])
        
        # Account features (5)
        feature_names.extend([
            'balance_normalized', 'portfolio_value_normalized',
            'realized_pnl_normalized', 'unrealized_pnl_normalized', 'fees_normalized'
        ])
        
        # Position features (5)
        feature_names.extend([
            'position_type', 'entry_price_normalized', 'position_size_normalized',
            'position_pnl_normalized', 'position_duration_normalized'
        ])
        
        # Asset encoding (5)
        feature_names.extend(['asset_eth', 'asset_dot', 'asset_sol', 'asset_ada', 'asset_avax'])
        
        # Timeframe encoding (6)
        feature_names.extend(['tf_1m', 'tf_5m', 'tf_15m', 'tf_1h', 'tf_4h', 'tf_1d'])
        
        return feature_names
    
    def _construct_state(self,
                    i: int,
                    data_dict: Dict[str, pd.DataFrame],
                    features_dict: Dict[str, pd.DataFrame],
                    execution_tf: str,
                    current_price: float,
                    balance: float,
                    equity: float,
                    position: float,
                    entry_price: float,
                    initial_balance: float,
                    symbol: str,
                    realized_pnl: float = 0.0,
                    total_fees_paid: float = 0.0,
                    position_opened_step: int = 0) -> np.ndarray:
        """
        Construct 183-dimensional state vector matching training environment
        
        ✅ FIXED: All issues corrected
        - Look-ahead bias removed (uses previous bar's OHLC, normalized by current open)
        - Correct asset encoding (ETH=0, SOL=1, DOT=2, AVAX=3, ADA=4)
        - Proper account features (realized_pnl, total_fees_paid)
        - Correct position duration tracking
        
        Args:
            i: Current timestep index
            data_dict: Multi-timeframe OHLCV data
            features_dict: Multi-timeframe features
            execution_tf: Execution timeframe
            current_price: Current bar's OPEN price
            balance: Current cash balance
            equity: Total portfolio value
            position: Current position size (in coins)
            entry_price: Entry price if in position
            initial_balance: Starting capital
            symbol: Trading symbol (e.g., 'ETH_USDT')
            realized_pnl: Cumulative realized PnL
            total_fees_paid: Cumulative fees paid
            position_opened_step: Step when current position was opened
            
        Returns:
            183-dimensional state vector
        """

        timeframes = self.config.get('timeframes', ['5m', '15m', '1h'])
        execution_data = data_dict[execution_tf]
        
        # Get current timestamp from execution timeframe
        current_timestamp = execution_data.index[i]
        
        # ========================================================================
        # PART 1: Features from each timeframe (150 dims = 3 × 50)
        # ========================================================================
        all_timeframe_features = []
        
        for tf in timeframes:
            tf_data = data_dict[tf]
            tf_features = features_dict[tf]
            
            # Find the most recent completed bar for this timeframe
            available_data = tf_data.loc[tf_data.index < current_timestamp]
            
            if len(available_data) > 0:
                last_timestamp = available_data.index[-1]
                if last_timestamp in tf_features.index:
                    tf_feat = tf_features.loc[last_timestamp].values[:50]
                else:
                    tf_feat = np.zeros(50)
            else:
                tf_feat = np.zeros(50)
            
            # Ensure exactly 50 features
            if len(tf_feat) < 50:
                padding = np.zeros(50 - len(tf_feat))
                tf_feat = np.concatenate([tf_feat, padding])
            elif len(tf_feat) > 50:
                tf_feat = tf_feat[:50]
            
            all_timeframe_features.append(tf_feat)
        
        market_features = np.concatenate(all_timeframe_features)  # 150 dims
        
        # ========================================================================
        # PART 2: Position features (5 dims)
        # ========================================================================
        if position > 0:
            position_duration = i - position_opened_step
            unrealized_pnl = (current_price - entry_price) * position
        else:
            position_duration = 0
            unrealized_pnl = 0.0
        
        position_info = np.array([
            float(1.0 if position > 0 else 0.0),
            entry_price / current_price if entry_price > 0 else 0.0,
            (position * current_price / initial_balance) if position > 0 else 0.0,
            unrealized_pnl / initial_balance if position > 0 else 0.0,
            position_duration / 100
        ], dtype=np.float32)
        
        # ========================================================================
        # PART 3: Account features (5 dims)
        # ========================================================================
        account_info = np.array([
            balance / initial_balance,
            equity / initial_balance,
            realized_pnl / initial_balance,
            unrealized_pnl / initial_balance,
            total_fees_paid / initial_balance
        ], dtype=np.float32)
        
        # ========================================================================
        # PART 4: Asset encoding (5 dims)
        # ========================================================================
        asset_map = {
            'ETH_USDT': 0, 'ETH/USDT': 0, 'ETH/USD': 0, 'ETHUSD': 0,
            'SOL_USDT': 1, 'SOL/USDT': 1, 'SOL/USD': 1, 'SOLUSD': 1,
            'DOT_USDT': 2, 'DOT/USDT': 2, 'DOT/USD': 2, 'DOTUSD': 2,
            'AVAX_USDT': 3, 'AVAX/USDT': 3, 'AVAX/USD': 3, 'AVAXUSD': 3,
            'ADA_USDT': 4, 'ADA/USDT': 4, 'ADA/USD': 4, 'ADAUSD': 4
        }
        
        asset_encoding = np.zeros(5, dtype=np.float32)
        if symbol in asset_map:
            asset_encoding[asset_map[symbol]] = 1.0
        
        # ========================================================================
        # PART 5: Timeframe encodings (18 dims = 3 × 6)
        # ========================================================================
        timeframe_map = {
            '1m': 0, '5m': 1, '15m': 2, '30m': 3, '1h': 4, '4h': 5
        }
        
        all_tf_encodings = []
        for tf in timeframes:
            encoding = np.zeros(6, dtype=np.float32)
            if tf in timeframe_map:
                encoding[timeframe_map[tf]] = 1.0
            all_tf_encodings.append(encoding)
        
        timeframe_encodings = np.concatenate(all_tf_encodings)  # 18 dims
        
        # ========================================================================
        # COMBINE ALL PARTS
        # ========================================================================
        state = np.concatenate([
            market_features,      # 150 dimensions (3 × 50)
            position_info,        # 5 dimensions
            account_info,         # 5 dimensions
            asset_encoding,       # 5 dimensions
            timeframe_encodings   # 18 dimensions (3 × 6)
        ])  # Total: 183 dimensions
        
        # Clean and validate
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        state = np.clip(state, -10, 10)
        
        assert len(state) == 183, f"State must be 183 dims, got {len(state)}"
        
        return state.astype(np.float32)
    
    def run_all_symbols(self) -> Dict:
        """
        Run walk-forward validation on all symbols
        
        Returns:
            Dictionary with results for all symbols
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING WALK-FORWARD VALIDATION")
        if self.config.get('explain', False):
            logger.info(" WITH EXPLAINABILITY")
        logger.info("="*80)
        
        # Load models first
        self.load_models()
        
        # Run for each symbol
        all_results = {}
        summary = []
        
        for symbol in self.config['symbols']:
            try:
                results = self.run_backtest_for_symbol(symbol)
                
                # Skip if error
                if 'error' in results:
                    logger.error(f" Skipping {symbol} due to error: {results['error']}")
                    continue
                    
                all_results[symbol] = results
                
                # ✅ MODIFIED: Add duration to summary
                summary.append({
                    'Symbol': symbol,
                    'Return %': f"{results['total_return_pct']:.2f}%",
                    'Sharpe': f"{results['sharpe_ratio']:.2f}",
                    'Max DD %': f"{results['max_drawdown_pct']:.2f}%",
                    'Trades': results['num_trades'],
                    'Win Rate': f"{results['win_rate']*100:.1f}%",
                    'Avg Duration': results.get('avg_duration_display', 'N/A')  # ✅ NEW
                })
                
            except Exception as e:
                logger.error(f" Error processing {symbol}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("="*80)
        
        if summary:
            summary_df = pd.DataFrame(summary)
            logger.info("\n" + summary_df.to_string(index=False))
            
            # Calculate overall stats
            total_returns = [r['total_return'] for r in all_results.values()]
            avg_return = np.mean(total_returns) * 100
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values()])
            
            logger.info(f"\n{'='*80}")
            logger.info(f"OVERALL PERFORMANCE")
            logger.info(f"{'='*80}")
            logger.info(f"Average Return: {avg_return:.2f}%")
            logger.info(f"Average Sharpe: {avg_sharpe:.2f}")
            logger.info(f"Successful on: {len([r for r in total_returns if r > 0])}/{len(total_returns)} assets")
            
            if self.config.get('explain', False):
                logger.info(f"\n Explainability reports saved to: {self.config.get('explain_dir', 'logs/backtest_explanations')}")
        else:
            logger.error(" No successful results!")
            summary_df = pd.DataFrame()
            avg_return = 0
            avg_sharpe = 0
        
        return {
            'individual_results': all_results,
            'summary': summary_df,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe
        }


def run_walk_forward_validation(
                    explain: bool = False,
                    explain_frequency: int = 100,
                    verbose: bool = False,
                    explain_dir: str = 'logs/backtest_explanations'
                ) -> Dict:
    """
    Main function to run walk-forward validation WITH EXPLAINABILITY
    
    Args:
        explain: Enable explainability
        explain_frequency: How often to print explanations
        verbose: Explain every decision
        explain_dir: Directory to save explanations
    """
    config = {
        'data_path': 'data/raw/',
        'models_path': 'models/',
        'symbols': ['SOL_USDT'],
        'timeframes': ['5m', '15m', '1h'],
        'execution_timeframe': '5m',
        'test_start_date': '2025-10-01',
        'test_end_date': '2025-11-20',
        'initial_balance': 10000,
        'fee': 0.001,  # ✅ FIXED: Match training
        'slippage': 0.0005,  # ✅ FIXED: Match training
        # Explainability
        'explain': explain,
        'explain_frequency': explain_frequency,
        'verbose': verbose,
        'explain_dir': explain_dir
    }
    
    runner = WalkForwardRunner(config)
    results = runner.run_all_symbols()
    
    return results


if __name__ == '__main__':
    # Example usage:
    # Without explainability (standard)
    # results = run_walk_forward_validation()
    
    # With explainability (to understand agent decisions)
    results = run_walk_forward_validation(
        explain=True,
        explain_frequency=100,
        verbose=False
    )