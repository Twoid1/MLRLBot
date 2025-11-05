"""
Walk-Forward Validation Runner
Complete pipeline for running walk-forward validation with proper asset identification
FIXED VERSION: Multi-timeframe support + proper numpy array handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib
import torch
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardRunner:
    """
    Complete walk-forward validation runner
    Handles data loading, feature engineering with symbols, model loading, and backtesting
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
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_path': 'data/raw/',  #  FIXED: Base path for all timeframes
            'models_path': 'models/',
            'symbols': ['ETH_USDT', 'DOT_USDT', 'SOL_USDT', 'ADA_USDT', 'AVAX_USDT'],
            'timeframes': ['1h', '4h', '1d'],  #  FIXED: Multiple timeframes
            'test_start_date': '2025-01-01',
            'test_end_date': '2025-10-26',
            'initial_balance': 10000,
            'fee': 0.001,
            'slippage': 0.0005,
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
         FIXED: Load data for multiple timeframes
        
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
         FIXED: Calculate features with multi-timeframe data
        
        Args:
            data_dict: Dictionary of {timeframe: DataFrame}
            symbol: Symbol name (e.g., 'ETH_USDT')
            
        Returns:
            DataFrame with features
        """
        logger.info(f"Calculating features for {symbol}...")
        
        # Convert symbol format (ETH_USDT -> ETH/USDT)
        symbol_formatted = symbol.replace('_', '/')
        
        # Get main timeframe data
        main_df = data_dict.get('1h')
        if main_df is None:
            raise ValueError(f"No 1h data found for {symbol}")
        
        #  FIXED: Calculate features with multi-timeframe support
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
         FIXED: Get ML model predictions (returns numpy array)
        
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
            return predictions  #  Returns numpy array
            
        except Exception as e:
            logger.error(f" Error getting ML predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([[0.33, 0.34, 0.33]] * len(features))
    
    def run_backtest_for_symbol(self, symbol: str) -> Dict:
        """
         FIXED: Run backtest with multi-timeframe data
        
        Args:
            symbol: Symbol to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running walk-forward validation for {symbol}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. ✅ FIXED: Load multi-timeframe data
            data_dict = self.load_multi_timeframe_data(symbol)
            
            if '1h' not in data_dict:
                raise ValueError(f"No 1h data found for {symbol}")
            
            main_df = data_dict['1h']
            
            # 2. Filter to test period
            test_start = pd.to_datetime(self.config['test_start_date'])
            test_end = pd.to_datetime(self.config['test_end_date'])
            
            # ✅ FIXED: Filter all timeframes
            filtered_data = {}
            for tf, df in data_dict.items():
                filtered_data[tf] = df.loc[test_start:test_end]
            
            test_data = filtered_data['1h']
            
            logger.info(f"Test period: {test_start} to {test_end}")
            logger.info(f"Test data points: {len(test_data)}")
            
            # 3. ✅ FIXED: Calculate features with multi-timeframe data
            features = self.prepare_features(filtered_data, symbol)
            
            # 4. Get ML predictions (returns numpy array)
            predictions = self.get_ml_predictions(symbol, features)
            
            # 5. Run backtest
            results = self._run_backtest(test_data, features, predictions, symbol)
            
            return results
            
        except Exception as e:
            logger.error(f" Error processing {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'symbol': symbol, 'error': str(e)}
    
    def _run_backtest(self, 
                     data: pd.DataFrame, 
                     features: pd.DataFrame, 
                     predictions: np.ndarray,  # ✅ FIXED: Correct type
                     symbol: str) -> Dict:
        """
        ✅ FIXED: Run backtest with proper numpy array handling
        
        Args:
            data: Price data (1h timeframe)
            features: Features DataFrame
            predictions: ML predictions numpy array (n_samples, 3)
            symbol: Symbol name
            
        Returns:
            Results dictionary
        """
        # Initialize
        balance = self.config['initial_balance']
        initial_balance = self.config['initial_balance']
        max_position_value = initial_balance * 0.95
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # ✅ FIXED: Align only data and features (not predictions!)
        common_index = data.index.intersection(features.index)
        data = data.loc[common_index]
        features = features.loc[common_index]
        
        # ✅ FIXED: Trim predictions array to match
        predictions = predictions[:len(common_index)]
        
        logger.info(f"Backtesting on {len(data)} periods...")
        
        # ✅ FIXED: Use range instead of enumerate for cleaner indexing
        for i in range(len(data)):
            timestamp = data.index[i]
            current_price = data.iloc[i]['close']
            
            # ✅ FIXED: Access predictions as numpy array
            if i < len(predictions):
                pred_down = predictions[i, 0]   # Column 0 = down probability
                pred_flat = predictions[i, 1]   # Column 1 = flat probability
                pred_up = predictions[i, 2]     # Column 2 = up probability
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
            final_price = data.iloc[-1]['close']
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
        
        # Calculate returns
        returns = equity_df['equity'].pct_change().dropna()
        total_return = (balance - self.config['initial_balance']) / self.config['initial_balance']
        
        # Calculate Sharpe ratio (annualized for hourly data)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0
        
        # Calculate max drawdown
        cummax = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Win rate
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
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'equity_curve': equity_df,
            'trades': trades_df
        }
        
        return results
    
    def run_all_symbols(self) -> Dict:
        """
        Run walk-forward validation on all symbols
        
        Returns:
            Dictionary with results for all symbols
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING WALK-FORWARD VALIDATION")
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
                
                # Add to summary
                summary.append({
                    'Symbol': symbol,
                    'Return %': f"{results['total_return_pct']:.2f}%",
                    'Sharpe': f"{results['sharpe_ratio']:.2f}",
                    'Max DD %': f"{results['max_drawdown_pct']:.2f}%",
                    'Trades': results['num_trades'],
                    'Win Rate': f"{results['win_rate']*100:.1f}%"
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


def run_walk_forward_validation():
    """
    Main function to run walk-forward validation
    """
    config = {
        'data_path': 'data/raw/',  # ✅ FIXED: Base path
        'models_path': 'models/',
        'symbols': ['ETH_USDT', 'DOT_USDT', 'SOL_USDT', 'ADA_USDT', 'AVAX_USDT'],
        'timeframes': ['1h', '4h', '1d'],  # ✅ FIXED: Multiple timeframes
        'test_start_date': '2025-01-01',
        'test_end_date': '2025-10-26',
        'initial_balance': 10000,
        'fee': 0.001,
        'slippage': 0.0005,
    }
    
    runner = WalkForwardRunner(config)
    results = runner.run_all_symbols()
    
    return results


if __name__ == '__main__':
    run_walk_forward_validation()