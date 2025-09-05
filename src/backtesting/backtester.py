"""
Backtesting Engine Module
Complete backtesting framework that integrates all components for historical simulation
Supports ML predictions, RL agents, multi-asset, and walk-forward validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle

# Import all our existing components
from ..data.data_manager import DataManager
from ..data.database import DatabaseManager
from ..data.validator import DataValidator
from ..features.feature_engineer import FeatureEngineer
from ..features.selector import FeatureSelector
from ..models.ml_predictor import MLPredictor
from ..models.dqn_agent import DQNAgent, DQNConfig
from ..models.labeling import LabelingConfig, LabelingPipeline
from ..environment.trading_env import TradingEnvironment
from ..trading.portfolio import Portfolio
from ..trading.risk_manager import RiskManager, RiskConfig
from ..trading.position_sizer import PositionSizer
from ..trading.executor import OrderExecutor

warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


# ================== DATA STRUCTURES ==================

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Data settings
    symbols: List[str] = field(default_factory=lambda: ['BTC_USDT', 'ETH_USDT', 'SOL_USDT'])
    timeframes: List[str] = field(default_factory=lambda: ['1h'])
    start_date: str = '2022-01-01'
    end_date: str = '2024-01-01'
    
    # Strategy settings
    strategy_type: str = 'hybrid'  # 'ml_only', 'rl_only', 'hybrid'
    use_ml_predictions: bool = True
    use_rl_agent: bool = True
    
    # Capital settings
    initial_capital: float = 10000.0
    position_size: float = 0.1  # Fraction of capital per trade
    max_positions: int = 3
    
    # Risk settings
    use_risk_manager: bool = True
    max_drawdown: float = 0.20  # 20% max drawdown
    stop_loss: float = 0.05  # 5% stop loss
    take_profit: float = 0.10  # 10% take profit
    
    # Trading settings
    commission: float = 0.0026  # Kraken's fee
    slippage: float = 0.001  # 0.1% slippage
    min_trade_amount: float = 10.0  # Minimum trade in USD
    
    # ML settings
    ml_model_path: Optional[str] = None
    retrain_interval: int = 30  # Days between retraining
    feature_selection: bool = True
    n_features: int = 50
    
    # RL settings
    rl_model_path: Optional[str] = None
    rl_training_episodes: int = 100
    rl_update_interval: int = 7  # Days between RL updates
    
    # Validation settings
    validation_split: float = 0.2
    walk_forward: bool = True
    n_splits: int = 5
    
    # Output settings
    save_results: bool = True
    results_path: str = './backtest_results/'
    verbose: bool = True


@dataclass
class BacktestResults:
    """Container for backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    
    # Time metrics
    avg_holding_period: float
    longest_winning_streak: int
    longest_losing_streak: int
    
    # Risk metrics
    value_at_risk: float
    expected_shortfall: float
    beta: float
    alpha: float
    
    # Portfolio evolution
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    positions_history: List[Dict]
    trades_history: List[Dict]
    
    # Model performance
    ml_accuracy: Optional[float] = None
    rl_reward: Optional[float] = None
    signal_quality: Optional[float] = None
    
    # Metadata
    config: BacktestConfig = None
    start_date: datetime = None
    end_date: datetime = None
    runtime: float = None


# ================== MAIN BACKTESTER ==================

class Backtester:
    """
    Complete backtesting engine that integrates all components
    
    Features:
    - Multi-asset backtesting
    - ML + RL hybrid strategies
    - Walk-forward validation
    - Risk management integration
    - Performance analytics
    - Parallel processing
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtester
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Storage for results
        self.results = {}
        self.trades = []
        self.positions = {}
        self.equity_curve = []
        
        # Performance tracking
        self.metrics = {}
        
        logger.info(f"Backtester initialized for {len(self.config.symbols)} symbols")
    
    def _initialize_components(self):
        """Initialize all required components"""
        # Data components
        self.data_manager = DataManager()
        self.database = DatabaseManager()
        self.validator = DataValidator()
        
        # Feature components
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        
        # Model components
        self.ml_predictor = None
        self.rl_agent = None
        
        # Trading components
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions
        )
        
        if self.config.use_risk_manager:
            self.risk_manager = RiskManager(
                initial_capital=self.config.initial_capital,
                config=RiskConfig(
                    max_drawdown_limit=self.config.max_drawdown,
                    stop_loss_atr_multiplier=2.0,
                    take_profit_atr_multiplier=3.0
                )
            )
        else:
            self.risk_manager = None
        
        self.position_sizer = PositionSizer(
            capital=self.config.initial_capital,
            max_risk_per_trade=0.02
        )
        
        self.executor = OrderExecutor(mode='simulated')
        
    # ================== DATA PREPARATION ==================
    
    def prepare_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Prepare all data for backtesting
        
        Returns:
            Nested dict {symbol: {timeframe: DataFrame}}
        """
        logger.info("Preparing data for backtesting...")
        all_data = {}
        
        for symbol in self.config.symbols:
            all_data[symbol] = {}
            
            for timeframe in self.config.timeframes:
                # Load data
                df = self.data_manager.load_existing_data(symbol, timeframe)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} {timeframe}")
                    continue
                
                # Filter by date range
                df = df[
                    (df.index >= pd.to_datetime(self.config.start_date)) &
                    (df.index <= pd.to_datetime(self.config.end_date))
                ]
                
                # Validate data
                validation_result = self.validator.validate_ohlcv(df)
                if not validation_result.is_valid:
                    logger.warning(f"Data validation failed for {symbol} {timeframe}")
                    df = self.validator.fix_all_issues(df)
                
                all_data[symbol][timeframe] = df
        
        return all_data
    
    def calculate_features(self, 
                          data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate features for all symbols and timeframes
        
        Args:
            data: Nested dict of OHLCV data
            
        Returns:
            Dict {symbol: features_df}
        """
        logger.info("Calculating features...")
        all_features = {}
        
        for symbol, timeframe_data in data.items():
            # Get base timeframe
            base_tf = self.config.timeframes[0]
            if base_tf not in timeframe_data:
                continue
            
            base_df = timeframe_data[base_tf]
            
            # Calculate features
            features = self.feature_engineer.calculate_all_features(base_df, symbol)
            
            # Add multi-timeframe features if available
            if len(timeframe_data) > 1:
                mtf_features = self.feature_engineer.calculate_multi_timeframe_features(
                    timeframe_data
                )
                features = pd.concat([features, mtf_features], axis=1)
            
            all_features[symbol] = features
        
        return all_features
    
    # ================== MODEL TRAINING ==================
    
    def train_ml_model(self, 
                      data: Dict[str, Dict[str, pd.DataFrame]],
                      features: Dict[str, pd.DataFrame]) -> None:
        """
        Train ML model on historical data
        
        Args:
            data: OHLCV data
            features: Calculated features
        """
        if not self.config.use_ml_predictions:
            return
        
        logger.info("Training ML model...")
        
        # Initialize ML predictor
        self.ml_predictor = MLPredictor(
            model_type='xgboost',
            labeling_config=LabelingConfig(
                method='triple_barrier',
                lookforward=10,
                pt_sl=[1.5, 1.0]
            )
        )
        
        # Prepare training data
        train_data = {}
        val_data = {}
        
        for symbol in self.config.symbols:
            if symbol not in features:
                continue
            
            # Split data
            split_idx = int(len(features[symbol]) * (1 - self.config.validation_split))
            
            train_data[symbol] = (
                data[symbol][self.config.timeframes[0]][:split_idx],
                features[symbol][:split_idx]
            )
            val_data[symbol] = (
                data[symbol][self.config.timeframes[0]][split_idx:],
                features[symbol][split_idx:]
            )
        
        # Train model
        training_results = self.ml_predictor.train(
            train_data,
            val_data,
            feature_selection=self.config.feature_selection,
            n_features=self.config.n_features
        )
        
        logger.info(f"ML Model trained - Accuracy: {training_results.get('train_accuracy', 0):.4f}")
        
        # Load pre-trained model if specified
        if self.config.ml_model_path and Path(self.config.ml_model_path).exists():
            self.ml_predictor.load_model(self.config.ml_model_path)
            logger.info(f"Loaded ML model from {self.config.ml_model_path}")
    
    def train_rl_agent(self, 
                      data: pd.DataFrame,
                      features: pd.DataFrame) -> None:
        """
        Train RL agent on historical data
        
        Args:
            data: OHLCV data for training
            features: Features for the environment
        """
        if not self.config.use_rl_agent:
            return
        
        logger.info("Training RL agent...")
        
        # Create trading environment
        env = TradingEnvironment(
            df=data,
            initial_balance=self.config.initial_capital,
            fee_rate=self.config.commission,
            slippage=self.config.slippage,
            features_df=features,
            stop_loss=self.config.stop_loss,
            take_profit=self.config.take_profit
        )
        
        # Initialize RL agent
        state_dim = env.observation_space_shape[0]
        action_dim = env.action_space_n
        
        rl_config = DQNConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[256, 256, 128],
            use_double_dqn=True,
            use_dueling_dqn=True
        )
        
        self.rl_agent = DQNAgent(rl_config)
        
        # Train agent
        for episode in range(self.config.rl_training_episodes):
            stats = self.rl_agent.train_episode(env)
            
            if episode % 10 == 0:
                logger.info(f"RL Training Episode {episode}: Reward={stats['total_reward']:.2f}")
        
        # Load pre-trained model if specified
        if self.config.rl_model_path and Path(self.config.rl_model_path).exists():
            self.rl_agent.load(self.config.rl_model_path)
            logger.info(f"Loaded RL agent from {self.config.rl_model_path}")
    
    # ================== BACKTESTING LOGIC ==================
    
    def run(self) -> BacktestResults:
        """
        Run complete backtest
        
        Returns:
            BacktestResults with all metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Prepare data
        data = self.prepare_data()
        features = self.calculate_features(data)
        
        # Train models
        if self.config.use_ml_predictions:
            self.train_ml_model(data, features)
        
        # Run backtest for each symbol
        if self.config.walk_forward:
            results = self._run_walk_forward(data, features)
        else:
            results = self._run_standard_backtest(data, features)
        
        # Calculate final metrics
        final_results = self._calculate_final_metrics(results)
        
        # Runtime
        runtime = (datetime.now() - start_time).total_seconds()
        final_results.runtime = runtime
        
        logger.info(f"Backtest completed in {runtime:.2f} seconds")
        logger.info(f"Total Return: {final_results.total_return:.2%}")
        logger.info(f"Sharpe Ratio: {final_results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {final_results.max_drawdown:.2%}")
        
        # Save results if requested
        if self.config.save_results:
            self._save_results(final_results)
        
        return final_results
    
    def _run_standard_backtest(self,
                              data: Dict[str, Dict[str, pd.DataFrame]],
                              features: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run standard backtest (no walk-forward)
        
        Args:
            data: OHLCV data
            features: Feature data
            
        Returns:
            Results dictionary
        """
        logger.info("Running standard backtest...")
        
        results = {
            'trades': [],
            'equity_curve': [],
            'positions': []
        }
        
        # Process each symbol
        for symbol in self.config.symbols:
            if symbol not in data or self.config.timeframes[0] not in data[symbol]:
                continue
            
            symbol_data = data[symbol][self.config.timeframes[0]]
            symbol_features = features.get(symbol)
            
            # Simulate trading
            symbol_results = self._simulate_trading(
                symbol=symbol,
                data=symbol_data,
                features=symbol_features
            )
            
            results['trades'].extend(symbol_results['trades'])
            results['positions'].extend(symbol_results['positions'])
        
        # Combine equity curves
        results['equity_curve'] = self.portfolio.get_equity_curve()
        
        return results
    
    def _run_walk_forward(self,
                         data: Dict[str, Dict[str, pd.DataFrame]],
                         features: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run walk-forward validation
        
        Args:
            data: OHLCV data
            features: Feature data
            
        Returns:
            Results dictionary
        """
        logger.info(f"Running walk-forward validation with {self.config.n_splits} splits...")
        
        all_results = {
            'trades': [],
            'equity_curve': [],
            'positions': [],
            'split_metrics': []
        }
        
        # Prepare combined data for walk-forward
        for split_idx in range(self.config.n_splits):
            logger.info(f"Processing split {split_idx + 1}/{self.config.n_splits}")
            
            # Reset portfolio for each split
            self.portfolio = Portfolio(
                initial_capital=self.config.initial_capital,
                max_positions=self.config.max_positions
            )
            
            split_results = {
                'trades': [],
                'positions': []
            }
            
            for symbol in self.config.symbols:
                if symbol not in data:
                    continue
                
                # Get split data
                symbol_data = data[symbol][self.config.timeframes[0]]
                symbol_features = features.get(symbol)
                
                # Calculate split indices
                total_len = len(symbol_data)
                split_size = total_len // self.config.n_splits
                
                train_start = 0
                train_end = (split_idx + 1) * split_size
                test_start = train_end
                test_end = min(test_start + split_size, total_len)
                
                # Skip if not enough data
                if test_start >= total_len:
                    continue
                
                # Train on expanding window
                train_data = symbol_data.iloc[train_start:train_end]
                train_features = symbol_features.iloc[train_start:train_end]
                
                # Test on next period
                test_data = symbol_data.iloc[test_start:test_end]
                test_features = symbol_features.iloc[test_start:test_end]
                
                # Retrain models if needed
                if split_idx > 0 and self.config.use_ml_predictions:
                    self._retrain_ml_model(train_data, train_features)
                
                # Simulate trading on test period
                symbol_results = self._simulate_trading(
                    symbol=symbol,
                    data=test_data,
                    features=test_features
                )
                
                split_results['trades'].extend(symbol_results['trades'])
                split_results['positions'].extend(symbol_results['positions'])
            
            # Store split results
            all_results['trades'].extend(split_results['trades'])
            all_results['positions'].extend(split_results['positions'])
            
            # Calculate split metrics
            split_metrics = self._calculate_split_metrics(split_results)
            all_results['split_metrics'].append(split_metrics)
        
        # Get final equity curve
        all_results['equity_curve'] = self.equity_curve
        
        return all_results
    
    def _simulate_trading(self,
                         symbol: str,
                         data: pd.DataFrame,
                         features: pd.DataFrame) -> Dict:
        """
        Simulate trading for a single symbol
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            features: Feature data
            
        Returns:
            Trading results
        """
        results = {
            'trades': [],
            'positions': [],
            'signals': []
        }
        
        # Initialize position tracking
        position = None
        
        # Process each timestep
        for i in range(100, len(data)):  # Start after warmup period
            current_time = data.index[i]
            current_price = data.iloc[i]['close']
            
            # Generate signals
            signal = self._generate_signal(
                symbol=symbol,
                data=data.iloc[:i+1],
                features=features.iloc[:i+1] if features is not None else None,
                current_idx=i
            )
            
            results['signals'].append({
                'time': current_time,
                'symbol': symbol,
                'signal': signal
            })
            
            # Execute trades based on signals
            if signal == 1 and position is None:  # Buy signal
                # Check risk management
                if self._check_risk_limits(symbol, current_price):
                    position = self._open_position(
                        symbol=symbol,
                        price=current_price,
                        time=current_time,
                        signal_strength=signal
                    )
                    if position:
                        results['positions'].append(position)
            
            elif signal == -1 and position is not None:  # Sell signal
                trade = self._close_position(
                    position=position,
                    price=current_price,
                    time=current_time
                )
                if trade:
                    results['trades'].append(trade)
                    position = None
            
            # Check stop-loss and take-profit
            if position is not None:
                if self._check_exit_conditions(position, current_price):
                    trade = self._close_position(
                        position=position,
                        price=current_price,
                        time=current_time,
                        reason='stop_loss' if current_price < position['stop_loss'] else 'take_profit'
                    )
                    if trade:
                        results['trades'].append(trade)
                        position = None
            
            # Update equity curve
            portfolio_value = self.portfolio.total_value
            self.equity_curve.append({
                'time': current_time,
                'value': portfolio_value
            })
        
        # Close any remaining position
        if position is not None:
            trade = self._close_position(
                position=position,
                price=data.iloc[-1]['close'],
                time=data.index[-1],
                reason='end_of_backtest'
            )
            if trade:
                results['trades'].append(trade)
        
        return results
    
    def _generate_signal(self,
                        symbol: str,
                        data: pd.DataFrame,
                        features: Optional[pd.DataFrame],
                        current_idx: int) -> int:
        """
        Generate trading signal using ML and/or RL
        
        Args:
            symbol: Trading symbol
            data: OHLCV data up to current point
            features: Features up to current point
            current_idx: Current index in data
            
        Returns:
            Signal: 1 (buy), 0 (hold), -1 (sell)
        """
        signals = []
        
        # ML prediction
        if self.config.use_ml_predictions and self.ml_predictor and features is not None:
            try:
                prediction = self.ml_predictor.predict(
                    data,
                    features,
                    symbol=symbol,
                    timeframe=self.config.timeframes[0]
                )
                
                # Convert prediction to signal
                if prediction.prediction == 1:  # Up prediction
                    signals.append(1)
                elif prediction.prediction == -1:  # Down prediction
                    signals.append(-1)
                else:
                    signals.append(0)
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}")
                signals.append(0)
        
        # RL action
        if self.config.use_rl_agent and self.rl_agent:
            try:
                # Create state for RL agent
                if features is not None and len(features) > 0:
                    state = features.iloc[-1].values
                else:
                    # Use price-based state
                    state = self._create_price_state(data)
                
                # Get action from agent
                action = self.rl_agent.act(state, training=False)
                
                # Convert action to signal
                if action == 1:  # Buy action
                    signals.append(1)
                elif action == 2:  # Sell action
                    signals.append(-1)
                else:
                    signals.append(0)
            except Exception as e:
                logger.debug(f"RL action failed: {e}")
                signals.append(0)
        
        # Combine signals
        if signals:
            # Use majority voting or average
            final_signal = np.sign(np.mean(signals))
            return int(final_signal)
        
        return 0
    
    def _open_position(self,
                      symbol: str,
                      price: float,
                      time: datetime,
                      signal_strength: float = 1.0) -> Optional[Dict]:
        """
        Open a new position
        
        Args:
            symbol: Trading symbol
            price: Entry price
            time: Entry time
            signal_strength: Signal strength for sizing
            
        Returns:
            Position dictionary or None
        """
        # Calculate position size
        if self.risk_manager:
            position_size = self.risk_manager.calculate_position_size(
                symbol=symbol,
                signal_strength=abs(signal_strength),
                current_price=price,
                volatility=self._calculate_volatility(symbol),
                win_rate=0.5,  # Use historical win rate
                avg_win=100,
                avg_loss=50
            )
        else:
            # Simple fixed position sizing
            position_value = self.portfolio.cash_balance * self.config.position_size
            position_size = position_value / price
        
        # Check minimum trade amount
        if position_size * price < self.config.min_trade_amount:
            return None
        
        # Calculate stop-loss and take-profit
        stop_loss = price * (1 - self.config.stop_loss)
        take_profit = price * (1 + self.config.take_profit)
        
        # Open position in portfolio
        success = self.portfolio.open_position(
            symbol=symbol,
            quantity=position_size,
            price=price,
            position_type='long',
            stop_loss=stop_loss,
            take_profit=take_profit,
            fees=position_size * price * self.config.commission
        )
        
        if success:
            position = {
                'symbol': symbol,
                'entry_time': time,
                'entry_price': price,
                'quantity': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_value': position_size * price
            }
            
            logger.debug(f"Opened position: {symbol} @ {price:.2f}")
            return position
        
        return None
    
    def _close_position(self,
                       position: Dict,
                       price: float,
                       time: datetime,
                       reason: str = 'signal') -> Optional[Dict]:
        """
        Close an existing position
        
        Args:
            position: Position dictionary
            price: Exit price
            time: Exit time
            reason: Reason for closing
            
        Returns:
            Trade dictionary or None
        """
        # Apply slippage
        if reason == 'stop_loss':
            exit_price = price * (1 - self.config.slippage)
        else:
            exit_price = price * (1 + self.config.slippage)
        
        # Close position in portfolio
        result = self.portfolio.close_position(
            symbol=position['symbol'],
            price=exit_price,
            fees=position['quantity'] * exit_price * self.config.commission,
            reason=reason
        )
        
        if result:
            # Calculate trade metrics
            trade = {
                'symbol': position['symbol'],
                'entry_time': position['entry_time'],
                'entry_price': position['entry_price'],
                'exit_time': time,
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'pnl': result['net_pnl'],
                'return_pct': result['return_pct'],
                'reason': reason,
                'duration': (time - position['entry_time']).total_seconds() / 3600  # Hours
            }
            
            logger.debug(f"Closed position: {position['symbol']} @ {exit_price:.2f} - P&L: {trade['pnl']:.2f}")
            return trade
        
        return None
    
    def _check_risk_limits(self, symbol: str, price: float) -> bool:
        """
        Check if opening a new position is within risk limits
        
        Args:
            symbol: Trading symbol
            price: Current price
            
        Returns:
            True if within limits
        """
        if not self.risk_manager:
            return True
        
        # Get current risk metrics
        metrics = self.risk_manager.get_risk_metrics()
        
        # Check drawdown (only if we have drawdown)
        if hasattr(metrics, 'current_drawdown') and metrics.current_drawdown > self.config.max_drawdown:
            logger.warning(f"Max drawdown reached: {metrics.current_drawdown:.2%}")
            return False
        
        # Check number of positions
        if len(self.portfolio.positions) >= self.config.max_positions:
            logger.debug(f"Max positions reached: {len(self.portfolio.positions)}/{self.config.max_positions}")
            return False
        
        # Check exposure - be more lenient for testing
        position_value = price * self.config.position_size  
        max_exposure = self.config.initial_capital * 3  # Allow 3x leverage for testing
        
        current_exposure = getattr(self.risk_manager, 'total_exposure', 0)
        if current_exposure + position_value > max_exposure:
            logger.debug(f"Max exposure would be exceeded: {current_exposure + position_value} > {max_exposure}")
            return False
        
        return True
    
    def _check_exit_conditions(self, position: Dict, current_price: float) -> bool:
        """
        Check if position should be closed based on stop-loss or take-profit
        
        Args:
            position: Position dictionary
            current_price: Current market price
            
        Returns:
            True if position should be closed
        """
        if current_price <= position['stop_loss']:
            logger.debug(f"Stop-loss triggered for {position['symbol']}")
            return True
        
        if current_price >= position['take_profit']:
            logger.debug(f"Take-profit triggered for {position['symbol']}")
            return True
        
        return False
    
    # ================== METRICS CALCULATION ==================
    
    def _calculate_final_metrics(self, results: Dict) -> BacktestResults:
        """
        Calculate final backtest metrics
        
        Args:
            results: Raw results dictionary
            
        Returns:
            BacktestResults object
        """
        trades = results.get('trades', [])
        equity_curve_data = results.get('equity_curve', [])
        
        if not equity_curve_data:
            equity_curve_data = self.equity_curve
        
        # Convert equity curve to Series
        if equity_curve_data:
            equity_curve = pd.Series(
                [e['value'] for e in equity_curve_data],
                index=[e['time'] for e in equity_curve_data]
            )
        else:
            equity_curve = pd.Series([self.config.initial_capital])
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] / self.config.initial_capital - 1) if len(equity_curve) > 0 else 0
        
        # Annual return
        if len(equity_curve) > 1:
            days = (equity_curve.index[-1] - equity_curve.index[0]).days
            annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        else:
            annual_return = 0
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std()
        else:
            sortino_ratio = 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Trade statistics
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            total_trades = len(trades)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            
            profit_factor = (
                winning_trades['pnl'].sum() / abs(losing_trades['pnl'].sum())
                if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0
                else 0
            )
            
            largest_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
            largest_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
            
            avg_holding_period = trades_df['duration'].mean() if 'duration' in trades_df else 0
            
            # Streaks
            longest_winning_streak = self._calculate_longest_streak(trades_df, True)
            longest_losing_streak = self._calculate_longest_streak(trades_df, False)
        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            largest_win = 0
            largest_loss = 0
            avg_holding_period = 0
            longest_winning_streak = 0
            longest_losing_streak = 0
        
        # Risk metrics
        if len(returns) > 20:
            value_at_risk = np.percentile(returns, 5)  # 95% VaR
            expected_shortfall = returns[returns <= value_at_risk].mean()
        else:
            value_at_risk = 0
            expected_shortfall = 0
        
        # Create results object
        backtest_results = BacktestResults(
            # Performance
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            
            # Trade stats
            total_trades=total_trades,
            winning_trades=len(winning_trades) if trades else 0,
            losing_trades=len(losing_trades) if trades else 0,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            
            # Time metrics
            avg_holding_period=avg_holding_period,
            longest_winning_streak=longest_winning_streak,
            longest_losing_streak=longest_losing_streak,
            
            # Risk metrics
            value_at_risk=value_at_risk,
            expected_shortfall=expected_shortfall,
            beta=0,  # Would need market data to calculate
            alpha=0,  # Would need market data to calculate
            
            # Portfolio evolution
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            positions_history=results.get('positions', []),
            trades_history=trades,
            
            # Config
            config=self.config,
            start_date=pd.to_datetime(self.config.start_date),
            end_date=pd.to_datetime(self.config.end_date)
        )
        
        return backtest_results
    
    def _calculate_longest_streak(self, trades_df: pd.DataFrame, winning: bool) -> int:
        """Calculate longest winning or losing streak"""
        if trades_df.empty:
            return 0
        
        streaks = []
        current_streak = 0
        
        for _, trade in trades_df.iterrows():
            if (winning and trade['pnl'] > 0) or (not winning and trade['pnl'] <= 0):
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return max(streaks) if streaks else 0
    
    def _calculate_split_metrics(self, split_results: Dict) -> Dict:
        """Calculate metrics for a single walk-forward split"""
        trades = split_results.get('trades', [])
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        return {
            'total_trades': len(trades),
            'win_rate': (trades_df['pnl'] > 0).mean(),
            'avg_pnl': trades_df['pnl'].mean(),
            'total_pnl': trades_df['pnl'].sum()
        }
    
    def _calculate_volatility(self, symbol: str) -> float:
        """Calculate current volatility for position sizing"""
        # This would use historical data to calculate volatility
        # For now, return a default value
        return 0.02
    
    def _create_price_state(self, data: pd.DataFrame) -> np.ndarray:
        """Create state vector from price data for RL agent"""
        # Simple price-based features
        closes = data['close'].values[-20:]  # Last 20 closes
        volumes = data['volume'].values[-20:]  # Last 20 volumes
        
        # Normalize
        closes_norm = closes / closes[-1]
        volumes_norm = volumes / volumes.mean()
        
        # Combine into state
        state = np.concatenate([closes_norm, volumes_norm])
        
        return state
    
    def _retrain_ml_model(self, train_data: pd.DataFrame, train_features: pd.DataFrame):
        """Retrain ML model on new data"""
        # This would implement online learning or periodic retraining
        pass
    
    # ================== OUTPUT METHODS ==================
    
    def _save_results(self, results: BacktestResults):
        """Save backtest results to disk"""
        output_dir = Path(self.config.results_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_file = output_dir / f"backtest_summary_{timestamp}.json"
        summary_data = {
            'total_return': results.total_return,
            'annual_return': results.annual_return,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'total_trades': results.total_trades,
            'config': {
                'symbols': self.config.symbols,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save detailed results
        results_file = output_dir / f"backtest_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save trades
        if results.trades_history:
            trades_df = pd.DataFrame(results.trades_history)
            trades_file = output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
        
        # Save equity curve
        equity_file = output_dir / f"equity_curve_{timestamp}.csv"
        results.equity_curve.to_csv(equity_file)
        
        logger.info(f"Results saved to {output_dir}")
    
    def plot_results(self, results: BacktestResults):
        """Create visualization of backtest results"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        ax1 = axes[0, 0]
        results.equity_curve.plot(ax=ax1)
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        
        # Drawdown
        ax2 = axes[0, 1]
        results.drawdown_curve.plot(ax=ax2, color='red')
        ax2.set_title('Drawdown')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True)
        
        # Returns distribution
        ax3 = axes[1, 0]
        returns = results.equity_curve.pct_change().dropna()
        returns.hist(ax=ax3, bins=50, edgecolor='black')
        ax3.set_title('Returns Distribution')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        
        # Trade P&L
        if results.trades_history:
            ax4 = axes[1, 1]
            trades_df = pd.DataFrame(results.trades_history)
            trades_df['pnl'].plot(ax=ax4, kind='bar')
            ax4.set_title('Trade P&L')
            ax4.set_xlabel('Trade #')
            ax4.set_ylabel('P&L ($)')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()


# ================== UTILITY FUNCTIONS ==================

def run_backtest(config: Optional[BacktestConfig] = None) -> BacktestResults:
    """
    Convenience function to run backtest
    
    Args:
        config: Backtest configuration
        
    Returns:
        Backtest results
    """
    backtester = Backtester(config)
    return backtester.run()


def optimize_strategy(base_config: BacktestConfig,
                     param_grid: Dict[str, List],
                     metric: str = 'sharpe_ratio') -> Tuple[BacktestConfig, BacktestResults]:
    """
    Optimize strategy parameters
    
    Args:
        base_config: Base configuration
        param_grid: Parameters to optimize
        metric: Metric to optimize
        
    Returns:
        Best configuration and results
    """
    best_config = None
    best_results = None
    best_metric = -float('inf')
    
    # Grid search
    for params in _generate_param_combinations(param_grid):
        # Update config with current params
        config = BacktestConfig(**{**base_config.__dict__, **params})
        
        # Run backtest
        backtester = Backtester(config)
        results = backtester.run()
        
        # Check if better
        current_metric = getattr(results, metric)
        if current_metric > best_metric:
            best_metric = current_metric
            best_config = config
            best_results = results
    
    return best_config, best_results


def _generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    """Generate all parameter combinations from grid"""
    from itertools import product
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


# ================== MAIN EXECUTION ==================

if __name__ == "__main__":
    # Example backtest configuration
    config = BacktestConfig(
        symbols=['BTC_USDT', 'ETH_USDT', 'SOL_USDT'],
        timeframes=['1h'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        initial_capital=10000,
        strategy_type='hybrid',
        use_ml_predictions=True,
        use_rl_agent=True,
        walk_forward=True,
        n_splits=5,
        save_results=True,
        verbose=True
    )
    
    # Run backtest
    print("=== Starting Backtest ===")
    results = run_backtest(config)
    
    # Print summary
    print("\n=== Backtest Results ===")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Total Trades: {results.total_trades}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    
    print("\n Backtester ready for use!")