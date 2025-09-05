"""
State Management Module
Handles state representation and feature extraction for the trading environment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import hashlib


@dataclass
class MarketState:
    """Market data state"""
    current_price: float
    high: float
    low: float
    volume: float
    price_history: np.ndarray
    volume_history: np.ndarray
    returns_history: np.ndarray
    volatility: float
    trend: float
    support_level: float
    resistance_level: float


@dataclass
class AccountState:
    """Account and position state"""
    balance: float
    equity: float
    position_size: float
    position_type: int  # -1: short, 0: flat, 1: long
    entry_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float


@dataclass
class TechnicalState:
    """Technical indicators state"""
    rsi: float
    macd: float
    macd_signal: float
    bb_position: float  # Position within Bollinger Bands
    volume_ratio: float  # Current volume vs average
    momentum: float
    support_distance: float  # Distance to support
    resistance_distance: float  # Distance to resistance


class StateManager:
    """
    Manages state representation for the trading environment
    """
    
    def __init__(self, 
                 window_size: int = 50,
                 feature_version: str = 'v1',
                 normalize: bool = True):
        """
        Initialize state manager
        
        Args:
            window_size: Lookback window for historical features
            feature_version: Version of feature extraction
            normalize: Whether to normalize features
        """
        self.window_size = window_size
        self.feature_version = feature_version
        self.normalize = normalize
        
        # Feature statistics for normalization
        self.feature_means = {}
        self.feature_stds = {}
        
        # State visit tracking for exploration
        self.state_visits = {}
        
    def extract_market_state(self, 
                            ohlcv_data: pd.DataFrame,
                            current_step: int) -> MarketState:
        """
        Extract market state from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            current_step: Current position in data
            
        Returns:
            MarketState object
        """
        # Get current values
        current_price = ohlcv_data.iloc[current_step]['close']
        current_high = ohlcv_data.iloc[current_step]['high']
        current_low = ohlcv_data.iloc[current_step]['low']
        current_volume = ohlcv_data.iloc[current_step]['volume']
        
        # Get historical window
        start_idx = max(0, current_step - self.window_size)
        window_data = ohlcv_data.iloc[start_idx:current_step + 1]
        
        # Price and volume history
        price_history = window_data['close'].values
        volume_history = window_data['volume'].values
        
        # Calculate returns
        returns = window_data['close'].pct_change().fillna(0).values
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Calculate trend (linear regression slope)
        if len(price_history) > 1:
            x = np.arange(len(price_history))
            z = np.polyfit(x, price_history, 1)
            trend = z[0] / current_price  # Normalize by current price
        else:
            trend = 0
        
        # Support and resistance levels
        support_level = window_data['low'].min()
        resistance_level = window_data['high'].max()
        
        return MarketState(
            current_price=current_price,
            high=current_high,
            low=current_low,
            volume=current_volume,
            price_history=price_history,
            volume_history=volume_history,
            returns_history=returns,
            volatility=volatility,
            trend=trend,
            support_level=support_level,
            resistance_level=resistance_level
        )
    
    def extract_account_state(self, env) -> AccountState:
        """
        Extract account state from environment
        
        Args:
            env: Trading environment instance
            
        Returns:
            AccountState object
        """
        return AccountState(
            balance=env.balance,
            equity=env.equity,
            position_size=env.position_size,
            position_type=int(env.position),
            entry_price=env.entry_price,
            unrealized_pnl=env.unrealized_pnl,
            realized_pnl=env.realized_pnl,
            total_trades=len(env.trades),
            win_rate=env._get_win_rate(),
            sharpe_ratio=env._get_sharpe_ratio(),
            max_drawdown=env.max_drawdown,
            current_drawdown=env.current_drawdown
        )
    
    def extract_technical_state(self,
                               ohlcv_data: pd.DataFrame,
                               indicators_df: pd.DataFrame,
                               current_step: int) -> TechnicalState:
        """
        Extract technical indicators state
        
        Args:
            ohlcv_data: DataFrame with OHLCV columns
            indicators_df: DataFrame with calculated indicators
            current_step: Current position in data
            
        Returns:
            TechnicalState object
        """
        if indicators_df is None or current_step >= len(indicators_df):
            # Return default values if no indicators
            return TechnicalState(
                rsi=50, macd=0, macd_signal=0, bb_position=0.5,
                volume_ratio=1, momentum=0, support_distance=0, resistance_distance=0
            )
        
        # Get indicator values
        indicators = indicators_df.iloc[current_step]
        
        # Extract key indicators (adjust based on what's available)
        rsi = indicators.get('rsi_14', 50) / 100  # Normalize to [0, 1]
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        bb_position = indicators.get('bb_position', 0.5)
        
        # Volume ratio
        current_volume = ohlcv_data.iloc[current_step]['volume']
        avg_volume = ohlcv_data['volume'].rolling(20).mean().iloc[current_step]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Momentum (rate of change)
        if current_step >= 10:
            past_price = ohlcv_data.iloc[current_step - 10]['close']
            current_price = ohlcv_data.iloc[current_step]['close']
            momentum = (current_price - past_price) / past_price
        else:
            momentum = 0
        
        # Distance to support/resistance
        current_price = ohlcv_data.iloc[current_step]['close']
        support = indicators.get('support_1', current_price * 0.95)
        resistance = indicators.get('resistance_1', current_price * 1.05)
        
        support_distance = (current_price - support) / current_price
        resistance_distance = (resistance - current_price) / current_price
        
        return TechnicalState(
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            bb_position=bb_position,
            volume_ratio=volume_ratio,
            momentum=momentum,
            support_distance=support_distance,
            resistance_distance=resistance_distance
        )
    
    def create_state_vector(self,
                          market_state: MarketState,
                          account_state: AccountState,
                          technical_state: Optional[TechnicalState] = None) -> np.ndarray:
        """
        Create feature vector from states
        
        Args:
            market_state: Market state
            account_state: Account state
            technical_state: Technical indicators state
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Market features
        if self.feature_version == 'v1':
            # Basic features
            features.extend([
                market_state.current_price,
                market_state.volatility,
                market_state.trend,
                (market_state.current_price - market_state.support_level) / market_state.current_price,
                (market_state.resistance_level - market_state.current_price) / market_state.current_price
            ])
            
            # Price history (last N prices normalized)
            if len(market_state.price_history) > 0:
                normalized_prices = market_state.price_history / market_state.current_price
                features.extend(normalized_prices[-10:])  # Last 10 prices
            
            # Volume features
            if len(market_state.volume_history) > 0:
                avg_volume = np.mean(market_state.volume_history)
                if avg_volume > 0:
                    normalized_volumes = market_state.volume_history / avg_volume
                    features.extend(normalized_volumes[-5:])  # Last 5 volumes
        
        elif self.feature_version == 'v2':
            # Advanced features with more history
            features.extend(self._extract_advanced_market_features(market_state))
        
        # Account features
        features.extend([
            account_state.balance / 10000,  # Normalize by initial balance
            account_state.equity / 10000,
            account_state.position_type,
            account_state.position_size,
            account_state.unrealized_pnl / 1000,
            account_state.realized_pnl / 1000,
            account_state.win_rate,
            account_state.sharpe_ratio,
            account_state.max_drawdown,
            account_state.current_drawdown
        ])
        
        # Technical features
        if technical_state:
            features.extend([
                technical_state.rsi,
                technical_state.macd / 100,  # Normalize MACD
                technical_state.macd_signal / 100,
                technical_state.bb_position,
                technical_state.volume_ratio,
                technical_state.momentum,
                technical_state.support_distance,
                technical_state.resistance_distance
            ])
        
        # Convert to numpy array
        state_vector = np.array(features, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize:
            state_vector = self._normalize_features(state_vector)
        
        return state_vector
    
    def _extract_advanced_market_features(self, market_state: MarketState) -> List[float]:
        """
        Extract advanced market features
        
        Args:
            market_state: Market state object
            
        Returns:
            List of advanced features
        """
        features = []
        
        # Price momentum at different scales
        if len(market_state.price_history) >= 20:
            prices = market_state.price_history
            features.append((prices[-1] - prices[-5]) / prices[-5])  # 5-period momentum
            features.append((prices[-1] - prices[-10]) / prices[-10])  # 10-period momentum
            features.append((prices[-1] - prices[-20]) / prices[-20])  # 20-period momentum
        else:
            features.extend([0, 0, 0])
        
        # Volume profile
        if len(market_state.volume_history) >= 10:
            volumes = market_state.volume_history
            recent_avg = np.mean(volumes[-5:])
            older_avg = np.mean(volumes[-10:-5])
            volume_trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            features.append(volume_trend)
        else:
            features.append(0)
        
        # Volatility regime
        if len(market_state.returns_history) >= 20:
            returns = market_state.returns_history
            recent_vol = np.std(returns[-10:])
            older_vol = np.std(returns[-20:-10])
            vol_ratio = recent_vol / older_vol if older_vol > 0 else 1
            features.append(vol_ratio)
        else:
            features.append(1)
        
        # Price position in recent range
        if len(market_state.price_history) >= 20:
            recent_high = np.max(market_state.price_history[-20:])
            recent_low = np.min(market_state.price_history[-20:])
            price_position = (market_state.current_price - recent_low) / (recent_high - recent_low) \
                           if recent_high > recent_low else 0.5
            features.append(price_position)
        else:
            features.append(0.5)
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using running statistics
        
        Args:
            features: Raw feature vector
            
        Returns:
            Normalized feature vector
        """
        # Simple min-max normalization for now
        # In production, you'd want to track running statistics
        normalized = np.clip(features, -10, 10) / 10
        return normalized
    
    def discretize_state(self, state_vector: np.ndarray, n_bins: int = 10) -> str:
        """
        Discretize continuous state for tabular methods
        
        Args:
            state_vector: Continuous state vector
            n_bins: Number of bins per dimension
            
        Returns:
            Discretized state as string
        """
        # Discretize each feature
        discretized = []
        for feature in state_vector:
            bin_idx = int(np.clip((feature + 1) * n_bins / 2, 0, n_bins - 1))
            discretized.append(str(bin_idx))
        
        # Create state string
        state_str = '_'.join(discretized)
        return state_str
    
    def hash_state(self, state_vector: np.ndarray) -> str:
        """
        Create hash of state for state visitation tracking
        
        Args:
            state_vector: State vector
            
        Returns:
            State hash
        """
        # Convert to bytes and hash
        state_bytes = state_vector.tobytes()
        state_hash = hashlib.md5(state_bytes).hexdigest()[:8]
        return state_hash
    
    def track_state_visit(self, state_vector: np.ndarray) -> int:
        """
        Track state visitation for exploration
        
        Args:
            state_vector: Current state vector
            
        Returns:
            Visit count for this state
        """
        state_hash = self.hash_state(state_vector)
        
        if state_hash not in self.state_visits:
            self.state_visits[state_hash] = 0
        
        self.state_visits[state_hash] += 1
        return self.state_visits[state_hash]
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about state visitation
        
        Returns:
            Dictionary with state statistics
        """
        if not self.state_visits:
            return {}
        
        visit_counts = list(self.state_visits.values())
        
        return {
            'unique_states': len(self.state_visits),
            'total_visits': sum(visit_counts),
            'max_visits': max(visit_counts),
            'min_visits': min(visit_counts),
            'avg_visits': np.mean(visit_counts),
            'most_visited_states': sorted(
                self.state_visits.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def augment_state(self, 
                     state_vector: np.ndarray,
                     additional_features: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Augment state with additional features
        
        Args:
            state_vector: Base state vector
            additional_features: Additional features to add
            
        Returns:
            Augmented state vector
        """
        if additional_features is None:
            return state_vector
        
        # Add additional features
        extra_features = list(additional_features.values())
        augmented = np.concatenate([state_vector, extra_features])
        
        return augmented
    
    def save_normalization_stats(self, filepath: str) -> None:
        """
        Save normalization statistics
        
        Args:
            filepath: Path to save statistics
        """
        import pickle
        
        stats = {
            'means': self.feature_means,
            'stds': self.feature_stds,
            'window_size': self.window_size,
            'feature_version': self.feature_version
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(stats, f)
    
    def load_normalization_stats(self, filepath: str) -> None:
        """
        Load normalization statistics
        
        Args:
            filepath: Path to load statistics from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        
        self.feature_means = stats['means']
        self.feature_stds = stats['stds']
        self.window_size = stats['window_size']
        self.feature_version = stats['feature_version']


# Test the state manager
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(45000, 46000, len(dates)),
        'low': np.random.uniform(39000, 40000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Initialize state manager
    state_manager = StateManager(window_size=20)
    
    # Extract states
    market_state = state_manager.extract_market_state(sample_data, 50)
    
    print("Market State:")
    print(f"  Current Price: {market_state.current_price:.2f}")
    print(f"  Volatility: {market_state.volatility:.4f}")
    print(f"  Trend: {market_state.trend:.4f}")
    print(f"  Support: {market_state.support_level:.2f}")
    print(f"  Resistance: {market_state.resistance_level:.2f}")
    
    # Create mock account state
    from dataclasses import replace
    account_state = AccountState(
        balance=10000, equity=10500, position_size=0.1,
        position_type=1, entry_price=42000, unrealized_pnl=100,
        realized_pnl=200, total_trades=10, win_rate=0.6,
        sharpe_ratio=1.5, max_drawdown=0.1, current_drawdown=0.05
    )
    
    # Create state vector
    state_vector = state_manager.create_state_vector(market_state, account_state)
    print(f"\nState Vector Shape: {state_vector.shape}")
    print(f"State Vector Sample: {state_vector[:10]}")
    
    # Test state hashing
    state_hash = state_manager.hash_state(state_vector)
    print(f"\nState Hash: {state_hash}")
    
    # Track visits
    visits = state_manager.track_state_visit(state_vector)
    print(f"State Visits: {visits}")
    
    print("\n State Manager ready!")