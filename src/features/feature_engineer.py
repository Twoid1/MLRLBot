"""
Feature Engineering Module
Main pipeline for calculating all features from OHLCV data
Combines price features, technical indicators, patterns, and multi-timeframe features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path
import joblib
from datetime import datetime

# Import our custom indicators module
from .indicators import TechnicalIndicators


class FeatureEngineer:
    """
    Complete feature engineering pipeline for trading bot
    Generates 100+ features from OHLCV data, selects top 50
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary with feature settings
        """
        self.config = config or self._get_default_config()
        self.indicators = TechnicalIndicators()
        self.feature_names = []
        self.selected_features = None
        self.feature_importance = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'price_periods': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],  # Fibonacci periods
            'ma_periods': [10, 20, 50, 100, 200],
            'volume_periods': [5, 10, 20, 50],
            'lookback_periods': [5, 10, 20, 50, 100],
            'target_features': 50,  # Top N features to select
            'min_periods': 200,  # Minimum periods needed for calculation
        }
    
    # ================== MAIN PIPELINE ==================
    
    def calculate_all_features(self, 
                               df: pd.DataFrame, 
                               symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate all features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Optional symbol name for multi-asset features
            
        Returns:
            DataFrame with all calculated features
        """
        # Validate input
        self._validate_ohlcv(df)
        
        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # 1. Price-based features (30 features)
        price_features = self.calculate_price_features(df)
        features = pd.concat([features, price_features], axis=1)
        
        # 2. Volume features (20 features)
        volume_features = self.calculate_volume_features(df)
        features = pd.concat([features, volume_features], axis=1)
        
        # 3. Volatility features (15 features)
        volatility_features = self.calculate_volatility_features(df)
        features = pd.concat([features, volatility_features], axis=1)
        
        # 4. Technical indicators (50 features)
        indicator_features = self.calculate_indicator_features(df)
        features = pd.concat([features, indicator_features], axis=1)
        
        # 5. Pattern features (15 features)
        pattern_features = self.calculate_pattern_features(df)
        features = pd.concat([features, pattern_features], axis=1)
        
        # 6. Moving average features (20 features)
        ma_features = self.calculate_ma_features(df)
        features = pd.concat([features, ma_features], axis=1)
        
        # Add symbol encoding if provided
        if symbol:
            features['symbol_encoded'] = self._encode_symbol(symbol)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features
    
    # ================== PRICE FEATURES ==================
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features
        ~30 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns at different periods
        for period in self.config['price_periods']:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']
        
        # Price position in daily range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap features
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        features['gap_up'] = (features['gap'] > 0).astype(int)
        features['gap_down'] = (features['gap'] < 0).astype(int)
        
        # Rolling price features
        for period in [5, 10, 20]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            features[f'close_to_high_{period}'] = df['close'] / rolling_high
            features[f'close_to_low_{period}'] = df['close'] / rolling_low
            features[f'high_low_spread_{period}'] = (rolling_high - rolling_low) / df['close']
            
            # Distance from rolling mean
            rolling_mean = df['close'].rolling(period).mean()
            features[f'close_to_mean_{period}'] = df['close'] / rolling_mean - 1
        
        return features
    
    # ================== VOLUME FEATURES ==================
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features
        ~20 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic volume features
        features['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages and ratios
        for period in self.config['volume_periods']:
            vol_ma = df['volume'].rolling(period).mean()
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-10)
            
            # Volume standard deviation
            features[f'volume_std_{period}'] = df['volume'].rolling(period).std()
        
        # Price-Volume features
        features['price_volume'] = df['close'] * df['volume']
        features['price_volume_change'] = features['price_volume'].pct_change()
        
        # Volume-weighted price
        for period in [5, 10, 20]:
            vwap = (df['close'] * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)
            features[f'close_vwap_ratio_{period}'] = df['close'] / vwap
        
        # Volume momentum
        features['volume_momentum_3'] = df['volume'] / df['volume'].shift(3)
        features['volume_momentum_10'] = df['volume'] / df['volume'].shift(10)
        
        # High volume flag
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        features['high_volume'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)
        features['low_volume'] = (df['volume'] < vol_mean - vol_std).astype(int)
        
        return features
    
    # ================== VOLATILITY FEATURES ==================
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features
        ~15 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Rolling volatility (standard deviation of returns)
        returns = df['close'].pct_change()
        
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'volatility_ratio_{period}'] = (
                returns.rolling(period).std() / returns.rolling(period * 2).std()
            )
        
        # Parkinson volatility (using high-low)
        for period in [5, 10, 20]:
            hl_ratio = np.log(df['high'] / df['low'])
            features[f'parkinson_vol_{period}'] = np.sqrt(
                hl_ratio.rolling(period).apply(lambda x: np.sum(x**2) / (4 * len(x) * np.log(2)))
            )
        
        # Garman-Klass volatility
        for period in [10, 20]:
            term1 = 0.5 * np.log(df['high'] / df['low']) ** 2
            term2 = (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
            features[f'garman_klass_vol_{period}'] = np.sqrt(
                (term1 - term2).rolling(period).mean()
            )
        
        # Average True Range (ATR) - already calculated in indicators
        atr = self.indicators.atr(df['high'], df['low'], df['close'])
        features['atr'] = atr
        features['atr_ratio'] = atr / df['close']
        
        # Volatility change
        features['volatility_change'] = features['volatility_20'].pct_change(5)
        
        return features
    
    # ================== TECHNICAL INDICATORS ==================
    
    def calculate_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicator features
        ~50 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Momentum indicators
        features['rsi_14'] = self.indicators.rsi(df['close'], 14)
        features['rsi_21'] = self.indicators.rsi(df['close'], 21)
        features['rsi_7'] = self.indicators.rsi(df['close'], 7)
        
        # RSI divergence
        features['rsi_divergence'] = features['rsi_14'] - features['rsi_21']
        
        # MACD
        macd, signal, histogram = self.indicators.macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_signal_diff'] = macd - signal
        
        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_k_d_diff'] = stoch_k - stoch_d
        
        # Williams %R
        features['williams_r'] = self.indicators.williams_r(df['high'], df['low'], df['close'])
        
        # CCI
        features['cci'] = self.indicators.cci(df['high'], df['low'], df['close'])
        
        # ROC
        features['roc_10'] = self.indicators.roc(df['close'], 10)
        features['roc_20'] = self.indicators.roc(df['close'], 20)
        
        # MFI
        features['mfi'] = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = bb_upper - bb_lower
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features['bb_squeeze'] = features['bb_width'] / features['bb_width'].rolling(20).mean()
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.indicators.keltner_channels(df['high'], df['low'], df['close'])
        features['kc_position'] = (df['close'] - kc_lower) / (kc_upper - kc_lower + 1e-10)
        features['bb_kc_squeeze'] = (bb_upper < kc_upper) & (bb_lower > kc_lower)
        
        # Donchian Channels
        dc_upper, dc_middle, dc_lower = self.indicators.donchian_channels(df['high'], df['low'])
        features['dc_position'] = (df['close'] - dc_lower) / (dc_upper - dc_lower + 1e-10)
        
        # ADX
        features['adx'] = self.indicators.adx(df['high'], df['low'], df['close'])
        
        # Aroon
        aroon_up, aroon_down, aroon_osc = self.indicators.aroon(df['high'], df['low'])
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_oscillator'] = aroon_osc
        
        # PSAR
        features['psar'] = self.indicators.psar(df['high'], df['low'])
        features['psar_signal'] = (df['close'] > features['psar']).astype(int)
        
        # Volume indicators
        features['obv'] = self.indicators.obv(df['close'], df['volume'])
        features['obv_sma_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        
        features['ad_line'] = self.indicators.ad_line(df['high'], df['low'], df['close'], df['volume'])
        features['ad_line_sma_ratio'] = features['ad_line'] / features['ad_line'].rolling(20).mean()
        
        features['cmf'] = self.indicators.cmf(df['high'], df['low'], df['close'], df['volume'])
        features['fi'] = self.indicators.fi(df['close'], df['volume'])
        
        # VWAP
        features['vwap'] = self.indicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        features['close_vwap_ratio'] = df['close'] / (features['vwap'] + 1e-10)
        
        return features
    
    # ================== PATTERN FEATURES ==================
    
    def calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern-based features (candlestick patterns, support/resistance)
        ~15 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Candlestick features
        body = df['close'] - df['open']
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        total_range = df['high'] - df['low']
        
        features['body_ratio'] = body / (total_range + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-10)
        
        # Candlestick patterns (simplified without TA-Lib)
        # Doji
        features['is_doji'] = (np.abs(body) / (total_range + 1e-10) < 0.1).astype(int)
        
        # Hammer
        features['is_hammer'] = (
            (lower_shadow > 2 * np.abs(body)) & 
            (upper_shadow < np.abs(body) * 0.3)
        ).astype(int)
        
        # Shooting star
        features['is_shooting_star'] = (
            (upper_shadow > 2 * np.abs(body)) & 
            (lower_shadow < np.abs(body) * 0.3)
        ).astype(int)
        
        # Engulfing patterns
        prev_body = body.shift(1)
        features['bullish_engulfing'] = (
            (body > 0) & (prev_body < 0) & 
            (df['close'] > df['open'].shift(1)) & 
            (df['open'] < df['close'].shift(1))
        ).astype(int)
        
        features['bearish_engulfing'] = (
            (body < 0) & (prev_body > 0) & 
            (df['close'] < df['open'].shift(1)) & 
            (df['open'] > df['close'].shift(1))
        ).astype(int)
        
        # Support and Resistance levels
        for period in [20, 50]:
            # Recent highs and lows
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            features[f'distance_from_high_{period}'] = (rolling_high - df['close']) / df['close']
            features[f'distance_from_low_{period}'] = (df['close'] - rolling_low) / df['close']
            
            # Breaking levels
            features[f'breaking_high_{period}'] = (df['close'] > rolling_high.shift(1)).astype(int)
            features[f'breaking_low_{period}'] = (df['close'] < rolling_low.shift(1)).astype(int)
        
        # Pivot points
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        features['pivot_point'] = pivot
        features['distance_from_pivot'] = (df['close'] - pivot) / pivot
        
        return features
    
    # ================== MOVING AVERAGE FEATURES ==================
    
    def calculate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average based features
        ~20 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages
        for period in self.config['ma_periods']:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'close_sma_{period}_ratio'] = df['close'] / (sma + 1e-10)
            features[f'sma_{period}_slope'] = sma.pct_change(5)
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_{period}'] = ema
            features[f'close_ema_{period}_ratio'] = df['close'] / (ema + 1e-10)
        
        # MA crossovers
        features['ma_cross_10_20'] = (
            (features['sma_10'] > features['sma_20']) & 
            (features['sma_10'].shift(1) <= features['sma_20'].shift(1))
        ).astype(int)
        
        features['ma_cross_20_50'] = (
            (features['sma_20'] > features['sma_50']) & 
            (features['sma_20'].shift(1) <= features['sma_50'].shift(1))
        ).astype(int)
        
        features['ma_cross_50_200'] = (
            (features['sma_50'] > features['sma_200']) & 
            (features['sma_50'].shift(1) <= features['sma_200'].shift(1))
        ).astype(int)
        
        # MA alignment (trend strength)
        features['ma_alignment'] = (
            (features['sma_10'] > features['sma_20']) & 
            (features['sma_20'] > features['sma_50']) & 
            (features['sma_50'] > features['sma_100'])
        ).astype(int) - (
            (features['sma_10'] < features['sma_20']) & 
            (features['sma_20'] < features['sma_50']) & 
            (features['sma_50'] < features['sma_100'])
        ).astype(int)
        
        return features
    
    # ================== MULTI-TIMEFRAME FEATURES ==================
    
    def calculate_multi_timeframe_features(self, 
                                          data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate features using multiple timeframes
        Expects dict with keys like '5m', '15m', '1h', '4h', '1d'
        
        Args:
            data_dict: Dictionary of DataFrames with different timeframes
            
        Returns:
            DataFrame with multi-timeframe features aligned to the smallest timeframe
        """
        # Get base timeframe (smallest)
        timeframe_order = ['5m', '15m', '1h', '4h', '1d', '1w']
        base_tf = None
        for tf in timeframe_order:
            if tf in data_dict:
                base_tf = tf
                base_df = data_dict[tf]
                break
        
        if base_tf is None:
            raise ValueError("No valid timeframe found in data_dict")
        
        features = pd.DataFrame(index=base_df.index)
        
        # For each higher timeframe
        for tf_name, tf_data in data_dict.items():
            if tf_name == base_tf:
                continue
            
            # Resample to align with base timeframe
            resampled = self._align_timeframe(tf_data, base_df.index)
            
            # RSI alignment
            rsi = self.indicators.rsi(resampled['close'], 14)
            features[f'rsi_14_{tf_name}'] = rsi
            
            # Trend direction
            sma_20 = resampled['close'].rolling(20).mean()
            sma_50 = resampled['close'].rolling(50).mean()
            features[f'trend_{tf_name}'] = (sma_20 > sma_50).astype(int)
            
            # Momentum
            features[f'momentum_{tf_name}'] = resampled['close'].pct_change(10)
            
            # Support/Resistance levels
            features[f'resistance_{tf_name}'] = resampled['high'].rolling(50).max()
            features[f'support_{tf_name}'] = resampled['low'].rolling(50).min()
            
            # Volume profile
            features[f'volume_profile_{tf_name}'] = resampled['volume'].rolling(20).mean()
        
        # Cross-timeframe alignment features
        if '1h' in data_dict and '4h' in data_dict and '1d' in data_dict:
            # Trend alignment score
            trend_score = 0
            for tf in ['1h', '4h', '1d']:
                if f'trend_{tf}' in features.columns:
                    trend_score += features[f'trend_{tf}']
            features['trend_alignment_score'] = trend_score / 3
            
            # RSI confluence
            rsi_avg = 0
            rsi_count = 0
            for tf in ['1h', '4h', '1d']:
                if f'rsi_14_{tf}' in features.columns:
                    rsi_avg += features[f'rsi_14_{tf}']
                    rsi_count += 1
            if rsi_count > 0:
                features['rsi_confluence'] = rsi_avg / rsi_count
        
        return features
    
    # ================== FEATURE SELECTION ==================
    
    def select_top_features(self, 
                           features: pd.DataFrame, 
                           target: pd.Series,
                           n_features: Optional[int] = None) -> List[str]:
        """
        Select top N features based on importance
        
        Args:
            features: DataFrame with all features
            target: Target variable for feature importance calculation
            n_features: Number of features to select (default from config)
            
        Returns:
            List of selected feature names
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        n_features = n_features or self.config['target_features']
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx].values
        y = target[valid_idx].values
        
        # Calculate feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_rf = pd.Series(rf.feature_importances_, index=features.columns)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        importance_mi = pd.Series(mi_scores, index=features.columns)
        
        # Combine both importance measures
        combined_importance = (importance_rf + importance_mi) / 2
        combined_importance = combined_importance.sort_values(ascending=False)
        
        # Store importance scores
        self.feature_importance = combined_importance
        
        # Select top features
        self.selected_features = combined_importance.head(n_features).index.tolist()
        
        return self.selected_features
    
    # ================== HELPER METHODS ==================
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Validate OHLCV DataFrame"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if len(df) < self.config['min_periods']:
            warnings.warn(f"DataFrame has {len(df)} rows, less than recommended {self.config['min_periods']}")
    
    def _encode_symbol(self, symbol: str) -> float:
        """
        Encode symbol as numeric value for multi-asset training
        """
        symbol_map = {
            'BTC/USDT': 1.0, 'ETH/USDT': 2.0, 'BNB/USDT': 3.0,
            'ADA/USDT': 4.0, 'SOL/USDT': 5.0, 'XRP/USDT': 6.0,
            'DOT/USDT': 7.0, 'DOGE/USDT': 8.0, 'AVAX/USDT': 9.0,
            'MATIC/USDT': 10.0, 'LINK/USDT': 11.0, 'LTC/USDT': 12.0,
            'ATOM/USDT': 13.0, 'UNI/USDT': 14.0, 'ALGO/USDT': 15.0
        }
        return symbol_map.get(symbol, 0.0)
    
    def _align_timeframe(self, 
                        higher_tf: pd.DataFrame, 
                        base_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align higher timeframe data to base timeframe index
        Forward fill to propagate values
        """
        # Resample and forward fill
        aligned = higher_tf.reindex(base_index, method='ffill')
        return aligned
    
    def create_feature_matrix(self, 
                             symbol_data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Create feature matrix for multiple symbols and timeframes
        
        Args:
            symbol_data_dict: Nested dict {symbol: {timeframe: DataFrame}}
            
        Returns:
            Combined feature matrix with all symbols
        """
        all_features = []
        
        for symbol, timeframe_data in symbol_data_dict.items():
            # Get base timeframe data
            base_tf = self._get_base_timeframe(timeframe_data)
            base_df = timeframe_data[base_tf]
            
            # Calculate single timeframe features
            features = self.calculate_all_features(base_df, symbol)
            
            # Add multi-timeframe features
            if len(timeframe_data) > 1:
                mtf_features = self.calculate_multi_timeframe_features(timeframe_data)
                features = pd.concat([features, mtf_features], axis=1)
            
            all_features.append(features)
        
        # Combine all symbols
        combined_features = pd.concat(all_features, axis=0)
        
        return combined_features
    
    def _get_base_timeframe(self, timeframe_data: Dict[str, pd.DataFrame]) -> str:
        """Get the smallest available timeframe"""
        timeframe_order = ['5m', '15m', '1h', '4h', '1d', '1w']
        for tf in timeframe_order:
            if tf in timeframe_data:
                return tf
        raise ValueError("No valid timeframe found")
    
    def save_features(self, features: pd.DataFrame, name: str, path: str = 'data/features/') -> None:
        """
        Save features to disk
        
        Args:
            features: DataFrame with features
            name: Name for the feature set
            path: Directory to save features
        """
        filepath = Path(path)
        filepath.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features.to_parquet(filepath / f'{name}_features.parquet')
        
        # Save feature names and importance if available
        if self.selected_features:
            joblib.dump(self.selected_features, filepath / f'{name}_selected_features.pkl')
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(filepath / f'{name}_feature_importance.csv')
        
        print(f"Features saved to {filepath / name}_features.parquet")
    
    def load_features(self, name: str, path: str = 'data/features/') -> pd.DataFrame:
        """
        Load features from disk
        
        Args:
            name: Name of the feature set
            path: Directory containing features
            
        Returns:
            DataFrame with features
        """
        filepath = Path(path)
        
        # Load features
        features = pd.read_parquet(filepath / f'{name}_features.parquet')
        
        # Load selected features if available
        selected_file = filepath / f'{name}_selected_features.pkl'
        if selected_file.exists():
            self.selected_features = joblib.load(selected_file)
        
        # Load feature importance if available
        importance_file = filepath / f'{name}_feature_importance.csv'
        if importance_file.exists():
            self.feature_importance = pd.read_csv(importance_file, index_col=0).squeeze()
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names"""
        return self.selected_features
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        return self.feature_importance