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
from scipy import stats
from scipy.signal import argrelextrema

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
        Calculate ALL 100+ features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            symbol: Optional symbol name for multi-asset features
            
        Returns:
            DataFrame with all calculated features (100+ features)
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
        
        # 4. Technical indicators (50 features) - ENHANCED
        indicator_features = self.calculate_indicator_features(df)
        features = pd.concat([features, indicator_features], axis=1)
        
        # 5. Pattern features (25 features) - ENHANCED
        pattern_features = self.calculate_pattern_features(df)
        features = pd.concat([features, pattern_features], axis=1)
        
        # 6. Moving average features (20 features)
        ma_features = self.calculate_ma_features(df)
        features = pd.concat([features, ma_features], axis=1)
        
        # 7. Advanced features (15+ features) - NEW
        advanced_features = self.calculate_advanced_features(df)
        features = pd.concat([features, advanced_features], axis=1)
        
        # Add symbol encoding if provided
        if symbol:
            features['symbol_encoded'] = self._encode_symbol(symbol)
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        print(f"Total features calculated: {len(features.columns)}")
        
        return features
    
    # ================== PRICE FEATURES ==================
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features
        30 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns at different periods (20 features)
        for period in self.config['price_periods']:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios (4 features)
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_close_ratio'] = df['high'] / df['close']
        features['low_close_ratio'] = df['low'] / df['close']
        
        # Price position in daily range (1 feature)
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap features (3 features)
        features['gap'] = df['open'] / df['close'].shift(1) - 1
        features['gap_up'] = (features['gap'] > 0).astype(int)
        features['gap_down'] = (features['gap'] < 0).astype(int)
        
        # Rolling price features (12 features)
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
        20 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic volume features (1 feature)
        features['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages and ratios (12 features)
        for period in self.config['volume_periods']:
            vol_ma = df['volume'].rolling(period).mean()
            features[f'volume_ma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-10)
            
            # Volume standard deviation
            features[f'volume_std_{period}'] = df['volume'].rolling(period).std()
        
        # Price-Volume features (2 features)
        features['price_volume'] = df['close'] * df['volume']
        features['price_volume_change'] = features['price_volume'].pct_change()
        
        # Volume-weighted price (3 features)
        for period in [5, 10, 20]:
            vwap = (df['close'] * df['volume']).rolling(period).sum() / (df['volume'].rolling(period).sum() + 1e-10)
            features[f'close_vwap_ratio_{period}'] = df['close'] / vwap
        
        # Volume momentum (2 features)
        features['volume_momentum_3'] = df['volume'] / df['volume'].shift(3)
        features['volume_momentum_10'] = df['volume'] / df['volume'].shift(10)
        
        # High/Low volume flags (2 features)
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        features['high_volume'] = (df['volume'] > vol_mean + 2 * vol_std).astype(int)
        features['low_volume'] = (df['volume'] < vol_mean - vol_std).astype(int)
        
        return features
    
    # ================== VOLATILITY FEATURES ==================
    
    def calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features
        15 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Rolling volatility (8 features)
        returns = df['close'].pct_change()
        
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'volatility_ratio_{period}'] = (
                returns.rolling(period).std() / returns.rolling(period * 2).std()
            )
        
        # Parkinson volatility (3 features)
        for period in [5, 10, 20]:
            hl_ratio = np.log(df['high'] / df['low'])
            features[f'parkinson_vol_{period}'] = np.sqrt(
                hl_ratio.rolling(period).apply(lambda x: np.sum(x**2) / (4 * len(x) * np.log(2)))
            )
        
        # Garman-Klass volatility (2 features)
        for period in [10, 20]:
            term1 = 0.5 * np.log(df['high'] / df['low']) ** 2
            term2 = (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
            features[f'garman_klass_vol_{period}'] = np.sqrt(
                (term1 - term2).rolling(period).mean()
            )
        
        # ATR features (2 features)
        atr = self.indicators.atr(df['high'], df['low'], df['close'])
        features['atr'] = atr
        features['atr_ratio'] = atr / df['close']
        
        # Volatility change (1 feature - but volatility_20 might not exist yet)
        if 'volatility_20' in features.columns:
            features['volatility_change'] = features['volatility_20'].pct_change(5)
        
        return features
    
    # ================== TECHNICAL INDICATORS ==================
    
    def calculate_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicator features - ENHANCED
        50 features (added missing indicators)
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI variations (4 features)
        features['rsi_14'] = self.indicators.rsi(df['close'], 14)
        features['rsi_21'] = self.indicators.rsi(df['close'], 21)
        features['rsi_7'] = self.indicators.rsi(df['close'], 7)
        features['rsi_divergence'] = features['rsi_14'] - features['rsi_21']
        
        # MACD (4 features)
        macd, signal, histogram = self.indicators.macd(df['close'])
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_signal_diff'] = macd - signal
        
        # Stochastic (3 features)
        stoch_k, stoch_d = self.indicators.stochastic(df['high'], df['low'], df['close'])
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        features['stoch_k_d_diff'] = stoch_k - stoch_d
        
        # Other momentum indicators (5 features)
        features['williams_r'] = self.indicators.williams_r(df['high'], df['low'], df['close'])
        features['cci'] = self.indicators.cci(df['high'], df['low'], df['close'])
        features['roc_10'] = self.indicators.roc(df['close'], 10)
        features['roc_20'] = self.indicators.roc(df['close'], 20)
        features['mfi'] = self.indicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Bollinger Bands (5 features)
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(df['close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_width'] = bb_upper - bb_lower
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features['bb_squeeze'] = features['bb_width'] / features['bb_width'].rolling(20).mean()
        
        # Keltner Channels (2 features)
        kc_upper, kc_middle, kc_lower = self.indicators.keltner_channels(df['high'], df['low'], df['close'])
        features['kc_position'] = (df['close'] - kc_lower) / (kc_upper - kc_lower + 1e-10)
        features['bb_kc_squeeze'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)
        
        # Donchian Channels (1 feature)
        dc_upper, dc_middle, dc_lower = self.indicators.donchian_channels(df['high'], df['low'])
        features['dc_position'] = (df['close'] - dc_lower) / (dc_upper - dc_lower + 1e-10)
        
        # Trend indicators (5 features)
        features['adx'] = self.indicators.adx(df['high'], df['low'], df['close'])
        aroon_up, aroon_down, aroon_osc = self.indicators.aroon(df['high'], df['low'])
        features['aroon_up'] = aroon_up
        features['aroon_down'] = aroon_down
        features['aroon_oscillator'] = aroon_osc
        features['psar'] = self.indicators.psar(df['high'], df['low'])
        features['psar_signal'] = (df['close'] > features['psar']).astype(int)
        
        # Volume indicators (8 features)
        features['obv'] = self.indicators.obv(df['close'], df['volume'])
        features['obv_sma_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        features['ad_line'] = self.indicators.ad_line(df['high'], df['low'], df['close'], df['volume'])
        features['ad_line_sma_ratio'] = features['ad_line'] / features['ad_line'].rolling(20).mean()
        features['cmf'] = self.indicators.cmf(df['high'], df['low'], df['close'], df['volume'])
        features['fi'] = self.indicators.fi(df['close'], df['volume'])
        features['vwap'] = self.indicators.vwap(df['high'], df['low'], df['close'], df['volume'])
        features['close_vwap_ratio'] = df['close'] / (features['vwap'] + 1e-10)
        
        # ICHIMOKU CLOUD (5 features) - NEW
        features['ichimoku_tenkan'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        features['ichimoku_kijun'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        features['ichimoku_senkou_a'] = ((features['ichimoku_tenkan'] + features['ichimoku_kijun']) / 2).shift(26)
        features['ichimoku_senkou_b'] = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        features['ichimoku_chikou'] = df['close'].shift(-26)
        
        # CHAIKIN OSCILLATOR (1 feature) - NEW
        ad = ((2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + 1e-10)) * df['volume']
        features['chaikin_osc'] = ad.ewm(span=3).mean() - ad.ewm(span=10).mean()
        
        # ULTIMATE OSCILLATOR (1 feature) - NEW
        bp = df['close'] - np.minimum(df['low'], df['close'].shift(1))
        tr = np.maximum(df['high'] - df['low'], 
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                  abs(df['low'] - df['close'].shift(1))))
        avg7 = (bp.rolling(7).sum() / tr.rolling(7).sum())
        avg14 = (bp.rolling(14).sum() / tr.rolling(14).sum())
        avg28 = (bp.rolling(28).sum() / tr.rolling(28).sum())
        features['ultimate_oscillator'] = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
        
        # TRIX (1 feature) - NEW
        ema1 = df['close'].ewm(span=14, adjust=False).mean()
        ema2 = ema1.ewm(span=14, adjust=False).mean()
        ema3 = ema2.ewm(span=14, adjust=False).mean()
        features['trix'] = ema3.pct_change() * 10000
        
        return features
    
    # ================== PATTERN FEATURES ==================
    
    def calculate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pattern-based features - ENHANCED
        25 features (added missing patterns)
        """
        features = pd.DataFrame(index=df.index)
        
        # Candlestick body and shadow features (3 features)
        body = df['close'] - df['open']
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        total_range = df['high'] - df['low']
        
        features['body_ratio'] = body / (total_range + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-10)
        
        # Basic candlestick patterns (5 features)
        features['is_doji'] = (np.abs(body) / (total_range + 1e-10) < 0.1).astype(int)
        features['is_hammer'] = ((lower_shadow > 2 * np.abs(body)) & 
                                 (upper_shadow < np.abs(body) * 0.3)).astype(int)
        features['is_shooting_star'] = ((upper_shadow > 2 * np.abs(body)) & 
                                        (lower_shadow < np.abs(body) * 0.3)).astype(int)
        
        # Engulfing patterns (2 features)
        prev_body = body.shift(1)
        features['bullish_engulfing'] = ((body > 0) & (prev_body < 0) & 
                                         (df['close'] > df['open'].shift(1)) & 
                                         (df['open'] < df['close'].shift(1))).astype(int)
        features['bearish_engulfing'] = ((body < 0) & (prev_body > 0) & 
                                         (df['close'] < df['open'].shift(1)) & 
                                         (df['open'] > df['close'].shift(1))).astype(int)
        
        # Advanced candlestick patterns (5 features) - NEW
        # Morning star pattern
        features['morning_star'] = (
            (body.shift(2) < 0) &  # First candle is bearish
            (np.abs(body.shift(1)) < np.abs(body.shift(2)) * 0.3) &  # Second is small
            (body > 0) &  # Third is bullish
            (df['close'] > df['open'].shift(2))  # Close above first open
        ).astype(int)
        
        # Evening star pattern
        features['evening_star'] = (
            (body.shift(2) > 0) &  # First candle is bullish
            (np.abs(body.shift(1)) < np.abs(body.shift(2)) * 0.3) &  # Second is small
            (body < 0) &  # Third is bearish
            (df['close'] < df['open'].shift(2))  # Close below first open
        ).astype(int)
        
        # Three white soldiers
        features['three_white_soldiers'] = (
            (body > 0) & (body.shift(1) > 0) & (body.shift(2) > 0) &
            (df['close'] > df['close'].shift(1)) & 
            (df['close'].shift(1) > df['close'].shift(2))
        ).astype(int)
        
        # Three black crows
        features['three_black_crows'] = (
            (body < 0) & (body.shift(1) < 0) & (body.shift(2) < 0) &
            (df['close'] < df['close'].shift(1)) & 
            (df['close'].shift(1) < df['close'].shift(2))
        ).astype(int)
        
        # Tweezer tops and bottoms
        features['tweezer_pattern'] = (
            (np.abs(df['high'] - df['high'].shift(1)) < df['close'] * 0.001) |
            (np.abs(df['low'] - df['low'].shift(1)) < df['close'] * 0.001)
        ).astype(int)
        
        # Support and Resistance (8 features)
        for period in [20, 50]:
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            features[f'distance_from_high_{period}'] = (rolling_high - df['close']) / df['close']
            features[f'distance_from_low_{period}'] = (df['close'] - rolling_low) / df['close']
            features[f'breaking_high_{period}'] = (df['close'] > rolling_high.shift(1)).astype(int)
            features[f'breaking_low_{period}'] = (df['close'] < rolling_low.shift(1)).astype(int)
        
        # Pivot points (2 features)
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        features['pivot_point'] = pivot
        features['distance_from_pivot'] = (df['close'] - pivot) / pivot
        
        # Chart patterns detection (5 features) - NEW
        # Head and shoulders (simplified detection)
        window = 20
        peaks = argrelextrema(df['high'].values, np.greater, order=window//2)[0]
        valleys = argrelextrema(df['low'].values, np.less, order=window//2)[0]
        
        features['potential_reversal'] = 0
        for i in range(len(df)):
            # Check for potential reversal patterns
            recent_peaks = [p for p in peaks if i-window <= p < i]
            recent_valleys = [v for v in valleys if i-window <= v < i]
            if len(recent_peaks) >= 2 or len(recent_valleys) >= 2:
                features.iloc[i, features.columns.get_loc('potential_reversal')] = 1
        
        # Double top/bottom detection
        features['double_top'] = 0
        features['double_bottom'] = 0
        for i in range(window, len(df)):
            window_high = df['high'].iloc[i-window:i]
            window_low = df['low'].iloc[i-window:i]
            
            # Detect double top
            high_peaks = argrelextrema(window_high.values, np.greater, order=3)[0]
            if len(high_peaks) >= 2:
                peak_values = window_high.iloc[high_peaks].values
                if np.abs(peak_values[-1] - peak_values[-2]) / peak_values[-1] < 0.02:
                    features.iloc[i, features.columns.get_loc('double_top')] = 1
            
            # Detect double bottom
            low_valleys = argrelextrema(window_low.values, np.less, order=3)[0]
            if len(low_valleys) >= 2:
                valley_values = window_low.iloc[low_valleys].values
                if np.abs(valley_values[-1] - valley_values[-2]) / valley_values[-1] < 0.02:
                    features.iloc[i, features.columns.get_loc('double_bottom')] = 1
        
        # Triangle pattern detection (ascending/descending)
        features['triangle_pattern'] = 0
        for i in range(window*2, len(df)):
            window_data = df.iloc[i-window*2:i]
            highs = window_data['high'].rolling(5).max()
            lows = window_data['low'].rolling(5).min()
            
            # Check for converging lines (triangle)
            high_slope = np.polyfit(range(len(highs.dropna())), highs.dropna(), 1)[0]
            low_slope = np.polyfit(range(len(lows.dropna())), lows.dropna(), 1)[0]
            
            if np.abs(high_slope + low_slope) < np.abs(high_slope) * 0.1:
                features.iloc[i, features.columns.get_loc('triangle_pattern')] = 1
        
        # Flag pattern detection
        features['flag_pattern'] = ((df['high'] - df['low']) < 
                                   (df['high'] - df['low']).rolling(10).mean() * 0.5).astype(int)
        
        return features
    
    # ================== MOVING AVERAGE FEATURES ==================
    
    def calculate_ma_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average based features
        20 features
        """
        features = pd.DataFrame(index=df.index)
        
        # Simple Moving Averages (15 features)
        for period in self.config['ma_periods']:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'close_sma_{period}_ratio'] = df['close'] / (sma + 1e-10)
            features[f'sma_{period}_slope'] = sma.pct_change(5)
        
        # Exponential Moving Averages (6 features)
        for period in [12, 26]:
            ema = df['close'].ewm(span=period, adjust=False).mean()
            features[f'ema_{period}'] = ema
            features[f'close_ema_{period}_ratio'] = df['close'] / (ema + 1e-10)
            features[f'ema_{period}_slope'] = ema.pct_change(5)
        
        # MA crossovers (3 features)
        if 'sma_10' in features and 'sma_20' in features:
            features['ma_cross_10_20'] = (
                (features['sma_10'] > features['sma_20']) & 
                (features['sma_10'].shift(1) <= features['sma_20'].shift(1))
            ).astype(int)
        
        if 'sma_20' in features and 'sma_50' in features:
            features['ma_cross_20_50'] = (
                (features['sma_20'] > features['sma_50']) & 
                (features['sma_20'].shift(1) <= features['sma_50'].shift(1))
            ).astype(int)
        
        if 'sma_50' in features and 'sma_200' in features:
            features['ma_cross_50_200'] = (
                (features['sma_50'] > features['sma_200']) & 
                (features['sma_50'].shift(1) <= features['sma_200'].shift(1))
            ).astype(int)
        
        # MA alignment (1 feature)
        if all(f'sma_{p}' in features for p in [10, 20, 50, 100]):
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
        Calculate features using multiple timeframes - ENHANCED
        30 features
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
            
            # Core multi-timeframe features (6 per timeframe)
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
            
            # Additional multi-timeframe features (4 per timeframe) - NEW
            # Volatility alignment
            features[f'volatility_{tf_name}'] = resampled['close'].pct_change().rolling(20).std()
            
            # MACD signal
            macd, signal, _ = self.indicators.macd(resampled['close'])
            features[f'macd_signal_{tf_name}'] = (macd > signal).astype(int)
            
            # Bollinger Band position
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(resampled['close'])
            features[f'bb_position_{tf_name}'] = (resampled['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            # Volume trend
            features[f'volume_trend_{tf_name}'] = resampled['volume'].rolling(20).mean().pct_change(5)
        
        # Cross-timeframe alignment features (10 features) - ENHANCED
        if '1h' in data_dict and '4h' in data_dict and '1d' in data_dict:
            # Trend alignment score
            trend_score = 0
            for tf in ['1h', '4h', '1d']:
                if f'trend_{tf}' in features.columns:
                    trend_score += features[f'trend_{tf}']
            features['trend_alignment_score'] = trend_score / 3
            
            # RSI confluence
            rsi_values = []
            for tf in ['1h', '4h', '1d']:
                if f'rsi_14_{tf}' in features.columns:
                    rsi_values.append(features[f'rsi_14_{tf}'])
            if rsi_values:
                features['rsi_confluence'] = pd.concat(rsi_values, axis=1).mean(axis=1)
                features['rsi_divergence_score'] = pd.concat(rsi_values, axis=1).std(axis=1)
            
            # Momentum alignment
            momentum_values = []
            for tf in ['1h', '4h', '1d']:
                if f'momentum_{tf}' in features.columns:
                    momentum_values.append(features[f'momentum_{tf}'])
            if momentum_values:
                features['momentum_alignment'] = pd.concat(momentum_values, axis=1).mean(axis=1)
            
            # Volatility convergence
            vol_values = []
            for tf in ['1h', '4h', '1d']:
                if f'volatility_{tf}' in features.columns:
                    vol_values.append(features[f'volatility_{tf}'])
            if vol_values:
                features['volatility_convergence'] = pd.concat(vol_values, axis=1).std(axis=1)
            
            # Support/Resistance confluence
            if all(f'support_{tf}' in features for tf in ['1h', '4h', '1d']):
                features['support_confluence'] = pd.concat(
                    [features[f'support_{tf}'] for tf in ['1h', '4h', '1d']], axis=1
                ).mean(axis=1)
            
            if all(f'resistance_{tf}' in features for tf in ['1h', '4h', '1d']):
                features['resistance_confluence'] = pd.concat(
                    [features[f'resistance_{tf}'] for tf in ['1h', '4h', '1d']], axis=1
                ).mean(axis=1)
            
            # Volume profile alignment
            if all(f'volume_profile_{tf}' in features for tf in ['1h', '4h', '1d']):
                features['volume_profile_ratio'] = (
                    features['volume_profile_1h'] / (features['volume_profile_1d'] + 1e-10)
                )
            
            # MACD alignment score
            if all(f'macd_signal_{tf}' in features for tf in ['1h', '4h', '1d']):
                macd_score = sum(features[f'macd_signal_{tf}'] for tf in ['1h', '4h', '1d'])
                features['macd_alignment_score'] = macd_score / 3
            
            # Timeframe strength indicator
            features['timeframe_strength'] = (
                features.get('trend_alignment_score', 0) * 0.3 +
                features.get('momentum_alignment', 0) * 0.3 +
                features.get('macd_alignment_score', 0) * 0.4
            )
        
        return features
    
     # ================== ADVANCED FEATURES (15+) - NEW ==================
    
    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced statistical and market microstructure features
        15+ features - NEW
        """
        features = pd.DataFrame(index=df.index)
        
        # Market Microstructure Features (5 features)
        # Amihud Illiquidity
        returns = df['close'].pct_change()
        features['amihud_illiquidity'] = np.abs(returns) / (df['volume'] + 1e-10)
        features['amihud_illiquidity_ma'] = features['amihud_illiquidity'].rolling(20).mean()
        
        # Roll's implied spread
        features['roll_spread'] = 2 * np.sqrt(np.abs(returns.rolling(20).cov(returns.shift(1))))
        
        # Kyle's lambda (price impact)
        features['kyle_lambda'] = returns.rolling(20).std() / np.sqrt(df['volume'].rolling(20).mean() + 1e-10)
        
        # Volume-synchronized probability of informed trading (VPIN approximation)
        features['vpin_proxy'] = np.abs(returns) / (df['volume'].rolling(20).std() + 1e-10)
        
        # Statistical Features (5 features)
        # Skewness and Kurtosis
        features['return_skew'] = returns.rolling(20).skew()
        features['return_kurtosis'] = returns.rolling(20).kurt()
        
        # Jarque-Bera test statistic for normality
        n = 20
        jb_stat = []
        for i in range(len(returns)):
            if i < n:
                jb_stat.append(np.nan)
            else:
                window = returns.iloc[i-n:i].dropna()
                if len(window) > 3:
                    s = stats.skew(window)
                    k = stats.kurtosis(window)
                    jb = (len(window)/6) * (s**2 + (k**2)/4)
                    jb_stat.append(jb)
                else:
                    jb_stat.append(np.nan)
        features['jarque_bera'] = jb_stat
        
        # Hurst exponent (trend persistence)
        def hurst_exponent(ts, lag=20):
            """Calculate Hurst exponent"""
            lags = range(2, min(lag, len(ts)//2))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        hurst_values = []
        for i in range(len(df)):
            if i < 50:
                hurst_values.append(np.nan)
            else:
                try:
                    h = hurst_exponent(df['close'].iloc[i-50:i].values)
                    hurst_values.append(h)
                except:
                    hurst_values.append(np.nan)
        features['hurst_exponent'] = hurst_values
        
        # Autocorrelation
        features['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
        
        # Entropy-based features (3 features)
        # Shannon entropy of returns
        def shannon_entropy(series, bins=10):
            """Calculate Shannon entropy"""
            hist, _ = np.histogram(series.dropna(), bins=bins)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist))
        
        features['shannon_entropy'] = returns.rolling(20).apply(
            lambda x: shannon_entropy(x) if len(x.dropna()) > 5 else np.nan
        )
        
        # Approximate entropy
        features['approx_entropy'] = returns.rolling(20).std() / (returns.rolling(10).std() + 1e-10)
        
        # Permutation entropy (simplified)
        features['perm_entropy'] = returns.rolling(20).apply(
            lambda x: len(np.unique(np.argsort(x))) / len(x) if len(x) > 0 else np.nan
        )
        
        # Regime detection features (3 features)
        # Moving average regime
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['regime_bullish'] = (sma_20 > sma_50).astype(int)
        features['regime_change'] = features['regime_bullish'].diff().abs()
        
        # Volatility regime (high/low vol)
        vol = returns.rolling(20).std()
        vol_median = vol.rolling(100).median()
        features['high_vol_regime'] = (vol > vol_median * 1.5).astype(int)
        
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