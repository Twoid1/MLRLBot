"""
Pattern Recognition Module - COMPLETE FIXED VERSION
All boolean operations fixed for NaN handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings


@dataclass
class PatternSignal:
    """Container for pattern detection results"""
    pattern_name: str
    timestamp: pd.Timestamp
    confidence: float
    direction: str  # 'bullish', 'bearish', or 'neutral'
    price_level: float
    metadata: Dict


class PatternRecognition:
    """
    Advanced pattern recognition for technical analysis
    All patterns implemented from scratch without TA-lib
    FIXED VERSION - Handles NaN values properly
    """
    
    def __init__(self, min_pattern_bars: int = 5):
        """
        Initialize pattern recognition
        
        Args:
            min_pattern_bars: Minimum bars required for pattern formation
        """
        self.min_pattern_bars = min_pattern_bars
        self.detected_patterns = []
        
    # ================== MAIN DETECTION PIPELINE ==================
    
    def detect_all_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all patterns in OHLCV data
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with pattern indicators
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Candlestick patterns
        candle_patterns = self.detect_candlestick_patterns(df)
        patterns = pd.concat([patterns, candle_patterns], axis=1)
        
        # Chart patterns
        chart_patterns = self.detect_chart_patterns(df)
        patterns = pd.concat([patterns, chart_patterns], axis=1)
        
        # Support and Resistance
        sr_levels = self.detect_support_resistance(df)
        patterns = pd.concat([patterns, sr_levels], axis=1)
        
        # Trend patterns
        trend_patterns = self.detect_trend_patterns(df)
        patterns = pd.concat([patterns, trend_patterns], axis=1)
        
        # Volume patterns
        volume_patterns = self.detect_volume_patterns(df)
        patterns = pd.concat([patterns, volume_patterns], axis=1)
        
        # Price action patterns
        price_action = self.detect_price_action_patterns(df)
        patterns = pd.concat([patterns, price_action], axis=1)
        
        return patterns
    
    # ================== CANDLESTICK PATTERNS (FIXED) ==================
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect all candlestick patterns - FIXED for NaN handling
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Calculate basic candle metrics
        body = df['close'] - df['open']
        body_abs = np.abs(body)
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        total_range = df['high'] - df['low']
        
        # Single candle patterns
        patterns['doji'] = self._detect_doji(df, body_abs, total_range)
        patterns['hammer'] = self._detect_hammer(df, body, upper_shadow, lower_shadow, body_abs)
        patterns['inverted_hammer'] = self._detect_inverted_hammer(df, body, upper_shadow, lower_shadow, body_abs)
        patterns['shooting_star'] = self._detect_shooting_star(df, body, upper_shadow, lower_shadow, body_abs)
        patterns['spinning_top'] = self._detect_spinning_top(body_abs, upper_shadow, lower_shadow, total_range)
        patterns['marubozu'] = self._detect_marubozu(body_abs, upper_shadow, lower_shadow, total_range)
        
        # Two candle patterns
        patterns['engulfing_bullish'] = self._detect_bullish_engulfing(df, body)
        patterns['engulfing_bearish'] = self._detect_bearish_engulfing(df, body)
        patterns['harami_bullish'] = self._detect_bullish_harami(df, body)
        patterns['harami_bearish'] = self._detect_bearish_harami(df, body)
        patterns['piercing_line'] = self._detect_piercing_line(df, body)
        patterns['dark_cloud_cover'] = self._detect_dark_cloud_cover(df, body)
        patterns['tweezer_top'] = self._detect_tweezer_top(df)
        patterns['tweezer_bottom'] = self._detect_tweezer_bottom(df)
        
        # Three candle patterns
        patterns['morning_star'] = self._detect_morning_star(df, body, body_abs)
        patterns['evening_star'] = self._detect_evening_star(df, body, body_abs)
        patterns['three_white_soldiers'] = self._detect_three_white_soldiers(df, body)
        patterns['three_black_crows'] = self._detect_three_black_crows(df, body)
        patterns['three_inside_up'] = self._detect_three_inside_up(df, body)
        patterns['three_inside_down'] = self._detect_three_inside_down(df, body)
        
        # Pattern strength scores
        patterns['candlestick_bullish_score'] = (
            patterns[['hammer', 'inverted_hammer', 'engulfing_bullish', 
                     'harami_bullish', 'piercing_line', 'tweezer_bottom',
                     'morning_star', 'three_white_soldiers', 'three_inside_up']].sum(axis=1)
        )
        
        patterns['candlestick_bearish_score'] = (
            patterns[['shooting_star', 'engulfing_bearish', 'harami_bearish',
                     'dark_cloud_cover', 'tweezer_top', 'evening_star',
                     'three_black_crows', 'three_inside_down']].sum(axis=1)
        )
        
        return patterns
    
    def _detect_doji(self, df, body_abs, total_range):
        """Detect Doji pattern"""
        doji_threshold = 0.1
        result = (body_abs <= total_range * doji_threshold)
        return result.fillna(0).astype(int)
    
    def _detect_hammer(self, df, body, upper_shadow, lower_shadow, body_abs):
        """Detect Hammer pattern - FIXED"""
        cond1 = (lower_shadow >= body_abs * 2).fillna(False)
        cond2 = (upper_shadow <= body_abs * 0.3).fillna(False)
        cond3 = (body > 0).fillna(False)
        cond4 = (df['close'].rolling(5).mean() < df['close']).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4).astype(int)
    
    def _detect_inverted_hammer(self, df, body, upper_shadow, lower_shadow, body_abs):
        """Detect Inverted Hammer pattern - FIXED"""
        cond1 = (upper_shadow >= body_abs * 2).fillna(False)
        cond2 = (lower_shadow <= body_abs * 0.3).fillna(False)
        cond3 = (body > 0).fillna(False)
        cond4 = (df['close'].rolling(5).mean() < df['close']).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4).astype(int)
    
    def _detect_shooting_star(self, df, body, upper_shadow, lower_shadow, body_abs):
        """Detect Shooting Star pattern - FIXED"""
        cond1 = (upper_shadow >= body_abs * 2).fillna(False)
        cond2 = (lower_shadow <= body_abs * 0.3).fillna(False)
        cond3 = (body < 0).fillna(False)
        cond4 = (df['close'].rolling(5).mean() > df['close']).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4).astype(int)
    
    def _detect_spinning_top(self, body_abs, upper_shadow, lower_shadow, total_range):
        """Detect Spinning Top pattern - FIXED"""
        cond1 = (body_abs <= total_range * 0.3).fillna(False)
        cond2 = (upper_shadow >= body_abs).fillna(False)
        cond3 = (lower_shadow >= body_abs).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    def _detect_marubozu(self, body_abs, upper_shadow, lower_shadow, total_range):
        """Detect Marubozu pattern - FIXED"""
        cond1 = (body_abs >= total_range * 0.95).fillna(False)
        cond2 = (upper_shadow <= total_range * 0.02).fillna(False)
        cond3 = (lower_shadow <= total_range * 0.02).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    def _detect_bullish_engulfing(self, df, body):
        """Detect Bullish Engulfing pattern - FIXED"""
        prev_body = body.shift(1)
        
        cond1 = (body > 0).fillna(False)
        cond2 = (prev_body < 0).fillna(False)
        cond3 = (df['open'] < df['close'].shift(1)).fillna(False)
        cond4 = (df['close'] > df['open'].shift(1)).fillna(False)
        cond5 = (np.abs(body) > np.abs(prev_body)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_bearish_engulfing(self, df, body):
        """Detect Bearish Engulfing pattern - FIXED"""
        prev_body = body.shift(1)
        
        cond1 = (body < 0).fillna(False)
        cond2 = (prev_body > 0).fillna(False)
        cond3 = (df['open'] > df['close'].shift(1)).fillna(False)
        cond4 = (df['close'] < df['open'].shift(1)).fillna(False)
        cond5 = (np.abs(body) > np.abs(prev_body)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_bullish_harami(self, df, body):
        """Detect Bullish Harami pattern - FIXED"""
        prev_body = body.shift(1)
        
        cond1 = (body > 0).fillna(False)
        cond2 = (prev_body < 0).fillna(False)
        cond3 = (df['open'] > df['close'].shift(1)).fillna(False)
        cond4 = (df['close'] < df['open'].shift(1)).fillna(False)
        cond5 = (np.abs(body) < np.abs(prev_body)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_bearish_harami(self, df, body):
        """Detect Bearish Harami pattern - FIXED"""
        prev_body = body.shift(1)
        
        cond1 = (body < 0).fillna(False)
        cond2 = (prev_body > 0).fillna(False)
        cond3 = (df['open'] < df['close'].shift(1)).fillna(False)
        cond4 = (df['close'] > df['open'].shift(1)).fillna(False)
        cond5 = (np.abs(body) < np.abs(prev_body)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_piercing_line(self, df, body):
        """Detect Piercing Line pattern - FIXED"""
        prev_body = body.shift(1)
        prev_midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
        
        cond1 = (body > 0).fillna(False)
        cond2 = (prev_body < 0).fillna(False)
        cond3 = (df['open'] < df['low'].shift(1)).fillna(False)
        cond4 = (df['close'] > prev_midpoint).fillna(False)
        cond5 = (df['close'] < df['open'].shift(1)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_dark_cloud_cover(self, df, body):
        """Detect Dark Cloud Cover pattern - FIXED"""
        prev_body = body.shift(1)
        prev_midpoint = (df['open'].shift(1) + df['close'].shift(1)) / 2
        
        cond1 = (body < 0).fillna(False)
        cond2 = (prev_body > 0).fillna(False)
        cond3 = (df['open'] > df['high'].shift(1)).fillna(False)
        cond4 = (df['close'] < prev_midpoint).fillna(False)
        cond5 = (df['close'] > df['close'].shift(1)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5).astype(int)
    
    def _detect_tweezer_top(self, df):
        """Detect Tweezer Top pattern - FIXED"""
        high_similarity = (np.abs(df['high'] - df['high'].shift(1)) < df['high'] * 0.002).fillna(False)
        high_level = (df['high'] > df['high'].rolling(10).mean()).fillna(False)
        bearish_second = (df['close'] < df['open']).fillna(False)
        
        return (high_similarity & high_level & bearish_second).astype(int)
    
    def _detect_tweezer_bottom(self, df):
        """Detect Tweezer Bottom pattern - FIXED"""
        low_similarity = (np.abs(df['low'] - df['low'].shift(1)) < df['low'] * 0.002).fillna(False)
        low_level = (df['low'] < df['low'].rolling(10).mean()).fillna(False)
        bullish_second = (df['close'] > df['open']).fillna(False)
        
        return (low_similarity & low_level & bullish_second).astype(int)
    
    def _detect_morning_star(self, df, body, body_abs):
        """Detect Morning Star pattern - FIXED"""
        first_bearish = (body.shift(2) < 0).fillna(False)
        second_small = (body_abs.shift(1) < body_abs.shift(2) * 0.3).fillna(False)
        third_bullish = (body > 0).fillna(False)
        third_closes_above = (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2).fillna(False)
        
        return (first_bearish & second_small & third_bullish & third_closes_above).astype(int)
    
    def _detect_evening_star(self, df, body, body_abs):
        """Detect Evening Star pattern - FIXED"""
        first_bullish = (body.shift(2) > 0).fillna(False)
        second_small = (body_abs.shift(1) < body_abs.shift(2) * 0.3).fillna(False)
        third_bearish = (body < 0).fillna(False)
        third_closes_below = (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2).fillna(False)
        
        return (first_bullish & second_small & third_bearish & third_closes_below).astype(int)
    
    def _detect_three_white_soldiers(self, df, body):
        """Detect Three White Soldiers pattern - FIXED"""
        cond1 = (body > 0).fillna(False)
        cond2 = (body.shift(1) > 0).fillna(False)
        cond3 = (body.shift(2) > 0).fillna(False)
        cond4 = (df['close'] > df['close'].shift(1)).fillna(False)
        cond5 = (df['close'].shift(1) > df['close'].shift(2)).fillna(False)
        cond6 = (df['open'] > df['open'].shift(1)).fillna(False)
        cond7 = (df['open'].shift(1) > df['open'].shift(2)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7).astype(int)
    
    def _detect_three_black_crows(self, df, body):
        """Detect Three Black Crows pattern - FIXED"""
        cond1 = (body < 0).fillna(False)
        cond2 = (body.shift(1) < 0).fillna(False)
        cond3 = (body.shift(2) < 0).fillna(False)
        cond4 = (df['close'] < df['close'].shift(1)).fillna(False)
        cond5 = (df['close'].shift(1) < df['close'].shift(2)).fillna(False)
        cond6 = (df['open'] < df['open'].shift(1)).fillna(False)
        cond7 = (df['open'].shift(1) < df['open'].shift(2)).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7).astype(int)
    
    def _detect_three_inside_up(self, df, body):
        """Detect Three Inside Up pattern - FIXED"""
        harami = self._detect_bullish_harami(df.shift(1), body.shift(1))
        third_confirms = ((body > 0) & (df['close'] > df['close'].shift(1))).fillna(False)
        # Fix: Convert shifted harami to boolean explicitly
        harami_shifted = harami.shift(1).fillna(0).astype(bool)
        return (harami_shifted & third_confirms).astype(int)
    
    def _detect_three_inside_down(self, df, body):
        """Detect Three Inside Down pattern - FIXED"""
        harami = self._detect_bearish_harami(df.shift(1), body.shift(1))
        third_confirms = ((body < 0) & (df['close'] < df['close'].shift(1))).fillna(False)
        # Fix: Convert shifted harami to boolean explicitly
        harami_shifted = harami.shift(1).fillna(0).astype(bool)
        return (harami_shifted & third_confirms).astype(int)
    
    # ================== CHART PATTERNS ==================
    
    def detect_chart_patterns(self, df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
        """
        Detect chart patterns (triangles, head and shoulders, etc.)
        """
        patterns = pd.DataFrame(index=df.index)
        
        # Find local extrema
        highs, lows = self._find_extrema(df, window)
        
        # Pattern detection
        patterns['triangle_ascending'] = self._detect_ascending_triangle(df, highs, lows, window)
        patterns['triangle_descending'] = self._detect_descending_triangle(df, highs, lows, window)
        patterns['triangle_symmetrical'] = self._detect_symmetrical_triangle(df, highs, lows, window)
        patterns['wedge_rising'] = self._detect_rising_wedge(df, highs, lows, window)
        patterns['wedge_falling'] = self._detect_falling_wedge(df, highs, lows, window)
        patterns['flag_bullish'] = self._detect_bullish_flag(df, window)
        patterns['flag_bearish'] = self._detect_bearish_flag(df, window)
        patterns['double_top'] = self._detect_double_top(df, highs, window)
        patterns['double_bottom'] = self._detect_double_bottom(df, lows, window)
        patterns['head_shoulders'] = self._detect_head_shoulders(df, highs, window)
        patterns['inverse_head_shoulders'] = self._detect_inverse_head_shoulders(df, lows, window)
        
        # Pattern breakout detection
        patterns['breakout_up'] = self._detect_breakout_up(df, window)
        patterns['breakout_down'] = self._detect_breakout_down(df, window)
        
        return patterns
    
    def _find_extrema(self, df: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series]:
        """Find local highs and lows"""
        highs = pd.Series(0, index=df.index)
        high_indices = argrelextrema(df['high'].values, np.greater, order=window//4)[0]
        highs.iloc[high_indices] = 1
        
        lows = pd.Series(0, index=df.index)
        low_indices = argrelextrema(df['low'].values, np.less, order=window//4)[0]
        lows.iloc[low_indices] = 1
        
        return highs, lows
    
    def _detect_ascending_triangle(self, df, highs, lows, window):
        """Detect Ascending Triangle pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            
            high_values = df['high'][window_slice][highs[window_slice] == 1]
            low_values = df['low'][window_slice][lows[window_slice] == 1]
            
            if len(high_values) >= 2 and len(low_values) >= 2:
                high_std = high_values.std() / high_values.mean()
                
                if len(low_values) >= 2:
                    low_slope = linregress(range(len(low_values)), low_values)[0]
                    
                    if high_std < 0.01 and low_slope > 0:
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_descending_triangle(self, df, highs, lows, window):
        """Detect Descending Triangle pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            
            high_values = df['high'][window_slice][highs[window_slice] == 1]
            low_values = df['low'][window_slice][lows[window_slice] == 1]
            
            if len(high_values) >= 2 and len(low_values) >= 2:
                low_std = low_values.std() / low_values.mean()
                
                if len(high_values) >= 2:
                    high_slope = linregress(range(len(high_values)), high_values)[0]
                    
                    if low_std < 0.01 and high_slope < 0:
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_symmetrical_triangle(self, df, highs, lows, window):
        """Detect Symmetrical Triangle pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            
            high_values = df['high'][window_slice][highs[window_slice] == 1]
            low_values = df['low'][window_slice][lows[window_slice] == 1]
            
            if len(high_values) >= 2 and len(low_values) >= 2:
                high_slope = linregress(range(len(high_values)), high_values)[0]
                low_slope = linregress(range(len(low_values)), low_values)[0]
                
                if high_slope < 0 and low_slope > 0:
                    if abs(high_slope / low_slope) < 2:
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_rising_wedge(self, df, highs, lows, window):
        """Detect Rising Wedge pattern (bearish)"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            
            high_values = df['high'][window_slice][highs[window_slice] == 1]
            low_values = df['low'][window_slice][lows[window_slice] == 1]
            
            if len(high_values) >= 3 and len(low_values) >= 3:
                high_slope = linregress(range(len(high_values)), high_values)[0]
                low_slope = linregress(range(len(low_values)), low_values)[0]
                
                if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
                    pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_falling_wedge(self, df, highs, lows, window):
        """Detect Falling Wedge pattern (bullish)"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            
            high_values = df['high'][window_slice][highs[window_slice] == 1]
            low_values = df['low'][window_slice][lows[window_slice] == 1]
            
            if len(high_values) >= 3 and len(low_values) >= 3:
                high_slope = linregress(range(len(high_values)), high_values)[0]
                low_slope = linregress(range(len(low_values)), low_values)[0]
                
                if high_slope < 0 and low_slope < 0 and low_slope < high_slope:
                    pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_bullish_flag(self, df, window):
        """Detect Bullish Flag pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            pre_return = (df['close'].iloc[i-window//2] - df['close'].iloc[i-window]) / df['close'].iloc[i-window]
            flag_return = (df['close'].iloc[i] - df['close'].iloc[i-window//2]) / df['close'].iloc[i-window//2]
            
            if pre_return > 0.1 and -0.05 < flag_return < 0.02:
                pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_bearish_flag(self, df, window):
        """Detect Bearish Flag pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            pre_return = (df['close'].iloc[i-window//2] - df['close'].iloc[i-window]) / df['close'].iloc[i-window]
            flag_return = (df['close'].iloc[i] - df['close'].iloc[i-window//2]) / df['close'].iloc[i-window//2]
            
            if pre_return < -0.1 and -0.02 < flag_return < 0.05:
                pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_double_top(self, df, highs, window):
        """Detect Double Top pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            high_indices = df.index[window_slice][highs[window_slice] == 1]
            
            if len(high_indices) >= 2:
                peak1_idx = high_indices[-2]
                peak2_idx = high_indices[-1]
                peak1_val = df['high'].loc[peak1_idx]
                peak2_val = df['high'].loc[peak2_idx]
                
                if abs(peak1_val - peak2_val) / peak1_val < 0.03:
                    valley = df['low'][peak1_idx:peak2_idx].min()
                    if valley < peak1_val * 0.95:
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_double_bottom(self, df, lows, window):
        """Detect Double Bottom pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            low_indices = df.index[window_slice][lows[window_slice] == 1]
            
            if len(low_indices) >= 2:
                trough1_idx = low_indices[-2]
                trough2_idx = low_indices[-1]
                trough1_val = df['low'].loc[trough1_idx]
                trough2_val = df['low'].loc[trough2_idx]
                
                if abs(trough1_val - trough2_val) / trough1_val < 0.03:
                    peak = df['high'][trough1_idx:trough2_idx].max()
                    if peak > trough1_val * 1.05:
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_head_shoulders(self, df, highs, window):
        """Detect Head and Shoulders pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            high_indices = df.index[window_slice][highs[window_slice] == 1]
            
            if len(high_indices) >= 3:
                peaks = [df['high'].loc[idx] for idx in high_indices[-3:]]
                
                if len(peaks) == 3:
                    left_shoulder, head, right_shoulder = peaks
                    
                    if (head > left_shoulder * 1.03 and 
                        head > right_shoulder * 1.03 and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_inverse_head_shoulders(self, df, lows, window):
        """Detect Inverse Head and Shoulders pattern"""
        pattern = pd.Series(0, index=df.index)
        
        for i in range(window, len(df)):
            window_slice = slice(i-window, i)
            low_indices = df.index[window_slice][lows[window_slice] == 1]
            
            if len(low_indices) >= 3:
                troughs = [df['low'].loc[idx] for idx in low_indices[-3:]]
                
                if len(troughs) == 3:
                    left_shoulder, head, right_shoulder = troughs
                    
                    if (head < left_shoulder * 0.97 and 
                        head < right_shoulder * 0.97 and
                        abs(left_shoulder - right_shoulder) / left_shoulder < 0.03):
                        pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_breakout_up(self, df, window):
        """Detect upward breakout - FIXED"""
        resistance = df['high'].rolling(window).max().shift(1)
        cond1 = (df['close'] > resistance).fillna(False)
        cond2 = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).fillna(False)
        return (cond1 & cond2).astype(int)
    
    def _detect_breakout_down(self, df, window):
        """Detect downward breakout - FIXED"""
        support = df['low'].rolling(window).min().shift(1)
        cond1 = (df['close'] < support).fillna(False)
        cond2 = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).fillna(False)
        return (cond1 & cond2).astype(int)
    
    # ================== SUPPORT AND RESISTANCE ==================
    
    def detect_support_resistance(self, df: pd.DataFrame, 
                                window: int = 50,
                                num_levels: int = 5) -> pd.DataFrame:
        """
        Detect support and resistance levels
        """
        features = pd.DataFrame(index=df.index)
        
        # Adjust window for small datasets
        actual_window = min(window, len(df) // 2)
        
        # Find price levels using multiple methods
        levels = self._find_sr_levels(df, actual_window, num_levels)
        
        # Initialize all columns with NaN first to ensure they exist
        for i in range(1, num_levels + 1):
            features[f'resistance_{i}'] = np.nan
            features[f'distance_to_resistance_{i}'] = np.nan
            features[f'support_{i}'] = np.nan
            features[f'distance_to_support_{i}'] = np.nan
        
        # Fill in actual resistance levels
        for i, level in enumerate(levels[:num_levels]):
            if i < len(levels):
                features[f'resistance_{i+1}'] = level
                features[f'distance_to_resistance_{i+1}'] = (level - df['close']) / df['close']
        
        # Support levels (below current price)
        support_levels = self._find_support_levels(df, actual_window, num_levels)
        for i, level in enumerate(support_levels[:num_levels]):
            if i < len(support_levels):
                features[f'support_{i+1}'] = level
                features[f'distance_to_support_{i+1}'] = (df['close'] - level) / df['close']
        
        # Strength of levels - handle case where no levels found
        if levels:
            features['nearest_resistance_strength'] = self._calculate_level_strength(df, levels[0])
        else:
            features['nearest_resistance_strength'] = 0.0
            
        if support_levels:
            features['nearest_support_strength'] = self._calculate_level_strength(df, support_levels[0])
        else:
            features['nearest_support_strength'] = 0.0
        
        # Price relative to levels - FIXED with safe column access
        if 'resistance_1' in features.columns:
            features['above_resistance'] = (df['close'] > features['resistance_1']).fillna(0).astype(int)
        else:
            features['above_resistance'] = 0
            
        if 'support_1' in features.columns:
            features['below_support'] = (df['close'] < features['support_1']).fillna(0).astype(int)
        else:
            features['below_support'] = 0
        
        return features
    
    def _find_sr_levels(self, df: pd.DataFrame, window: int, num_levels: int) -> List[float]:
        """Find support/resistance levels using price clustering"""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            # Fallback if sklearn not available
            high_low = pd.concat([df['high'], df['low']])
            return sorted(high_low.dropna().unique())[:num_levels]
        
        # Collect high and low points
        price_points = pd.concat([df['high'], df['low']])
        price_points = price_points.dropna().values.reshape(-1, 1)
        
        # Need at least num_levels points to cluster
        if len(price_points) < num_levels:
            # Just return sorted unique prices if not enough data
            unique_prices = sorted(set(price_points.flatten()))
            return unique_prices[:num_levels]
        
        # Cluster to find levels
        if len(price_points) > num_levels:
            kmeans = KMeans(n_clusters=min(num_levels, len(price_points)), random_state=42, n_init=10)
            kmeans.fit(price_points)
            levels = sorted(kmeans.cluster_centers_.flatten())
        else:
            levels = sorted(price_points.flatten())
        
        return levels
    
    def _find_support_levels(self, df: pd.DataFrame, window: int, num_levels: int) -> List[float]:
        """Find support levels below current price"""
        if len(df) < 2:
            return []
            
        current_price = df['close'].iloc[-1]
        
        # Find recent lows
        lows = []
        step = max(1, window // 5)  # Ensure step is at least 1
        
        for i in range(window, len(df), step):
            if i <= len(df):
                window_low = df['low'].iloc[max(0, i-window):i].min()
                if window_low < current_price:
                    lows.append(window_low)
        
        # If no lows found, use the minimum price as support
        if not lows and len(df) > 0:
            min_price = df['low'].min()
            if min_price < current_price:
                lows = [min_price]
        
        # Remove duplicates and sort
        lows = sorted(list(set(lows)), reverse=True)
        
        return lows[:num_levels]
    
    def _calculate_level_strength(self, df: pd.DataFrame, level: float, window: int = 20) -> float:
        """Calculate strength of a support/resistance level"""
        touches = 0
        for i in range(len(df)):
            if abs(df['high'].iloc[i] - level) / level < 0.005:
                touches += 1
            if abs(df['low'].iloc[i] - level) / level < 0.005:
                touches += 1
        
        return min(touches / window, 1.0)
    
    # ================== TREND PATTERNS (FIXED) ==================
    
    def detect_trend_patterns(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Detect trend-based patterns
        """
        features = pd.DataFrame(index=df.index)
        
        # Trend strength
        features['trend_strength'] = self._calculate_trend_strength(df, window)
        
        # Higher highs and lower lows
        features['higher_highs'] = self._detect_higher_highs(df, window)
        features['lower_lows'] = self._detect_lower_lows(df, window)
        features['higher_lows'] = self._detect_higher_lows(df, window)
        features['lower_highs'] = self._detect_lower_highs(df, window)
        
        # Trend classification - FIXED
        cond_up = features['higher_highs'].fillna(False) & features['higher_lows'].fillna(False)
        cond_down = features['lower_highs'].fillna(False) & features['lower_lows'].fillna(False)
        
        features['uptrend'] = cond_up.astype(int)
        features['downtrend'] = cond_down.astype(int)
        features['sideways'] = (~cond_up & ~cond_down).astype(int)
        
        # Trend lines
        features['above_trendline'] = self._detect_above_trendline(df, window)
        features['below_trendline'] = self._detect_below_trendline(df, window)
        
        # Trend reversals
        features['trend_reversal_up'] = self._detect_trend_reversal_up(df, features)
        features['trend_reversal_down'] = self._detect_trend_reversal_down(df, features)
        
        return features
    
    def _calculate_trend_strength(self, df, window):
        """Calculate trend strength using linear regression"""
        trend_strength = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            y = df['close'].iloc[i-window:i].values
            x = np.arange(window)
            
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Normalize slope by price level
            normalized_slope = slope / df['close'].iloc[i]
            trend_strength.iloc[i] = normalized_slope * r_value**2
        
        return trend_strength
    
    def _detect_higher_highs(self, df, window):
        """Detect higher highs pattern - FIXED"""
        rolling_high = df['high'].rolling(window).max()
        return (rolling_high > rolling_high.shift(window)).fillna(0).astype(int)
    
    def _detect_lower_lows(self, df, window):
        """Detect lower lows pattern - FIXED"""
        rolling_low = df['low'].rolling(window).min()
        return (rolling_low < rolling_low.shift(window)).fillna(0).astype(int)
    
    def _detect_higher_lows(self, df, window):
        """Detect higher lows pattern - FIXED"""
        rolling_low = df['low'].rolling(window).min()
        return (rolling_low > rolling_low.shift(window)).fillna(0).astype(int)
    
    def _detect_lower_highs(self, df, window):
        """Detect lower highs pattern - FIXED"""
        rolling_high = df['high'].rolling(window).max()
        return (rolling_high < rolling_high.shift(window)).fillna(0).astype(int)
    
    def _detect_above_trendline(self, df, window):
        """Detect if price is above trendline - FIXED"""
        trendline = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            y = df['close'].iloc[i-window:i].values
            x = np.arange(window)
            
            slope, intercept = linregress(x, y)[:2]
            trendline.iloc[i] = slope * window + intercept
        
        return (df['close'] > trendline).fillna(0).astype(int)
    
    def _detect_below_trendline(self, df, window):
        """Detect if price is below trendline - FIXED"""
        trendline = pd.Series(index=df.index, dtype=float)
        
        for i in range(window, len(df)):
            y = df['close'].iloc[i-window:i].values
            x = np.arange(window)
            
            slope, intercept = linregress(x, y)[:2]
            trendline.iloc[i] = slope * window + intercept
        
        return (df['close'] < trendline).fillna(0).astype(int)
    
    def _detect_trend_reversal_up(self, df, features):
        """Detect upward trend reversal - FIXED"""
        cond1 = features['downtrend'].shift(1).fillna(0).astype(bool)
        cond2 = ~features['downtrend'].fillna(0).astype(bool)
        cond3 = (df['close'] > df['close'].shift(1)).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    def _detect_trend_reversal_down(self, df, features):
        """Detect downward trend reversal - FIXED"""
        cond1 = features['uptrend'].shift(1).fillna(0).astype(bool)
        cond2 = ~features['uptrend'].fillna(0).astype(bool)
        cond3 = (df['close'] < df['close'].shift(1)).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    # ================== VOLUME PATTERNS (FIXED) ==================
    
    def detect_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect volume-based patterns
        """
        features = pd.DataFrame(index=df.index)
        
        # Volume trends
        vol_ma = df['volume'].rolling(20).mean()
        
        # Volume spikes - FIXED
        features['volume_spike'] = (df['volume'] > vol_ma * 2).fillna(0).astype(int)
        features['volume_dry_up'] = (df['volume'] < vol_ma * 0.5).fillna(0).astype(int)
        
        # Volume divergence - FIXED
        price_change = df['close'].pct_change(5, fill_method=None)  # Fix deprecation warning
        volume_change = df['volume'].pct_change(5, fill_method=None)  # Fix deprecation warning
        
        cond1 = ((price_change > 0.05) & (volume_change < -0.2)).fillna(False)
        cond2 = ((price_change < -0.05) & (volume_change < -0.2)).fillna(False)
        
        features['volume_price_divergence'] = (cond1 | cond2).astype(int)
        
        # Accumulation/Distribution patterns - FIXED
        cond_acc = ((df['close'] > df['open']) & (df['volume'] > vol_ma)).fillna(False)
        cond_dist = ((df['close'] < df['open']) & (df['volume'] > vol_ma)).fillna(False)
        
        features['accumulation'] = cond_acc.astype(int)
        features['distribution'] = cond_dist.astype(int)
        
        # Volume at price levels
        features['high_volume_node'] = self._detect_high_volume_nodes(df)
        
        return features
    
    def _detect_high_volume_nodes(self, df, bins: int = 20):
        """Detect high volume price levels - FIXED to handle NaN values"""
        # Check if we have valid price data
        valid_prices = df['close'].dropna()
        
        if len(valid_prices) < 2:
            # Not enough valid data to create bins
            return pd.Series(0, index=df.index)
        
        # Check if all prices are the same (would cause binning error)
        if valid_prices.nunique() == 1:
            # All prices are the same, no meaningful volume nodes
            return pd.Series(0, index=df.index)
        
        try:
            # Create price bins only with valid prices
            min_price = valid_prices.min()
            max_price = valid_prices.max()
            
            # Create bins with a small buffer to avoid edge issues
            buffer = (max_price - min_price) * 0.001
            bin_edges = np.linspace(min_price - buffer, max_price + buffer, bins + 1)
            
            # Ensure unique bin edges (handle case where price range is very small)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                return pd.Series(0, index=df.index)
            
            # Create price bins
            price_bins = pd.cut(df['close'], bins=bin_edges, duplicates='drop')
            
            # Calculate volume at each price level, explicitly handling observed parameter
            volume_profile = df.groupby(price_bins, observed=True)['volume'].sum()
            
            # Find high volume nodes (above 75th percentile)
            if len(volume_profile) > 0:
                high_volume_threshold = volume_profile.quantile(0.75)
            else:
                return pd.Series(0, index=df.index)
            
            # Map back to original data
            features = pd.Series(0, index=df.index)
            for i, price in enumerate(df['close']):
                if pd.notna(price):  # Only process non-NaN prices
                    price_bin = pd.cut([price], bins=bin_edges, duplicates='drop')[0]
                    if pd.notna(price_bin) and price_bin in volume_profile.index:
                        if volume_profile[price_bin] > high_volume_threshold:
                            features.iloc[i] = 1
            
            return features
            
        except Exception as e:
            # If any error occurs, return zeros rather than failing
            print(f"Warning in _detect_high_volume_nodes: {e}")
            return pd.Series(0, index=df.index)
    
    # ================== PRICE ACTION PATTERNS (FIXED) ==================
    
    def detect_price_action_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect price action patterns
        """
        features = pd.DataFrame(index=df.index)
        
        # Pin bars
        features['pin_bar_bullish'] = self._detect_pin_bar_bullish(df)
        features['pin_bar_bearish'] = self._detect_pin_bar_bearish(df)
        
        # Inside and outside bars
        features['inside_bar'] = self._detect_inside_bar(df)
        features['outside_bar'] = self._detect_outside_bar(df)
        
        # Price rejection
        features['rejection_upper'] = self._detect_upper_rejection(df)
        features['rejection_lower'] = self._detect_lower_rejection(df)
        
        # Momentum patterns
        features['momentum_bullish'] = self._detect_bullish_momentum(df)
        features['momentum_bearish'] = self._detect_bearish_momentum(df)
        
        # Gap patterns - FIXED
        features['gap_up'] = (df['open'] > df['close'].shift(1) * 1.002).fillna(0).astype(int)
        features['gap_down'] = (df['open'] < df['close'].shift(1) * 0.998).fillna(0).astype(int)
        features['gap_filled'] = self._detect_gap_fill(df)
        
        return features
    
    def _detect_pin_bar_bullish(self, df):
        """Detect bullish pin bar - FIXED"""
        body = np.abs(df['close'] - df['open'])
        lower_wick = np.minimum(df['close'], df['open']) - df['low']
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        
        cond1 = (lower_wick > body * 2).fillna(False)
        cond2 = (lower_wick > upper_wick * 2).fillna(False)
        cond3 = (df['close'] > df['open']).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    def _detect_pin_bar_bearish(self, df):
        """Detect bearish pin bar - FIXED"""
        body = np.abs(df['close'] - df['open'])
        lower_wick = np.minimum(df['close'], df['open']) - df['low']
        upper_wick = df['high'] - np.maximum(df['close'], df['open'])
        
        cond1 = (upper_wick > body * 2).fillna(False)
        cond2 = (upper_wick > lower_wick * 2).fillna(False)
        cond3 = (df['close'] < df['open']).fillna(False)
        
        return (cond1 & cond2 & cond3).astype(int)
    
    def _detect_inside_bar(self, df):
        """Detect inside bar pattern - FIXED"""
        cond1 = (df['high'] < df['high'].shift(1)).fillna(False)
        cond2 = (df['low'] > df['low'].shift(1)).fillna(False)
        
        return (cond1 & cond2).astype(int)
    
    def _detect_outside_bar(self, df):
        """Detect outside bar pattern - FIXED"""
        cond1 = (df['high'] > df['high'].shift(1)).fillna(False)
        cond2 = (df['low'] < df['low'].shift(1)).fillna(False)
        
        return (cond1 & cond2).astype(int)
    
    def _detect_upper_rejection(self, df):
        """Detect rejection from upper levels - FIXED"""
        cond1 = (df['high'] > df['high'].rolling(10).max().shift(1)).fillna(False)
        cond2 = (df['close'] < (df['high'] + df['low']) / 2).fillna(False)
        
        return (cond1 & cond2).astype(int)
    
    def _detect_lower_rejection(self, df):
        """Detect rejection from lower levels - FIXED"""
        cond1 = (df['low'] < df['low'].rolling(10).min().shift(1)).fillna(False)
        cond2 = (df['close'] > (df['high'] + df['low']) / 2).fillna(False)
        
        return (cond1 & cond2).astype(int)
    
    def _detect_bullish_momentum(self, df):
        """Detect strong bullish momentum - FIXED"""
        cond1 = (df['close'] > df['open']).fillna(False)
        cond2 = (df['close'].shift(1) > df['open'].shift(1)).fillna(False)
        cond3 = (df['close'] > df['close'].shift(1)).fillna(False)
        cond4 = (df['volume'] > df['volume'].rolling(20).mean()).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4).astype(int)
    
    def _detect_bearish_momentum(self, df):
        """Detect strong bearish momentum - FIXED"""
        cond1 = (df['close'] < df['open']).fillna(False)
        cond2 = (df['close'].shift(1) < df['open'].shift(1)).fillna(False)
        cond3 = (df['close'] < df['close'].shift(1)).fillna(False)
        cond4 = (df['volume'] > df['volume'].rolling(20).mean()).fillna(False)
        
        return (cond1 & cond2 & cond3 & cond4).astype(int)
    
    def _detect_gap_fill(self, df):
        """Detect if previous gap has been filled"""
        gap_up = df['open'] > df['close'].shift(1)
        gap_down = df['open'] < df['close'].shift(1)
        
        filled = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            if gap_up.iloc[i-1]:
                # Check if gap up was filled
                if df['low'].iloc[i] <= df['close'].iloc[i-2]:
                    filled.iloc[i] = 1
            elif gap_down.iloc[i-1]:
                # Check if gap down was filled
                if df['high'].iloc[i] >= df['close'].iloc[i-2]:
                    filled.iloc[i] = 1
        
        return filled
        