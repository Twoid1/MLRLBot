"""
Technical Indicators Module
Implements all technical indicators for feature engineering
No external TA library dependencies - everything from scratch
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


class TechnicalIndicators:
    """
    Technical indicators calculator - all implemented from scratch
    Designed for OHLCV data with no external dependencies
    """
    
    # ================== MOMENTUM INDICATORS ==================
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        Range: 0-100, Overbought >70, Oversold <30
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(close: pd.Series, 
             fast: int = 12, 
             slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD - Moving Average Convergence Divergence
        Returns: (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: pd.Series, 
                   low: pd.Series, 
                   close: pd.Series, 
                   period: int = 14,
                   smooth_k: int = 3,
                   smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        Returns: (%K, %D)
        Range: 0-100
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, 1e-10)
        
        k_percent = 100 * ((close - lowest_low) / denominator)
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, 
                   low: pd.Series, 
                   close: pd.Series, 
                   period: int = 14) -> pd.Series:
        """
        Williams %R
        Range: -100 to 0, Overbought > -20, Oversold < -80
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # Avoid division by zero
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, 1e-10)
        
        williams_r = -100 * ((highest_high - close) / denominator)
        
        return williams_r
    
    @staticmethod
    def cci(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        Range: Typically -100 to +100, but can exceed
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Avoid division by zero
        mad = mad.replace(0, 1e-10)
        
        cci = (typical_price - sma) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def roc(close: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change
        Percentage change over n periods
        """
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        return roc
    
    @staticmethod
    def mfi(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            volume: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Money Flow Index - Volume-weighted RSI
        Range: 0-100
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        money_flow_ratio = pd.Series(index=close.index, dtype=float)
        
        for i in range(period, len(typical_price)):
            period_slice = slice(i - period, i)
            
            positive_flow = 0
            negative_flow = 0
            
            for j in range(i - period + 1, i):
                if typical_price.iloc[j] > typical_price.iloc[j - 1]:
                    positive_flow += raw_money_flow.iloc[j]
                elif typical_price.iloc[j] < typical_price.iloc[j - 1]:
                    negative_flow += raw_money_flow.iloc[j]
            
            # Avoid division by zero
            if negative_flow == 0:
                money_flow_ratio.iloc[i] = 100
            else:
                mf_ratio = positive_flow / negative_flow
                money_flow_ratio.iloc[i] = 100 - (100 / (1 + mf_ratio))
        
        return money_flow_ratio
    
    # ================== VOLATILITY INDICATORS ==================
    
    @staticmethod
    def atr(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Average True Range
        Measures volatility
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def bollinger_bands(close: pd.Series, 
                        period: int = 20, 
                        std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        Returns: (upper_band, middle_band, lower_band)
        """
        middle_band = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def keltner_channels(high: pd.Series, 
                         low: pd.Series, 
                         close: pd.Series, 
                         period: int = 20, 
                         multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels
        Returns: (upper_channel, middle_line, lower_channel)
        """
        middle_line = close.ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)
        
        return upper_channel, middle_line, lower_channel
    
    @staticmethod
    def donchian_channels(high: pd.Series, 
                          low: pd.Series, 
                          period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channels
        Returns: (upper_channel, middle_channel, lower_channel)
        """
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, middle_channel, lower_channel
    
    # ================== TREND INDICATORS ==================
    
    @staticmethod
    def adx(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            period: int = 14) -> pd.Series:
        """
        Average Directional Index
        Measures trend strength (0-100)
        """
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When both are positive, keep only the larger
        mask = (plus_dm > 0) & (minus_dm > 0)
        plus_dm[mask & (plus_dm < minus_dm)] = 0
        minus_dm[mask & (minus_dm < plus_dm)] = 0
        
        # Calculate ATR
        atr = TechnicalIndicators.atr(high, low, close, period)
        
        # Smooth the directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def aroon(high: pd.Series, 
              low: pd.Series, 
              period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Aroon Indicator
        Returns: (aroon_up, aroon_down, aroon_oscillator)
        """
        aroon_up = high.rolling(window=period + 1).apply(
            lambda x: (period - x.argmax()) / period * 100, raw=True
        )
        aroon_down = low.rolling(window=period + 1).apply(
            lambda x: (period - x.argmin()) / period * 100, raw=True
        )
        aroon_oscillator = aroon_up - aroon_down
        
        return aroon_up, aroon_down, aroon_oscillator
    
    @staticmethod
    def psar(high: pd.Series, 
             low: pd.Series, 
             acceleration: float = 0.02, 
             maximum: float = 0.2) -> pd.Series:
        """
        Parabolic SAR
        Stop and Reverse indicator
        """
        psar = close = (high + low) / 2
        psar_values = []
        bull = True
        af = acceleration
        ep = high.iloc[0] if bull else low.iloc[0]
        hp = high.iloc[0]
        lp = low.iloc[0]
        
        for i in range(len(high)):
            if bull:
                psar_value = psar.iloc[i-1] + af * (ep - psar.iloc[i-1]) if i > 0 else low.iloc[0]
                
                if low.iloc[i] < psar_value:
                    bull = False
                    psar_value = hp
                    ep = low.iloc[i]
                    af = acceleration
                    lp = low.iloc[i]
                    hp = high.iloc[i]
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + acceleration, maximum)
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
            else:
                psar_value = psar.iloc[i-1] + af * (ep - psar.iloc[i-1]) if i > 0 else high.iloc[0]
                
                if high.iloc[i] > psar_value:
                    bull = True
                    psar_value = lp
                    ep = high.iloc[i]
                    af = acceleration
                    hp = high.iloc[i]
                    lp = low.iloc[i]
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + acceleration, maximum)
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
            
            psar_values.append(psar_value)
        
        return pd.Series(psar_values, index=high.index)
    
    @staticmethod
    def ichimoku(high: pd.Series, 
                 low: pd.Series, 
                 close: pd.Series,
                 conversion: int = 9,
                 base: int = 26,
                 span_b: int = 52,
                 displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Ichimoku Cloud
        Returns dictionary with all Ichimoku components
        """
        # Conversion Line (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion).max() + 
                          low.rolling(window=conversion).min()) / 2
        
        # Base Line (Kijun-sen)
        base_line = (high.rolling(window=base).max() + 
                    low.rolling(window=base).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=span_b).max() + 
                          low.rolling(window=span_b).min()) / 2).shift(displacement)
        
        # Lagging Span (Chikou Span)
        lagging_span = None
        
        return {
            'conversion_line': conversion_line,
            'base_line': base_line,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'lagging_span': lagging_span
        }
    
    # ================== VOLUME INDICATORS ==================
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume
        Cumulative volume based on price direction
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def ad_line(high: pd.Series, 
                low: pd.Series, 
                close: pd.Series, 
                volume: pd.Series) -> pd.Series:
        """
        Accumulation/Distribution Line
        """
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        
        return ad_line
    
    @staticmethod
    def vwap(high: pd.Series, 
             low: pd.Series, 
             close: pd.Series, 
             volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (high + low + close) / 3
        
        # Calculate cumulative values for the day
        cumulative_tpv = (typical_price * volume).cumsum()
        cumulative_volume = volume.cumsum()
        
        vwap = cumulative_tpv / (cumulative_volume + 1e-10)
        
        return vwap
    
    @staticmethod
    def volume_sma_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume to SMA ratio
        Shows if current volume is above/below average
        """
        volume_sma = volume.rolling(window=period).mean()
        ratio = volume / (volume_sma + 1e-10)
        
        return ratio
    
    @staticmethod
    def cmf(high: pd.Series, 
            low: pd.Series, 
            close: pd.Series, 
            volume: pd.Series, 
            period: int = 20) -> pd.Series:
        """
        Chaikin Money Flow
        Range: -1 to +1
        """
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        mfv = mfm * volume
        
        cmf = mfv.rolling(window=period).sum() / (volume.rolling(window=period).sum() + 1e-10)
        
        return cmf
    
    @staticmethod
    def fi(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
        """
        Force Index
        Combines price and volume
        """
        fi = (close.diff() * volume).ewm(span=period, adjust=False).mean()
        return fi
    
    # ================== HELPER METHODS ==================
    
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators at once
        Returns DataFrame with all indicator values
        """
        indicators = pd.DataFrame(index=df.index)
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must contain columns: {required}")
        
        # Momentum indicators
        indicators['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        indicators['rsi_21'] = TechnicalIndicators.rsi(df['close'], 21)
        
        macd, signal, hist = TechnicalIndicators.macd(df['close'])
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = hist
        
        k, d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        indicators['stoch_k'] = k
        indicators['stoch_d'] = d
        
        indicators['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        indicators['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        indicators['roc'] = TechnicalIndicators.roc(df['close'])
        indicators['mfi'] = TechnicalIndicators.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Volatility indicators
        indicators['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
        
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = bb_upper - bb_lower
        indicators['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        
        kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(
            df['high'], df['low'], df['close']
        )
        indicators['kc_upper'] = kc_upper
        indicators['kc_middle'] = kc_middle
        indicators['kc_lower'] = kc_lower
        
        dc_upper, dc_middle, dc_lower = TechnicalIndicators.donchian_channels(
            df['high'], df['low']
        )
        indicators['dc_upper'] = dc_upper
        indicators['dc_middle'] = dc_middle
        indicators['dc_lower'] = dc_lower
        
        # Trend indicators
        indicators['adx'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        
        aroon_up, aroon_down, aroon_osc = TechnicalIndicators.aroon(df['high'], df['low'])
        indicators['aroon_up'] = aroon_up
        indicators['aroon_down'] = aroon_down
        indicators['aroon_oscillator'] = aroon_osc
        
        indicators['psar'] = TechnicalIndicators.psar(df['high'], df['low'])
        
        # Volume indicators
        indicators['obv'] = TechnicalIndicators.obv(df['close'], df['volume'])
        indicators['ad_line'] = TechnicalIndicators.ad_line(
            df['high'], df['low'], df['close'], df['volume']
        )
        indicators['vwap'] = TechnicalIndicators.vwap(
            df['high'], df['low'], df['close'], df['volume']
        )
        indicators['volume_sma_ratio'] = TechnicalIndicators.volume_sma_ratio(df['volume'])
        indicators['cmf'] = TechnicalIndicators.cmf(
            df['high'], df['low'], df['close'], df['volume']
        )
        indicators['fi'] = TechnicalIndicators.fi(df['close'], df['volume'])
        
        return indicators
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, 
                            periods: list = [10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        Add multiple moving averages to DataFrame
        """
        ma_df = pd.DataFrame(index=df.index)
        
        for period in periods:
            ma_df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            ma_df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return ma_df
    
    @staticmethod
    def normalize_indicators(indicators: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize indicators to 0-1 range where appropriate
        """
        normalized = indicators.copy()
        
        # List of indicators that should be normalized
        to_normalize = ['rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 
                       'bb_position', 'aroon_up', 'aroon_down']
        
        for col in to_normalize:
            if col in normalized.columns:
                # These are already 0-100, so divide by 100
                normalized[col] = normalized[col] / 100
        
        # Williams %R is -100 to 0, normalize to 0-1
        if 'williams_r' in normalized.columns:
            normalized['williams_r'] = (normalized['williams_r'] + 100) / 100
        
        # CCI can be unbounded, clip and normalize
        if 'cci' in normalized.columns:
            normalized['cci'] = np.clip(normalized['cci'], -200, 200)
            normalized['cci'] = (normalized['cci'] + 200) / 400
        
        # MFI is 0-100
        if 'mfi' in normalized.columns:
            normalized['mfi'] = normalized['mfi'] / 100
        
        # CMF is -1 to 1, normalize to 0-1
        if 'cmf' in normalized.columns:
            normalized['cmf'] = (normalized['cmf'] + 1) / 2
        
        return normalized