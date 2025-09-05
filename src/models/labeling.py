"""
Labeling Module for Financial Machine Learning
Implements various labeling methods for supervised learning in trading
Including triple-barrier, fixed-time, and meta-labeling methods
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class LabelingConfig:
    """Configuration for labeling methods"""
    method: str = 'triple_barrier'  # 'triple_barrier', 'fixed_time', 'simple_returns'
    lookforward: int = 10  # Periods to look forward
    vol_window: int = 50  # Window for volatility calculation
    pt_sl: List[float] = None  # Profit-taking and stop-loss multipliers
    min_ret: float = 0.0001  # Minimum return to consider
    threshold: float = 0.02  # Threshold for simple returns
    num_classes: int = 3  # Number of classes (3 for down/flat/up)
    
    def __post_init__(self):
        if self.pt_sl is None:
            self.pt_sl = [1.0, 1.0]  # Default symmetric barriers


class TripleBarrierLabeler:
    """
    Implements triple-barrier labeling for financial time series
    Superior to simple returns-based labeling for ML trading
    
    Based on "Advances in Financial Machine Learning" by Marcos Lopez de Prado
    """
    
    def __init__(self,
                 lookback: int = 20,
                 vol_span: int = 50,
                 min_ret: float = 0.0001):
        """
        Initialize triple-barrier labeler
        
        Args:
            lookback: Lookback period for volatility calculation
            vol_span: Span for exponential weighted volatility
            min_ret: Minimum return to consider non-zero
        """
        self.lookback = lookback
        self.vol_span = vol_span
        self.min_ret = min_ret
        
    def apply_triple_barrier(self,
                            prices: pd.Series,
                            events: pd.DataFrame,
                            pt_sl: Union[float, List[float], pd.Series] = None,
                            molecule: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Apply triple-barrier method
        
        Args:
            prices: Price series
            events: DataFrame with columns [t1, trgt, side] 
                   t1: vertical barrier (time limit)
                   trgt: horizontal barrier width
                   side: position side (1: long, -1: short)
            pt_sl: Profit-taking and stop-loss multiples [pt, sl]
            molecule: Subset of events to process (for parallel processing)
            
        Returns:
            DataFrame with first touch times and labels
        """
        if molecule is None:
            molecule = events.index
        
        # Filter events
        events = events.loc[molecule]
        out = pd.DataFrame(index=molecule, columns=['t1', 'sl', 'pt'])
        
        if pt_sl is None:
            pt_sl = [1, 1]  # Default symmetric barriers
        
        if isinstance(pt_sl, float):
            pt_sl = [pt_sl, pt_sl]
        
        for loc, t1 in events['t1'].items():
            if pd.isna(t1):
                continue
                
            # Get price path from event to vertical barrier
            price_path = prices[loc:t1]
            
            if len(price_path) < 2:
                continue
            
            # Calculate returns
            returns = price_path / prices[loc] - 1
            
            # Apply horizontal barriers
            trgt = events.loc[loc, 'trgt']
            side = events.loc[loc, 'side'] if 'side' in events.columns else 1
            
            # Profit-taking barrier
            pt_level = trgt * pt_sl[0] * side
            # Stop-loss barrier  
            sl_level = -trgt * pt_sl[1] * side
            
            # Find first touches
            pt_touch = returns[returns > pt_level].index.min() if (returns > pt_level).any() else pd.NaT
            sl_touch = returns[returns < sl_level].index.min() if (returns < sl_level).any() else pd.NaT
            
            # Store results
            out.loc[loc, 't1'] = t1
            out.loc[loc, 'pt'] = pt_touch
            out.loc[loc, 'sl'] = sl_touch
        
        return out
    
    def get_events(self,
                prices: pd.Series,
                t_events: pd.DatetimeIndex,
                num_periods: int = 5,  # Changed from num_days to num_periods
                vol_window: int = 50,
                min_ret: float = 0.0,
                side: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Create events for triple-barrier labeling (FIXED VERSION)
        
        Args:
            prices: Price series
            t_events: Timestamps of events (e.g., when predictions are made)
            num_periods: Number of periods for vertical barrier (not days!)
            vol_window: Window for volatility calculation
            min_ret: Minimum return filter
            side: Position side series (1: long, -1: short, None: always long)
            
        Returns:
            Events DataFrame with barriers
        """
        # Calculate volatility
        returns = prices.pct_change()
        vol = returns.ewm(span=vol_window).std()
        
        # Create events DataFrame
        events = pd.DataFrame(index=t_events, columns=['t1', 'trgt', 'side'])
        
        for i, t in enumerate(t_events):
            # Skip if too close to the end
            if i + num_periods >= len(prices):
                continue
            
            # Vertical barrier (time limit) - use index position, not time delta
            try:
                t_idx = prices.index.get_loc(t)
                t1_idx = min(t_idx + num_periods, len(prices) - 1)
                t1 = prices.index[t1_idx]
            except:
                continue
            
            # Horizontal barrier width (based on volatility)
            if t in vol.index:
                trgt = vol.loc[t]
            else:
                trgt = vol.mean() if not vol.empty else self.min_ret
            
            # Skip if volatility is too low or NaN
            if pd.isna(trgt) or trgt < min_ret:
                continue
            
            events.loc[t, 't1'] = t1
            events.loc[t, 'trgt'] = trgt
            events.loc[t, 'side'] = side.loc[t] if side is not None and t in side.index else 1
        
        # Remove NaN events
        events = events.dropna()
        
        return events
    
    def get_labels(self,
                  prices: pd.Series,
                  events: pd.DataFrame,
                  barriers: pd.DataFrame) -> pd.Series:
        """
        Generate labels from barrier touches (FIXED VERSION)
        
        Args:
            prices: Price series
            events: Events DataFrame
            barriers: Barrier touch times from apply_triple_barrier
            
        Returns:
            Series with labels (-1: down, 0: flat/timeout, 1: up)
        """
        labels = pd.Series(index=barriers.index, dtype=float)
        
        for idx in barriers.index:
            # Get touch times
            t1 = barriers.loc[idx, 't1']
            pt = barriers.loc[idx, 'pt']
            sl = barriers.loc[idx, 'sl']
            
            # Create series of touch times and filter NaT
            touches = pd.Series({'t1': t1, 'pt': pt, 'sl': sl}).dropna()
            
            if len(touches) == 0:
                labels[idx] = 0  # No barrier touched
                continue
            
            # Determine which barrier was touched first
            first_touch_name = touches.idxmin()
            first_touch_time = touches.min()
            
            if first_touch_name == 'pt':
                labels[idx] = 1  # Profit-taking (up)
            elif first_touch_name == 'sl':
                labels[idx] = -1  # Stop-loss (down)
            else:  # t1 (vertical barrier)
                # Check final return at vertical barrier
                if idx in prices.index and first_touch_time in prices.index:
                    ret = prices[first_touch_time] / prices[idx] - 1
                    if ret > 0.001:
                        labels[idx] = 1
                    elif ret < -0.001:
                        labels[idx] = -1
                    else:
                        labels[idx] = 0
                else:
                    labels[idx] = 0
        
        return labels
    
    def label_data(self,
                df: pd.DataFrame,
                price_col: str = 'close',
                lookforward: int = 10,
                vol_window: int = 50,
                pt_sl: List[float] = None) -> pd.Series:
        """
        Complete labeling pipeline (IMPROVED VERSION)
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column to use for prices
            lookforward: Number of periods to look forward
            vol_window: Window for volatility calculation
            pt_sl: Profit-taking and stop-loss multipliers
            
        Returns:
            Series with labels
        """
        if pt_sl is None:
            pt_sl = [2, 2]
            
        prices = df[price_col]
        
        # Ensure we have enough data
        min_required = lookforward + vol_window
        if len(prices) < min_required:
            logger.warning(f"Not enough data for labeling. Need at least {min_required} points, got {len(prices)}")
            return pd.Series(0, index=df.index)
        
        # Don't try to label data too close to the end
        safe_end = max(1, len(prices) - lookforward - 1)
        
        # Create events at safe timestamps
        t_events = prices.index[:safe_end]
        
        # Get events with barriers (using periods, not days)
        events = self.get_events(
            prices=prices,
            t_events=t_events,
            num_periods=lookforward,  # Use periods instead of days
            vol_window=vol_window
        )
        
        # If no valid events, return zeros
        if len(events) == 0:
            logger.warning("No valid events created for labeling")
            return pd.Series(0, index=df.index)
        
        # Apply triple barrier
        barriers = self.apply_triple_barrier(
            prices=prices,
            events=events,
            pt_sl=pt_sl
        )
        
        # Get labels
        labels = self.get_labels(prices, events, barriers)
        
        # Convert to categorical: -1, 0, 1
        labels = labels.round().astype(int)
        
        # Reindex to match original DataFrame
        labels = labels.reindex(df.index, fill_value=0)
        
        return labels


class FixedTimeLabeler:
    """
    Fixed-time horizon labeling
    Labels based on returns over a fixed time period
    """
    
    def __init__(self, lookforward: int = 20, threshold: float = 0.02):
        """
        Initialize fixed-time labeler
        
        Args:
            lookforward: Number of periods to look forward
            threshold: Return threshold for classification
        """
        self.lookforward = lookforward
        self.threshold = threshold
    
    def label_data(self, 
                  df: pd.DataFrame,
                  price_col: str = 'close',
                  num_classes: int = 3) -> pd.Series:
        """
        Generate labels based on fixed-time returns
        
        Args:
            df: DataFrame with price data
            price_col: Column to use for prices
            num_classes: Number of classes (2 for binary, 3 for ternary)
            
        Returns:
            Series with labels
        """
        prices = df[price_col]
        
        # Calculate forward returns
        returns = prices.pct_change(self.lookforward).shift(-self.lookforward)
        
        if num_classes == 2:
            # Binary classification
            labels = (returns > 0).astype(int)
        elif num_classes == 3:
            # Ternary classification
            labels = pd.Series(0, index=df.index)
            labels[returns > self.threshold] = 1  # Up
            labels[returns < -self.threshold] = -1  # Down
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")
        
        # Fill NaN with 0
        labels = labels.fillna(0).astype(int)
        
        return labels


class TrendLabeler:
    """
    Trend-based labeling
    Labels based on local trends and patterns
    """
    
    def __init__(self, window: int = 20, min_slope: float = 0.001):
        """
        Initialize trend labeler
        
        Args:
            window: Window for trend calculation
            min_slope: Minimum slope to consider a trend
        """
        self.window = window
        self.min_slope = min_slope
    
    def label_data(self,
                  df: pd.DataFrame,
                  price_col: str = 'close') -> pd.Series:
        """
        Generate labels based on trend detection (IMPROVED)
        
        Args:
            df: DataFrame with price data
            price_col: Column to use for prices
            
        Returns:
            Series with trend labels
        """
        prices = df[price_col]
        
        # Calculate moving averages
        ma_short = prices.rolling(self.window // 2, min_periods=1).mean()
        ma_long = prices.rolling(self.window, min_periods=1).mean()
        
        # Calculate trend slope using linear regression
        labels = pd.Series(0, index=df.index)
        
        for i in range(self.window, len(prices)):
            window_prices = prices.iloc[i-self.window:i]
            x = np.arange(len(window_prices))
            y = window_prices.values
            
            # Linear regression to get slope
            if len(y) > 1:
                slope = np.polyfit(x, y, 1)[0] / np.mean(y)  # Normalize by mean
                
                if slope > self.min_slope:
                    labels.iloc[i] = 1  # Uptrend
                elif slope < -self.min_slope:
                    labels.iloc[i] = -1  # Downtrend
        
        return labels.astype(int)


class MetaLabeler:
    """
    Meta-labeling for position sizing
    Predicts the probability of correct primary prediction
    """
    
    def __init__(self):
        """Initialize meta-labeler"""
        self.triple_barrier = TripleBarrierLabeler()
    
    def get_meta_labels(self,
                       prices: pd.Series,
                       primary_labels: pd.Series,
                       events: pd.DataFrame,
                       pt_sl: List[float] = None) -> pd.Series:
        """
        Generate meta-labels (IMPROVED)
        
        Args:
            prices: Price series
            primary_labels: Primary model predictions
            events: Events DataFrame with barriers
            pt_sl: Profit-taking and stop-loss multipliers
            
        Returns:
            Series with meta-labels (0: wrong, 1: correct)
        """
        if pt_sl is None:
            pt_sl = [1, 1]
        
        # Apply triple barrier
        barriers = self.triple_barrier.apply_triple_barrier(
            prices=prices,
            events=events,
            pt_sl=pt_sl
        )
        
        # Get actual outcomes
        actual_labels = self.triple_barrier.get_labels(prices, events, barriers)
        
        # Meta-labels: 1 if primary prediction was correct, 0 otherwise
        meta_labels = pd.Series(index=primary_labels.index, dtype=float)
        
        for idx in primary_labels.index:
            if idx in actual_labels.index and primary_labels[idx] != 0:
                # Check if primary prediction matches actual outcome
                # Both should have same sign for correct prediction
                if np.sign(primary_labels[idx]) == np.sign(actual_labels[idx]):
                    meta_labels[idx] = 1
                else:
                    meta_labels[idx] = 0
            else:
                meta_labels[idx] = np.nan
        
        return meta_labels.fillna(0)


class SampleWeights:
    """
    Calculate sample weights for training
    Handles overlapping labels and sample uniqueness
    """
    
    def __init__(self):
        """Initialize sample weight calculator"""
        pass
    
    def get_sample_weights(self,
                          labels: pd.Series,
                          events: pd.DataFrame,
                          close_prices: pd.Series) -> pd.Series:
        """
        Calculate sample weights based on return attribution (IMPROVED)
        
        Args:
            labels: Label series
            events: Events DataFrame with t1 (end times)
            close_prices: Close price series
            
        Returns:
            Series with sample weights
        """
        # Calculate number of concurrent events
        num_concurrent = self._get_num_concurrent_events(events)
        
        # Calculate average uniqueness
        avg_uniqueness = self._get_average_uniqueness(events)
        
        # Calculate returns
        returns = close_prices.pct_change().abs()
        
        # Sample weights = avg_uniqueness * abs(returns) / num_concurrent
        weights = pd.Series(index=labels.index, dtype=float)
        
        for idx in labels.index:
            if idx in avg_uniqueness.index and idx in returns.index and idx in num_concurrent.index:
                # Weight by uniqueness and inverse of concurrent events
                concurrent = max(num_concurrent[idx], 1)
                weights[idx] = avg_uniqueness[idx] * returns[idx] / concurrent
            else:
                weights[idx] = 1.0  # Default weight
        
        # Normalize weights to sum to length
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)
        else:
            weights = pd.Series(1.0, index=labels.index)
        
        return weights
    
    def _get_num_concurrent_events(self, events: pd.DataFrame) -> pd.Series:
        """
        Calculate number of concurrent events at each point (IMPROVED)
        
        Args:
            events: Events DataFrame with t1 column
            
        Returns:
            Series with concurrent event counts
        """
        # Get all unique timestamps
        all_times = events.index.union(events['t1'].dropna().values).unique()
        timeline = pd.Series(0, index=all_times.sort_values())
        
        # Count overlapping events
        for start_time in events.index:
            if 't1' not in events.columns:
                continue
                
            end_time = events.loc[start_time, 't1']
            if pd.notna(end_time):
                # Increment count for all times in this event's duration
                mask = (timeline.index >= start_time) & (timeline.index <= end_time)
                timeline[mask] += 1
        
        return timeline
    
    def _get_average_uniqueness(self, events: pd.DataFrame) -> pd.Series:
        """
        Calculate average uniqueness of each label (IMPROVED)
        
        Args:
            events: Events DataFrame
            
        Returns:
            Series with average uniqueness
        """
        if 't1' not in events.columns:
            return pd.Series(1.0, index=events.index)
        
        # Calculate overlap matrix more efficiently
        n = len(events)
        uniqueness = pd.Series(1.0, index=events.index)
        
        for i, (idx_i, row_i) in enumerate(events.iterrows()):
            if pd.isna(row_i['t1']):
                continue
                
            overlaps = []
            for j, (idx_j, row_j) in enumerate(events.iterrows()):
                if i == j or pd.isna(row_j['t1']):
                    continue
                
                # Calculate overlap period
                overlap_start = max(idx_i, idx_j)
                overlap_end = min(row_i['t1'], row_j['t1'])
                
                if overlap_start < overlap_end:
                    # Calculate overlap fraction
                    total_duration = (row_i['t1'] - idx_i).total_seconds()
                    if total_duration > 0:
                        overlap_duration = (overlap_end - overlap_start).total_seconds()
                        overlap_fraction = overlap_duration / total_duration
                        overlaps.append(overlap_fraction)
            
            # Average uniqueness = 1 - average overlap
            if overlaps:
                uniqueness[idx_i] = 1 - np.mean(overlaps)
        
        return uniqueness


class LabelingPipeline:
    """
    Complete labeling pipeline combining multiple methods
    """
    
    def __init__(self, config: Optional[LabelingConfig] = None):
        """
        Initialize labeling pipeline
        
        Args:
            config: Labeling configuration
        """
        self.config = config or LabelingConfig()
        
        # Initialize labelers
        self.triple_barrier_labeler = TripleBarrierLabeler()
        self.fixed_time_labeler = FixedTimeLabeler(
            lookforward=self.config.lookforward,
            threshold=self.config.threshold
        )
        self.trend_labeler = TrendLabeler()
        self.meta_labeler = MetaLabeler()
        self.sample_weights = SampleWeights()
    
    def create_labels(self,
                     df: pd.DataFrame,
                     method: Optional[str] = None,
                     **kwargs) -> pd.Series:
        """
        Create labels using specified method
        
        Args:
            df: DataFrame with OHLCV data
            method: Labeling method to use
            **kwargs: Additional arguments for labeling
            
        Returns:
            Series with labels
        """
        method = method or self.config.method
        
        if method == 'triple_barrier':
            labels = self.triple_barrier_labeler.label_data(
                df,
                lookforward=kwargs.get('lookforward', self.config.lookforward),
                vol_window=kwargs.get('vol_window', self.config.vol_window),
                pt_sl=kwargs.get('pt_sl', self.config.pt_sl)
            )
        
        elif method == 'fixed_time':
            labels = self.fixed_time_labeler.label_data(
                df,
                num_classes=kwargs.get('num_classes', self.config.num_classes)
            )
        
        elif method == 'trend':
            labels = self.trend_labeler.label_data(df)
        
        elif method == 'simple_returns':
            # Simple return-based labeling
            lookforward = kwargs.get('lookforward', self.config.lookforward)
            threshold = kwargs.get('threshold', self.config.threshold)
            
            prices = df['close']
            returns = prices.pct_change(lookforward).shift(-lookforward)
            labels = pd.Series(0, index=df.index)
            labels[returns > threshold] = 1  # Up
            labels[returns < -threshold] = -1  # Down
            labels = labels.fillna(0).astype(int)
        
        else:
            raise ValueError(f"Unknown labeling method: {method}")
        
        return labels
    
    def create_labels_with_weights(self,
                                  df: pd.DataFrame,
                                  method: Optional[str] = None,
                                  **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Create labels with sample weights
        
        Args:
            df: DataFrame with OHLCV data
            method: Labeling method to use
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (labels, weights)
        """
        # Create labels
        labels = self.create_labels(df, method, **kwargs)
        
        # Create events for weight calculation
        prices = df['close']
        lookforward = kwargs.get('lookforward', self.config.lookforward)
        
        # Safe timestamps for events
        safe_end = max(1, len(prices) - lookforward - 1)
        t_events = prices.index[:safe_end]
        
        events = self.triple_barrier_labeler.get_events(
            prices=prices,
            t_events=t_events,
            num_periods=lookforward,  # Use periods, not days
            vol_window=self.config.vol_window
        )
        
        # Calculate weights
        if len(events) > 0:
            weights = self.sample_weights.get_sample_weights(labels, events, prices)
        else:
            weights = pd.Series(1.0, index=labels.index)
        
        return labels, weights
    
    def parallel_labeling(self,
                         df_dict: Dict[str, pd.DataFrame],
                         method: Optional[str] = None,
                         n_jobs: int = -1) -> Dict[str, pd.Series]:
        """
        Parallel labeling for multiple assets
        
        Args:
            df_dict: Dictionary of DataFrames {symbol: df}
            method: Labeling method
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary of labels {symbol: labels}
        """
        from multiprocessing import cpu_count
        
        if n_jobs == -1:
            n_jobs = cpu_count()
        
        results = {}
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit jobs
            futures = {
                executor.submit(self.create_labels, df, method): symbol
                for symbol, df in df_dict.items()
            }
            
            # Collect results
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    labels = future.result()
                    results[symbol] = labels
                    logger.info(f"Labeled {symbol}: {len(labels)} samples")
                except Exception as e:
                    logger.error(f"Error labeling {symbol}: {e}")
                    results[symbol] = pd.Series()
        
        return results
    
    def get_label_statistics(self, labels: pd.Series) -> Dict[str, Any]:
        """
        Get statistics about labels
        
        Args:
            labels: Label series
            
        Returns:
            Dictionary with statistics
        """
        # Remove NaN
        valid_labels = labels.dropna()
        
        if len(valid_labels) == 0:
            return {
                'total': len(labels),
                'valid': 0,
                'nan_count': len(labels),
                'nan_percentage': 100.0
            }
        
        # Get counts
        value_counts = valid_labels.value_counts()
        
        stats = {
            'total': len(labels),
            'valid': len(valid_labels),
            'nan_count': len(labels) - len(valid_labels),
            'nan_percentage': (len(labels) - len(valid_labels)) / len(labels) * 100,
            'class_distribution': value_counts.to_dict(),
            'class_percentages': (value_counts / len(valid_labels) * 100).to_dict(),
            'unique_classes': valid_labels.nunique(),
            'majority_class': value_counts.index[0] if len(value_counts) > 0 else None,
            'minority_class': value_counts.index[-1] if len(value_counts) > 0 else None,
            'class_balance_ratio': value_counts.min() / value_counts.max() if len(value_counts) > 1 and value_counts.max() > 0 else 0
        }
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("=== Labeling Module Test ===\n")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1h')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(45000, 46000, len(dates)),
        'low': np.random.uniform(39000, 40000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Test different labeling methods
    pipeline = LabelingPipeline()
    
    print("1. Testing Triple-Barrier Labeling...")
    tb_labels = pipeline.create_labels(sample_data, method='triple_barrier')
    tb_stats = pipeline.get_label_statistics(tb_labels)
    print(f"   Total samples: {tb_stats['total']}")
    print(f"   Valid labels: {tb_stats['valid']}")
    print(f"   Class distribution: {tb_stats.get('class_distribution', {})}")
    print(f"   Class balance ratio: {tb_stats.get('class_balance_ratio', 0):.3f}\n")
    
    print("2. Testing Fixed-Time Labeling...")
    ft_labels = pipeline.create_labels(sample_data, method='fixed_time')
    ft_stats = pipeline.get_label_statistics(ft_labels)
    print(f"   Class distribution: {ft_stats.get('class_distribution', {})}\n")
    
    print("3. Testing Trend Labeling...")
    trend_labels = pipeline.create_labels(sample_data, method='trend')
    trend_stats = pipeline.get_label_statistics(trend_labels)
    print(f"   Class distribution: {trend_stats.get('class_distribution', {})}\n")
    
    print("4. Testing Sample Weights...")
    labels_with_weights, weights = pipeline.create_labels_with_weights(sample_data)
    print(f"   Average weight: {weights.mean():.4f}")
    print(f"   Weight std: {weights.std():.4f}\n")
    
    print("=== Labeling Module Ready ===")