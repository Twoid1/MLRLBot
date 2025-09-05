"""
Performance Metrics Module
Comprehensive metrics calculation for trading strategy evaluation
Includes standard metrics, risk metrics, and advanced performance analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Complete set of performance metrics"""
    # Returns
    total_return: float
    annual_return: float
    monthly_return: float
    daily_return: float
    
    # Risk metrics
    volatility: float
    annual_volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_trade: float
    
    # Profit metrics
    profit_factor: float
    gross_profit: float
    gross_loss: float
    net_profit: float
    
    # Time metrics
    avg_holding_period: float
    max_holding_period: float
    min_holding_period: float
    time_in_market: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Streak metrics
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int
    
    # Recovery metrics
    recovery_factor: float
    payoff_ratio: float
    expectancy: float
    
    # Kelly metrics
    kelly_criterion: float
    optimal_f: float
    
    # Additional metrics
    omega_ratio: float
    ulcer_index: float
    tail_ratio: float
    common_sense_ratio: float
    cpc_index: float
    
    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    benchmark_correlation: Optional[float] = None


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics for trading strategies
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252,
                 confidence_level: float = 0.95):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            trading_days_per_year: Number of trading days per year
            confidence_level: Confidence level for VaR/CVaR
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        self.confidence_level = confidence_level
        
    def calculate_all_metrics(self,
                             equity_curve: pd.Series,
                             trades: Optional[pd.DataFrame] = None,
                             positions: Optional[pd.DataFrame] = None,
                             benchmark: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        
        Args:
            equity_curve: Series of portfolio values over time
            trades: DataFrame of completed trades
            positions: DataFrame of position history
            benchmark: Benchmark returns for comparison
            
        Returns:
            PerformanceMetrics object with all calculations
        """
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic return metrics
        total_return = self._calculate_total_return(equity_curve)
        annual_return = self._calculate_annual_return(total_return, equity_curve)
        monthly_return = self._calculate_period_return(returns, 21)
        daily_return = returns.mean()
        
        # Volatility metrics
        volatility = returns.std()
        annual_volatility = volatility * np.sqrt(self.trading_days)
        downside_volatility = self._calculate_downside_deviation(returns)
        
        # Drawdown metrics
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        max_drawdown = abs(drawdown_series.min())
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown_series)
        
        # Risk-adjusted returns
        sharpe = self._calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino = self._calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar = self._calculate_calmar_ratio(annual_return, max_drawdown)
        
        # Information and Treynor ratios (need benchmark)
        if benchmark is not None:
            info_ratio = self._calculate_information_ratio(returns, benchmark)
            treynor = self._calculate_treynor_ratio(returns, benchmark)
        else:
            info_ratio = 0
            treynor = 0
        
        # Trade statistics
        if trades is not None and not trades.empty:
            trade_metrics = self._calculate_trade_metrics(trades)
        else:
            trade_metrics = self._get_default_trade_metrics()
        
        # Distribution metrics
        skewness = skew(returns)
        kurt = kurtosis(returns)
        var_95 = self._calculate_var(returns, self.confidence_level)
        cvar_95 = self._calculate_cvar(returns, self.confidence_level)
        
        # Advanced metrics
        omega = self._calculate_omega_ratio(returns)
        ulcer = self._calculate_ulcer_index(drawdown_series)
        tail = self._calculate_tail_ratio(returns)
        csr = self._calculate_common_sense_ratio(trade_metrics)
        cpc = self._calculate_cpc_index(returns, trade_metrics)
        
        # Kelly and optimal f
        if trade_metrics['win_rate'] > 0 and trade_metrics['avg_loss'] != 0:
            kelly = self._calculate_kelly_criterion(
                trade_metrics['win_rate'],
                trade_metrics['avg_win'],
                abs(trade_metrics['avg_loss'])
            )
            optimal_f = self._calculate_optimal_f(trades) if trades is not None else 0
        else:
            kelly = 0
            optimal_f = 0
        
        # Create metrics object
        metrics = PerformanceMetrics(
            # Returns
            total_return=total_return,
            annual_return=annual_return,
            monthly_return=monthly_return,
            daily_return=daily_return,
            
            # Risk metrics
            volatility=volatility,
            annual_volatility=annual_volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            
            # Risk-adjusted returns
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            
            # Trade statistics
            total_trades=trade_metrics['total_trades'],
            winning_trades=trade_metrics['winning_trades'],
            losing_trades=trade_metrics['losing_trades'],
            win_rate=trade_metrics['win_rate'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            best_trade=trade_metrics['best_trade'],
            worst_trade=trade_metrics['worst_trade'],
            avg_trade=trade_metrics['avg_trade'],
            
            # Profit metrics
            profit_factor=trade_metrics['profit_factor'],
            gross_profit=trade_metrics['gross_profit'],
            gross_loss=trade_metrics['gross_loss'],
            net_profit=trade_metrics['net_profit'],
            
            # Time metrics
            avg_holding_period=trade_metrics['avg_holding_period'],
            max_holding_period=trade_metrics['max_holding_period'],
            min_holding_period=trade_metrics['min_holding_period'],
            time_in_market=self._calculate_time_in_market(positions) if positions is not None else 0,
            
            # Distribution metrics
            skewness=skewness,
            kurtosis=kurt,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Streak metrics
            max_consecutive_wins=trade_metrics['max_consecutive_wins'],
            max_consecutive_losses=trade_metrics['max_consecutive_losses'],
            current_streak=trade_metrics['current_streak'],
            
            # Recovery metrics
            recovery_factor=self._calculate_recovery_factor(total_return, max_drawdown),
            payoff_ratio=trade_metrics['payoff_ratio'],
            expectancy=trade_metrics['expectancy'],
            
            # Kelly metrics
            kelly_criterion=kelly,
            optimal_f=optimal_f,
            
            # Additional metrics
            omega_ratio=omega,
            ulcer_index=ulcer,
            tail_ratio=tail,
            common_sense_ratio=csr,
            cpc_index=cpc,
            
            # Metadata
            start_date=equity_curve.index[0] if len(equity_curve) > 0 else None,
            end_date=equity_curve.index[-1] if len(equity_curve) > 0 else None,
            trading_days=len(returns),
            benchmark_correlation=self._calculate_correlation(returns, benchmark) if benchmark is not None else None
        )
        
        return metrics
    
    # ================== RETURN CALCULATIONS ==================
    
    def _calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return"""
        if len(equity_curve) < 2:
            return 0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    def _calculate_annual_return(self, total_return: float, equity_curve: pd.Series) -> float:
        """Calculate annualized return"""
        if len(equity_curve) < 2:
            return 0
        
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days == 0:
            return 0
        
        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_period_return(self, returns: pd.Series, period: int) -> float:
        """Calculate average return over period"""
        if len(returns) < period:
            return returns.mean()
        
        period_returns = returns.rolling(period).apply(lambda x: (1 + x).prod() - 1)
        return period_returns.mean()
    
    # ================== RISK CALCULATIONS ==================
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        in_drawdown = drawdown_series < 0
        
        if not in_drawdown.any():
            return 0
        
        # Find drawdown periods
        drawdown_start = (~in_drawdown).shift(1) & in_drawdown
        drawdown_end = in_drawdown.shift(1) & (~in_drawdown)
        
        starts = drawdown_series.index[drawdown_start].tolist()
        ends = drawdown_series.index[drawdown_end].tolist()
        
        if not starts or not ends:
            return 0
        
        # Calculate durations
        durations = []
        for start, end in zip(starts, ends[:len(starts)]):
            duration = (end - start).days
            durations.append(duration)
        
        return max(durations) if durations else 0
    
    def _calculate_downside_deviation(self, returns: pd.Series, 
                                    target_return: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 0
        
        return np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    # ================== RISK-ADJUSTED RETURNS ==================
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, 
                               risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / self.trading_days
        return np.sqrt(self.trading_days) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series,
                                risk_free_rate: float) -> float:
        """Calculate Sortino ratio"""
        downside_vol = self._calculate_downside_deviation(returns, risk_free_rate / self.trading_days)
        
        if downside_vol == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / self.trading_days
        return np.sqrt(self.trading_days) * excess_returns.mean() / downside_vol
    
    def _calculate_calmar_ratio(self, annual_return: float,
                               max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0
        return annual_return / max_drawdown
    
    def _calculate_information_ratio(self, returns: pd.Series,
                                    benchmark: pd.Series) -> float:
        """Calculate Information ratio"""
        active_returns = returns - benchmark
        
        if active_returns.std() == 0:
            return 0
        
        return np.sqrt(self.trading_days) * active_returns.mean() / active_returns.std()
    
    def _calculate_treynor_ratio(self, returns: pd.Series,
                                benchmark: pd.Series) -> float:
        """Calculate Treynor ratio"""
        # Calculate beta
        covariance = returns.cov(benchmark)
        benchmark_variance = benchmark.var()
        
        if benchmark_variance == 0:
            return 0
        
        beta = covariance / benchmark_variance
        
        if beta == 0:
            return 0
        
        excess_returns = returns.mean() - self.risk_free_rate / self.trading_days
        return excess_returns / beta
    
    # ================== TRADE METRICS ==================
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-based metrics"""
        if trades.empty:
            return self._get_default_trade_metrics()
        
        # Ensure required columns
        if 'pnl' not in trades.columns:
            return self._get_default_trade_metrics()
        
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_trades = len(trades)
        num_winners = len(winning_trades)
        num_losers = len(losing_trades)
        
        win_rate = num_winners / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if num_winners > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if num_losers > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if num_winners > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if num_losers > 0 else 0
        net_profit = trades['pnl'].sum()
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        best_trade = trades['pnl'].max()
        worst_trade = trades['pnl'].min()
        avg_trade = trades['pnl'].mean()
        
        # Holding periods
        if 'duration' in trades.columns:
            avg_holding = trades['duration'].mean()
            max_holding = trades['duration'].max()
            min_holding = trades['duration'].min()
        else:
            avg_holding = max_holding = min_holding = 0
        
        # Consecutive wins/losses
        max_wins, max_losses, current = self._calculate_streaks(trades)
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        return {
            'total_trades': total_trades,
            'winning_trades': num_winners,
            'losing_trades': num_losers,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade': avg_trade,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding,
            'max_holding_period': max_holding,
            'min_holding_period': min_holding,
            'max_consecutive_wins': max_wins,
            'max_consecutive_losses': max_losses,
            'current_streak': current,
            'payoff_ratio': payoff_ratio,
            'expectancy': expectancy
        }
    
    def _get_default_trade_metrics(self) -> Dict:
        """Get default trade metrics when no trades available"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_trade': 0,
            'gross_profit': 0,
            'gross_loss': 0,
            'net_profit': 0,
            'profit_factor': 0,
            'avg_holding_period': 0,
            'max_holding_period': 0,
            'min_holding_period': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'payoff_ratio': 0,
            'expectancy': 0
        }
    
    def _calculate_streaks(self, trades: pd.DataFrame) -> Tuple[int, int, int]:
        """Calculate consecutive win/loss streaks"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0, 0, 0
        
        wins = []
        losses = []
        current_wins = 0
        current_losses = 0
        
        for pnl in trades['pnl']:
            if pnl > 0:
                current_wins += 1
                if current_losses > 0:
                    losses.append(current_losses)
                    current_losses = 0
            else:
                current_losses += 1
                if current_wins > 0:
                    wins.append(current_wins)
                    current_wins = 0
        
        # Add final streak
        if current_wins > 0:
            wins.append(current_wins)
            current = current_wins
        elif current_losses > 0:
            losses.append(current_losses)
            current = -current_losses
        else:
            current = 0
        
        max_wins = max(wins) if wins else 0
        max_losses = max(losses) if losses else 0
        
        return max_wins, max_losses, current
    
    # ================== ADVANCED METRICS ==================
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """Calculate Omega ratio"""
        if len(returns) == 0:
            return 0
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if losses.sum() == 0:
            return float('inf') if gains.sum() > 0 else 0
        
        return gains.sum() / losses.sum()
    
    def _calculate_ulcer_index(self, drawdown_series: pd.Series) -> float:
        """Calculate Ulcer Index"""
        if len(drawdown_series) == 0:
            return 0
        
        squared_drawdowns = drawdown_series ** 2
        return np.sqrt(squared_drawdowns.mean())
    
    def _calculate_tail_ratio(self, returns: pd.Series, percentile: float = 5) -> float:
        """Calculate tail ratio"""
        if len(returns) < 20:
            return 0
        
        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = abs(np.percentile(returns, percentile))
        
        if left_tail == 0:
            return float('inf') if right_tail > 0 else 0
        
        return right_tail / left_tail
    
    def _calculate_common_sense_ratio(self, trade_metrics: Dict) -> float:
        """Calculate Common Sense Ratio"""
        if trade_metrics['profit_factor'] == 0:
            return 0
        
        tail = trade_metrics['profit_factor'] * trade_metrics['payoff_ratio']
        
        if tail <= 0:
            return 0
        
        return np.log(tail)
    
    def _calculate_cpc_index(self, returns: pd.Series, trade_metrics: Dict) -> float:
        """Calculate CPC Index (combines win rate, payoff, and profit factor)"""
        if trade_metrics['win_rate'] == 0:
            return 0
        
        return (trade_metrics['win_rate'] * 
                trade_metrics['payoff_ratio'] * 
                trade_metrics['profit_factor'])
    
    def _calculate_kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        if b == 0:
            return 0
        
        kelly = (p * b - q) / b
        
        # Cap at 25% for safety
        return min(kelly, 0.25)
    
    def _calculate_optimal_f(self, trades: pd.DataFrame) -> float:
        """Calculate Ralph Vince's Optimal f"""
        if trades.empty or 'pnl' not in trades.columns:
            return 0
        
        returns = trades['pnl'].values
        
        # Search for optimal f between 0.01 and 1
        best_f = 0
        best_twi = 0
        
        for f in np.arange(0.01, 1.0, 0.01):
            twi = self._calculate_twi(returns, f)
            if twi > best_twi:
                best_twi = twi
                best_f = f
        
        return best_f
    
    def _calculate_twi(self, returns: np.ndarray, f: float) -> float:
        """Calculate Terminal Wealth Index for optimal f"""
        twi = 1.0
        
        for ret in returns:
            if ret < 0 and abs(ret) >= 1/f:
                return 0  # Would result in bankruptcy
            twi *= (1 + f * ret)
        
        return twi
    
    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate Recovery Factor"""
        if max_drawdown == 0:
            return 0
        return abs(total_return / max_drawdown)
    
    def _calculate_time_in_market(self, positions: pd.DataFrame) -> float:
        """Calculate percentage of time in market"""
        if positions is None or positions.empty:
            return 0
        
        # This would need position start/end times
        # Simplified version
        return 1.0
    
    def _calculate_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate correlation between two return series"""
        if returns2 is None or len(returns2) == 0:
            return 0
        
        # Align the series
        aligned = pd.DataFrame({'r1': returns1, 'r2': returns2}).dropna()
        
        if len(aligned) < 2:
            return 0
        
        return aligned['r1'].corr(aligned['r2'])
    
    # ================== REPORTING ==================
    
    def create_report(self, metrics: PerformanceMetrics) -> str:
        """Create a formatted text report of metrics"""
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE METRICS REPORT")
        report.append("=" * 60)
        
        # Returns
        report.append("\n### RETURNS ###")
        report.append(f"Total Return: {metrics.total_return:.2%}")
        report.append(f"Annual Return: {metrics.annual_return:.2%}")
        report.append(f"Monthly Return: {metrics.monthly_return:.2%}")
        report.append(f"Daily Return: {metrics.daily_return:.4%}")
        
        # Risk
        report.append("\n### RISK METRICS ###")
        report.append(f"Annual Volatility: {metrics.annual_volatility:.2%}")
        report.append(f"Max Drawdown: {metrics.max_drawdown:.2%}")
        report.append(f"Max DD Duration: {metrics.max_drawdown_duration} days")
        report.append(f"VaR (95%): {metrics.var_95:.2%}")
        report.append(f"CVaR (95%): {metrics.cvar_95:.2%}")
        
        # Risk-adjusted
        report.append("\n### RISK-ADJUSTED RETURNS ###")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append(f"Omega Ratio: {metrics.omega_ratio:.3f}")
        
        # Trading
        report.append("\n### TRADING STATISTICS ###")
        report.append(f"Total Trades: {metrics.total_trades}")
        report.append(f"Win Rate: {metrics.win_rate:.2%}")
        report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
        report.append(f"Expectancy: ${metrics.expectancy:.2f}")
        report.append(f"Avg Win: ${metrics.avg_win:.2f}")
        report.append(f"Avg Loss: ${metrics.avg_loss:.2f}")
        report.append(f"Best Trade: ${metrics.best_trade:.2f}")
        report.append(f"Worst Trade: ${metrics.worst_trade:.2f}")
        
        # Streaks
        report.append("\n### STREAK ANALYSIS ###")
        report.append(f"Max Consecutive Wins: {metrics.max_consecutive_wins}")
        report.append(f"Max Consecutive Losses: {metrics.max_consecutive_losses}")
        report.append(f"Current Streak: {metrics.current_streak}")
        
        # Kelly
        report.append("\n### POSITION SIZING ###")
        report.append(f"Kelly Criterion: {metrics.kelly_criterion:.2%}")
        report.append(f"Optimal f: {metrics.optimal_f:.3f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    equity = 10000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    equity_curve = pd.Series(equity, index=dates)
    
    # Sample trades
    trades = pd.DataFrame({
        'pnl': np.random.randn(50) * 100,
        'duration': np.random.randint(1, 20, 50)
    })
    
    # Calculate metrics
    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(equity_curve, trades)
    
    # Print report
    print(calculator.create_report(metrics))
    print("\n Metrics calculator ready!")