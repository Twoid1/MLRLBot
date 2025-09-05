"""
Position Sizer Module
Advanced position sizing algorithms for optimal capital allocation
Works with risk_manager.py for comprehensive risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats, optimize
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class MarketConditions:
    """Current market conditions for position sizing"""
    volatility: float
    trend_strength: float
    correlation_matrix: pd.DataFrame
    market_regime: str  # 'trending', 'ranging', 'volatile'
    liquidity: float
    spread: float
    volume_profile: Dict[str, float]


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation"""
    position_size: float
    position_value: float
    risk_amount: float
    confidence: float
    method_used: str
    adjustments_applied: List[str]
    max_loss: float
    expected_return: float
    risk_reward_ratio: float
    metadata: Dict[str, Any]


class PositionSizer:
    """
    Advanced position sizing calculator
    
    Implements multiple sizing algorithms:
    - Kelly Criterion (multiple variations)
    - Optimal F
    - Risk Parity
    - Maximum Sharpe Ratio
    - Monte Carlo sizing
    - Machine Learning based sizing
    """
    
    def __init__(self,
                 capital: float,
                 max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.06,
                 confidence_level: float = 0.95):
        """
        Initialize Position Sizer
        
        Args:
            capital: Available capital
            max_risk_per_trade: Maximum risk per trade (as fraction)
            max_portfolio_risk: Maximum portfolio risk
            confidence_level: Confidence level for calculations
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.confidence_level = confidence_level
        
        # Historical data for calculations
        self.trade_history = []
        self.returns_history = []
        self.position_outcomes = {}
        
        # Portfolio tracking
        self.current_positions = {}
        self.portfolio_weights = {}
        self.correlation_matrix = pd.DataFrame()
        
        # Cache for calculations
        self._kelly_cache = {}
        self._optimal_f_cache = {}
        
        logger.info(f"PositionSizer initialized with ${capital:,.2f} capital")
    
    # ================== KELLY CRITERION VARIATIONS ==================
    
    def kelly_criterion_basic(self,
                             win_probability: float,
                             win_amount: float,
                             loss_amount: float,
                             kelly_fraction: float = 0.25) -> PositionSizeResult:
        """
        Basic Kelly Criterion
        f* = (p*b - q) / b
        
        Args:
            win_probability: Probability of winning
            win_amount: Average win amount
            loss_amount: Average loss amount
            kelly_fraction: Fraction of Kelly to use (safety)
            
        Returns:
            Position sizing result
        """
        if loss_amount == 0:
            return self._create_default_result("kelly_basic")
        
        p = win_probability
        q = 1 - win_probability
        b = abs(win_amount / loss_amount)
        
        # Kelly formula
        kelly_f = (p * b - q) / b if b > 0 else 0
        
        # Apply safety fraction
        kelly_f *= kelly_fraction
        
        # Cap at maximum risk
        kelly_f = min(kelly_f, self.max_risk_per_trade)
        kelly_f = max(0, kelly_f)
        
        position_value = self.capital * kelly_f
        
        return PositionSizeResult(
            position_size=0,  # Will be calculated with price
            position_value=position_value,
            risk_amount=position_value * (1 - win_probability),
            confidence=self._calculate_kelly_confidence(p, b),
            method_used="kelly_basic",
            adjustments_applied=["kelly_fraction", "max_risk_cap"],
            max_loss=loss_amount,
            expected_return=p * win_amount - q * loss_amount,
            risk_reward_ratio=b,
            metadata={'kelly_f': kelly_f, 'fraction_used': kelly_fraction}
        )
    
    def kelly_criterion_continuous(self,
                                  mean_return: float,
                                  variance: float,
                                  kelly_fraction: float = 0.25) -> PositionSizeResult:
        """
        Kelly Criterion for continuous outcomes (Thorp's formula)
        f* = Î¼ / ÏƒÂ²
        
        Args:
            mean_return: Expected return (mean)
            variance: Return variance
            kelly_fraction: Fraction of Kelly to use
            
        Returns:
            Position sizing result
        """
        if variance == 0:
            return self._create_default_result("kelly_continuous")
        
        # Continuous Kelly formula
        kelly_f = mean_return / variance
        
        # Apply safety fraction
        kelly_f *= kelly_fraction
        
        # Apply constraints
        kelly_f = min(kelly_f, self.max_risk_per_trade)
        kelly_f = max(0, kelly_f)
        
        position_value = self.capital * kelly_f
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * np.sqrt(variance),
            confidence=self._calculate_sharpe_confidence(mean_return, np.sqrt(variance)),
            method_used="kelly_continuous",
            adjustments_applied=["kelly_fraction", "variance_adjustment"],
            max_loss=position_value * (mean_return - 2 * np.sqrt(variance)),
            expected_return=mean_return * position_value,
            risk_reward_ratio=mean_return / np.sqrt(variance) if variance > 0 else 0,
            metadata={'kelly_f': kelly_f, 'sharpe': mean_return / np.sqrt(variance)}
        )
    
    def kelly_with_multiple_outcomes(self,
                                    outcomes: List[Tuple[float, float]],
                                    kelly_fraction: float = 0.25) -> PositionSizeResult:
        """
        Kelly Criterion with multiple possible outcomes
        
        Args:
            outcomes: List of (probability, return) tuples
            kelly_fraction: Fraction of Kelly to use
            
        Returns:
            Position sizing result
        """
        # Calculate expected return
        expected_return = sum(p * r for p, r in outcomes)
        
        # Calculate second moment
        second_moment = sum(p * r**2 for p, r in outcomes)
        
        # Calculate variance
        variance = second_moment - expected_return**2
        
        if variance <= 0:
            return self._create_default_result("kelly_multiple")
        
        # Kelly formula for multiple outcomes
        kelly_f = expected_return / variance
        
        # Apply adjustments
        kelly_f *= kelly_fraction
        kelly_f = min(kelly_f, self.max_risk_per_trade)
        kelly_f = max(0, kelly_f)
        
        position_value = self.capital * kelly_f
        
        # Calculate downside risk
        downside_outcomes = [(p, r) for p, r in outcomes if r < 0]
        downside_risk = sum(p * abs(r) for p, r in downside_outcomes) if downside_outcomes else 0
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * downside_risk,
            confidence=self._calculate_outcome_confidence(outcomes),
            method_used="kelly_multiple",
            adjustments_applied=["kelly_fraction", "multiple_outcomes"],
            max_loss=position_value * min(r for _, r in outcomes),
            expected_return=expected_return * position_value,
            risk_reward_ratio=expected_return / np.sqrt(variance),
            metadata={
                'kelly_f': kelly_f,
                'num_outcomes': len(outcomes),
                'expected_return': expected_return,
                'variance': variance
            }
        )
    
    # ================== OPTIMAL F ==================
    
    def optimal_f(self,
                 trade_returns: List[float],
                 max_iterations: int = 1000) -> PositionSizeResult:
        """
        Ralph Vince's Optimal f calculation
        Maximizes Terminal Wealth Relative (TWR)
        
        Args:
            trade_returns: Historical trade returns
            max_iterations: Maximum iterations for optimization
            
        Returns:
            Position sizing result
        """
        if not trade_returns or len(trade_returns) < 10:
            return self._create_default_result("optimal_f")
        
        returns = np.array(trade_returns)
        
        # Find worst loss
        worst_loss = min(returns)
        if worst_loss >= 0:
            return self._create_default_result("optimal_f")
        
        def negative_twr(f):
            """Calculate negative TWR for minimization"""
            twr = 1
            for ret in returns:
                hpr = 1 + f * (ret / abs(worst_loss))
                if hpr <= 0:
                    return float('inf')
                twr *= hpr
            return -twr
        
        # Optimize
        result = optimize.minimize_scalar(
            negative_twr,
            bounds=(0.01, 0.99),
            method='bounded',
            options={'maxiter': max_iterations}
        )
        
        optimal_f_value = result.x if result.success else 0.25
        
        # Apply safety factor
        optimal_f_value *= 0.5  # Use 50% of optimal f
        
        # Apply constraints
        optimal_f_value = min(optimal_f_value, self.max_risk_per_trade)
        
        position_value = self.capital * optimal_f_value
        
        # Calculate metrics
        expected_return = np.mean(returns)
        std_return = np.std(returns)
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * abs(worst_loss),
            confidence=self._calculate_optimal_f_confidence(returns),
            method_used="optimal_f",
            adjustments_applied=["safety_factor_50%", "worst_loss_constraint"],
            max_loss=position_value * abs(worst_loss),
            expected_return=position_value * expected_return,
            risk_reward_ratio=expected_return / std_return if std_return > 0 else 0,
            metadata={
                'optimal_f': optimal_f_value,
                'worst_loss': worst_loss,
                'twr': -negative_twr(optimal_f_value)
            }
        )
    
    # ================== RISK PARITY ==================
    
    def risk_parity_sizing(self,
                         volatilities: Dict[str, float],
                         correlations: Optional[pd.DataFrame] = None,
                         existing_positions: Optional[Dict[str, float]] = None) -> Dict[str, PositionSizeResult]:
        """
        Risk Parity position sizing
        Equal risk contribution from each asset
        
        Args:
            volatilities: Asset volatilities
            correlations: Correlation matrix
            existing_positions: Current positions
            
        Returns:
            Position sizes for each asset
        """
        n_assets = len(volatilities)
        
        if n_assets == 0:
            return {}
        
        # Equal risk budget for each asset
        risk_budget = self.max_portfolio_risk / n_assets
        
        results = {}
        
        for asset, volatility in volatilities.items():
            if volatility <= 0:
                results[asset] = self._create_default_result("risk_parity")
                continue
            
            # Position value inversely proportional to volatility
            position_value = (risk_budget * self.capital) / volatility
            
            # Adjust for existing positions
            if existing_positions and asset in existing_positions:
                current_value = existing_positions[asset]
                position_value = max(0, position_value - current_value)
            
            # Apply maximum position constraint
            max_position = self.capital * self.max_risk_per_trade / volatility
            position_value = min(position_value, max_position)
            
            results[asset] = PositionSizeResult(
                position_size=0,
                position_value=position_value,
                risk_amount=position_value * volatility,
                confidence=0.5,  # Risk parity doesn't provide confidence
                method_used="risk_parity",
                adjustments_applied=["equal_risk_contribution"],
                max_loss=position_value * volatility * 2,
                expected_return=0,  # Risk parity is risk-focused
                risk_reward_ratio=1,
                metadata={
                    'risk_budget': risk_budget,
                    'volatility': volatility,
                    'n_assets': n_assets
                }
            )
        
        return results
    
    # ================== MAXIMUM SHARPE RATIO ==================
    
    def maximum_sharpe_sizing(self,
                            expected_returns: Dict[str, float],
                            volatilities: Dict[str, float],
                            correlations: pd.DataFrame,
                            risk_free_rate: float = 0.02) -> Dict[str, PositionSizeResult]:
        """
        Position sizing to maximize portfolio Sharpe ratio
        
        Args:
            expected_returns: Expected returns for each asset
            volatilities: Asset volatilities
            correlations: Correlation matrix
            risk_free_rate: Risk-free rate
            
        Returns:
            Optimal position sizes
        """
        assets = list(expected_returns.keys())
        n_assets = len(assets)
        
        if n_assets == 0:
            return {}
        
        # Convert to arrays
        returns = np.array([expected_returns[a] for a in assets])
        vols = np.array([volatilities[a] for a in assets])
        
        # Create covariance matrix
        cov_matrix = np.outer(vols, vols) * correlations.values
        
        def negative_sharpe(weights):
            """Calculate negative Sharpe ratio for minimization"""
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return float('inf')
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': lambda x: x}  # Non-negative weights
        ]
        
        # Initial guess (equal weight)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            constraints=constraints,
            bounds=[(0, 1) for _ in range(n_assets)]
        )
        
        optimal_weights = result.x if result.success else x0
        
        # Convert to position sizes
        results = {}
        total_capital = self.capital
        
        for i, asset in enumerate(assets):
            position_value = total_capital * optimal_weights[i]
            
            # Apply risk constraints
            max_position = total_capital * self.max_risk_per_trade
            position_value = min(position_value, max_position)
            
            results[asset] = PositionSizeResult(
                position_size=0,
                position_value=position_value,
                risk_amount=position_value * volatilities[asset],
                confidence=1 / (1 + result.fun) if result.success else 0.5,
                method_used="maximum_sharpe",
                adjustments_applied=["portfolio_optimization"],
                max_loss=position_value * volatilities[asset] * 2,
                expected_return=position_value * expected_returns[asset],
                risk_reward_ratio=expected_returns[asset] / volatilities[asset],
                metadata={
                    'weight': optimal_weights[i],
                    'sharpe': -result.fun if result.success else 0,
                    'optimization_success': result.success
                }
            )
        
        return results
    
    # ================== MONTE CARLO SIZING ==================
    
    def monte_carlo_sizing(self,
                         expected_return: float,
                         volatility: float,
                         n_simulations: int = 10000,
                         time_horizon: int = 252,
                         max_drawdown_limit: float = 0.20) -> PositionSizeResult:
        """
        Monte Carlo simulation for position sizing
        
        Args:
            expected_return: Expected annual return
            volatility: Annual volatility
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            max_drawdown_limit: Maximum acceptable drawdown
            
        Returns:
            Position sizing result
        """
        # Daily parameters
        daily_return = expected_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        # Test different position sizes
        test_sizes = np.linspace(0.01, self.max_risk_per_trade, 20)
        best_size = 0
        best_score = -float('inf')
        best_metrics = {}
        
        for size in test_sizes:
            # Run simulations
            final_values = []
            max_drawdowns = []
            
            for _ in range(n_simulations):
                # Generate random returns
                returns = np.random.normal(daily_return, daily_vol, time_horizon)
                
                # Calculate portfolio value path
                portfolio_value = self.capital
                peak_value = portfolio_value
                max_dd = 0
                
                for r in returns:
                    portfolio_value *= (1 + r * size)
                    peak_value = max(peak_value, portfolio_value)
                    drawdown = (peak_value - portfolio_value) / peak_value
                    max_dd = max(max_dd, drawdown)
                
                final_values.append(portfolio_value)
                max_drawdowns.append(max_dd)
            
            # Calculate metrics
            avg_final_value = np.mean(final_values)
            avg_drawdown = np.mean(max_drawdowns)
            prob_exceed_dd_limit = np.mean([dd > max_drawdown_limit for dd in max_drawdowns])
            
            # Score (maximize return while respecting drawdown limit)
            if prob_exceed_dd_limit < 0.05:  # Less than 5% chance of exceeding limit
                score = avg_final_value
                if score > best_score:
                    best_score = score
                    best_size = size
                    best_metrics = {
                        'avg_final_value': avg_final_value,
                        'avg_drawdown': avg_drawdown,
                        'prob_exceed_dd': prob_exceed_dd_limit,
                        'percentile_95_dd': np.percentile(max_drawdowns, 95)
                    }
        
        position_value = self.capital * best_size
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * volatility,
            confidence=1 - best_metrics.get('prob_exceed_dd', 1),
            method_used="monte_carlo",
            adjustments_applied=["simulation_optimization", "drawdown_constraint"],
            max_loss=position_value * best_metrics.get('percentile_95_dd', 0.2),
            expected_return=position_value * expected_return,
            risk_reward_ratio=expected_return / volatility,
            metadata={
                'optimal_size_fraction': best_size,
                'n_simulations': n_simulations,
                **best_metrics
            }
        )
    
    # ================== MACHINE LEARNING BASED SIZING ==================
    
    def ml_based_sizing(self,
                       features: Dict[str, float],
                       model_confidence: float,
                       predicted_return: float,
                       prediction_std: float) -> PositionSizeResult:
        """
        Position sizing based on ML model predictions
        
        Args:
            features: Feature dictionary used for prediction
            model_confidence: Model's confidence in prediction
            predicted_return: Predicted return
            prediction_std: Standard deviation of prediction
            
        Returns:
            Position sizing result
        """
        # Base position size from predicted Sharpe ratio
        if prediction_std > 0:
            predicted_sharpe = predicted_return / prediction_std
            base_size = min(predicted_sharpe / 3, self.max_risk_per_trade)  # Divide by 3 for safety
        else:
            base_size = self.max_risk_per_trade * 0.5
        
        # Adjust by model confidence
        adjusted_size = base_size * model_confidence
        
        # Further adjustments based on features
        adjustments = []
        
        # Market regime adjustment
        if 'volatility' in features:
            vol = features['volatility']
            if vol > 0.02:  # High volatility
                adjusted_size *= 0.7
                adjustments.append("high_volatility_reduction")
        
        # Trend strength adjustment
        if 'trend_strength' in features:
            trend = features['trend_strength']
            if abs(trend) > 0.7:  # Strong trend
                adjusted_size *= 1.2
                adjustments.append("strong_trend_increase")
        
        # Volume adjustment
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if vol_ratio < 0.5:  # Low volume
                adjusted_size *= 0.8
                adjustments.append("low_volume_reduction")
        
        # Final constraints
        adjusted_size = max(0, min(adjusted_size, self.max_risk_per_trade))
        position_value = self.capital * adjusted_size
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * prediction_std,
            confidence=model_confidence,
            method_used="ml_based",
            adjustments_applied=["confidence_scaling"] + adjustments,
            max_loss=position_value * (prediction_std * 2),
            expected_return=position_value * predicted_return,
            risk_reward_ratio=predicted_return / prediction_std if prediction_std > 0 else 0,
            metadata={
                'model_confidence': model_confidence,
                'predicted_sharpe': predicted_sharpe if prediction_std > 0 else 0,
                'features_used': list(features.keys()),
                'base_size': base_size,
                'final_size': adjusted_size
            }
        )
    
    # ================== DYNAMIC POSITION SIZING ==================
    
    def dynamic_sizing(self,
                      current_performance: Dict[str, float],
                      market_conditions: MarketConditions,
                      base_method: str = 'kelly') -> PositionSizeResult:
        """
        Dynamic position sizing based on current performance and market conditions
        
        Args:
            current_performance: Current trading performance metrics
            market_conditions: Current market conditions
            base_method: Base sizing method to use
            
        Returns:
            Dynamically adjusted position size
        """
        # Get base position size using specified method
        if base_method == 'kelly':
            base_result = self.kelly_criterion_basic(
                win_probability=current_performance.get('win_rate', 0.5),
                win_amount=current_performance.get('avg_win', 100),
                loss_amount=current_performance.get('avg_loss', 100)
            )
        else:
            base_result = self._create_default_result("dynamic")
        
        adjustments = []
        adjustment_factor = 1.0
        
        # Performance-based adjustments
        win_rate = current_performance.get('win_rate', 0.5)
        if win_rate > 0.6:  # Good performance
            adjustment_factor *= 1.2
            adjustments.append("high_win_rate_boost")
        elif win_rate < 0.4:  # Poor performance
            adjustment_factor *= 0.7
            adjustments.append("low_win_rate_reduction")
        
        # Drawdown adjustment
        drawdown = current_performance.get('drawdown', 0)
        if drawdown > 0.1:  # 10% drawdown
            adjustment_factor *= (1 - drawdown)
            adjustments.append("drawdown_reduction")
        
        # Market regime adjustments
        if market_conditions.market_regime == 'volatile':
            adjustment_factor *= 0.6
            adjustments.append("volatile_market_reduction")
        elif market_conditions.market_regime == 'trending':
            adjustment_factor *= 1.1
            adjustments.append("trending_market_boost")
        
        # Correlation adjustment
        if not market_conditions.correlation_matrix.empty:
            avg_correlation = market_conditions.correlation_matrix.mean().mean()
            if avg_correlation > 0.7:  # High correlation
                adjustment_factor *= 0.8
                adjustments.append("high_correlation_reduction")
        
        # Apply adjustments
        adjusted_value = base_result.position_value * adjustment_factor
        
        # Apply final constraints
        adjusted_value = min(adjusted_value, self.capital * self.max_risk_per_trade)
        
        return PositionSizeResult(
            position_size=0,
            position_value=adjusted_value,
            risk_amount=base_result.risk_amount * adjustment_factor,
            confidence=base_result.confidence * adjustment_factor,
            method_used="dynamic_" + base_method,
            adjustments_applied=base_result.adjustments_applied + adjustments,
            max_loss=base_result.max_loss * adjustment_factor,
            expected_return=base_result.expected_return * adjustment_factor,
            risk_reward_ratio=base_result.risk_reward_ratio,
            metadata={
                'base_method': base_method,
                'adjustment_factor': adjustment_factor,
                'market_regime': market_conditions.market_regime,
                **base_result.metadata
            }
        )
    
    # ================== UTILITY METHODS ==================
    
    def convert_to_shares(self,
                         position_value: float,
                         current_price: float,
                         lot_size: float = 1.0) -> float:
        """
        Convert position value to number of shares/units
        
        Args:
            position_value: Value of position in currency
            current_price: Current price per unit
            lot_size: Minimum trading unit
            
        Returns:
            Number of shares/units
        """
        if current_price <= 0:
            return 0
        
        shares = position_value / current_price
        
        # Round to lot size
        if lot_size > 0:
            shares = np.floor(shares / lot_size) * lot_size
        
        return shares
    
    def update_capital(self, new_capital: float) -> None:
        """Update available capital"""
        self.capital = new_capital
        logger.info(f"Capital updated to ${new_capital:,.2f}")
    
    def add_trade_result(self, trade_return: float) -> None:
        """Add trade result to history"""
        self.trade_history.append(trade_return)
        self.returns_history.append(trade_return)
        
        # Clear caches
        self._kelly_cache.clear()
        self._optimal_f_cache.clear()
    
    def _calculate_kelly_confidence(self, prob: float, odds: float) -> float:
        """Calculate confidence in Kelly criterion"""
        # Higher probability and better odds = higher confidence
        confidence = prob * min(odds / 2, 1)
        return min(confidence, 1.0)
    
    def _calculate_sharpe_confidence(self, mean: float, std: float) -> float:
        """Calculate confidence based on Sharpe ratio"""
        if std == 0:
            return 0
        sharpe = mean / std
        # Map Sharpe ratio to confidence (0-1)
        confidence = 1 / (1 + np.exp(-sharpe))  # Sigmoid
        return confidence
    
    def _calculate_outcome_confidence(self, outcomes: List[Tuple[float, float]]) -> float:
        """Calculate confidence from multiple outcomes"""
        # Higher expected value and lower variance = higher confidence
        expected = sum(p * r for p, r in outcomes)
        variance = sum(p * (r - expected)**2 for p, r in outcomes)
        
        if variance == 0:
            return 1.0 if expected > 0 else 0
        
        sharpe = expected / np.sqrt(variance)
        confidence = 1 / (1 + np.exp(-sharpe))
        return confidence
    
    def _calculate_optimal_f_confidence(self, returns: List[float]) -> float:
        """Calculate confidence in optimal f"""
        if len(returns) < 30:
            return 0.3  # Low confidence with little data
        
        # Calculate stability of returns
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        return_std = np.std(returns)
        avg_return = np.mean(returns)
        
        if return_std == 0:
            return 0.5
        
        # Combine metrics
        sharpe = avg_return / return_std
        confidence = win_rate * (1 / (1 + np.exp(-sharpe)))
        
        return min(confidence, 0.95)
    
    def _create_default_result(self, method: str) -> PositionSizeResult:
        """Create default position size result"""
        default_value = self.capital * self.max_risk_per_trade * 0.5
        
        return PositionSizeResult(
            position_size=0,
            position_value=default_value,
            risk_amount=default_value * 0.02,
            confidence=0.5,
            method_used=method + "_default",
            adjustments_applied=["default_fallback"],
            max_loss=default_value * 0.02,
            expected_return=0,
            risk_reward_ratio=1.0,
            metadata={'reason': 'insufficient_data_or_invalid_parameters'}
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of position sizer state"""
        return {
            'capital': self.capital,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_portfolio_risk': self.max_portfolio_risk,
            'n_trades': len(self.trade_history),
            'avg_return': np.mean(self.returns_history) if self.returns_history else 0,
            'current_positions': len(self.current_positions),
            'total_exposure': sum(self.current_positions.values())
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize position sizer
    sizer = PositionSizer(
        capital=10000,
        max_risk_per_trade=0.02,
        max_portfolio_risk=0.06
    )
    
    print("=== Position Sizing Examples ===\n")
    
    # 1. Basic Kelly Criterion
    print("1. Basic Kelly Criterion:")
    kelly_result = sizer.kelly_criterion_basic(
        win_probability=0.55,
        win_amount=200,
        loss_amount=150
    )
    print(f"   Position Value: ${kelly_result.position_value:.2f}")
    print(f"   Risk Amount: ${kelly_result.risk_amount:.2f}")
    print(f"   Confidence: {kelly_result.confidence:.2%}")
    print(f"   Risk/Reward: {kelly_result.risk_reward_ratio:.2f}")
    
    # 2. Continuous Kelly
    print("\n2. Continuous Kelly (Thorp's Formula):")
    continuous_result = sizer.kelly_criterion_continuous(
        mean_return=0.10,  # 10% expected return
        variance=0.04  # 4% variance (20% volatility)
    )
    print(f"   Position Value: ${continuous_result.position_value:.2f}")
    print(f"   Expected Return: ${continuous_result.expected_return:.2f}")
    
    # 3. Optimal F
    print("\n3. Optimal F:")
    trade_returns = [100, -50, 150, -30, 200, -60, 180, -40, 120, -20]
    optimal_f_result = sizer.optimal_f(trade_returns)
    print(f"   Position Value: ${optimal_f_result.position_value:.2f}")
    print(f"   Optimal F: {optimal_f_result.metadata['optimal_f']:.3f}")
    
    # 4. Risk Parity
    print("\n4. Risk Parity Sizing:")
    volatilities = {
        'BTC': 0.02,
        'ETH': 0.025,
        'SOL': 0.03
    }
    risk_parity_results = sizer.risk_parity_sizing(volatilities)
    for asset, result in risk_parity_results.items():
        print(f"   {asset}: ${result.position_value:.2f} (Risk: ${result.risk_amount:.2f})")
    
    # 5. Monte Carlo
    print("\n5. Monte Carlo Sizing:")
    mc_result = sizer.monte_carlo_sizing(
        expected_return=0.15,  # 15% annual return
        volatility=0.20,  # 20% annual volatility
        n_simulations=1000,
        max_drawdown_limit=0.15
    )
    print(f"   Position Value: ${mc_result.position_value:.2f}")
    print(f"   Confidence: {mc_result.confidence:.2%}")
    print(f"   Avg Drawdown: {mc_result.metadata.get('avg_drawdown', 0):.2%}")
    
    # 6. ML-based sizing
    print("\n6. ML-Based Sizing:")
    ml_result = sizer.ml_based_sizing(
        features={'volatility': 0.015, 'trend_strength': 0.8, 'volume_ratio': 1.2},
        model_confidence=0.75,
        predicted_return=0.08,
        prediction_std=0.12
    )
    print(f"   Position Value: ${ml_result.position_value:.2f}")
    print(f"   Model Confidence: {ml_result.confidence:.2%}")
    print(f"   Adjustments: {ml_result.adjustments_applied}")
    
    # Convert to shares
    btc_price = 45000
    shares = sizer.convert_to_shares(kelly_result.position_value, btc_price, lot_size=0.001)
    print(f"\n7. Convert to Shares:")
    print(f"   BTC Position: {shares:.6f} BTC @ ${btc_price}")
    print(f"   Total Value: ${shares * btc_price:.2f}")
    
    print("\n Position Sizer ready for integration!")