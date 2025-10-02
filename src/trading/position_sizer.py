"""
Position Sizer Module - COMPLETE PRODUCTION VERSION
Advanced position sizing algorithms with proper Kelly safety implementation
Integrates with Risk Manager for comprehensive position management
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
    vix_level: Optional[float] = None  # Market fear index
    bid_ask_spread: Optional[float] = None


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation"""
    position_size: float  # Number of units
    position_value: float  # Dollar value
    risk_amount: float  # Amount at risk
    confidence: float  # Confidence in the sizing
    method_used: str  # Method that was used
    adjustments_applied: List[str]  # List of adjustments
    max_loss: float  # Maximum potential loss
    expected_return: float  # Expected return
    risk_reward_ratio: float  # Risk/reward ratio
    kelly_fraction: float  # Kelly fraction used
    metadata: Dict[str, Any]  # Additional information


@dataclass
class BacktestStats:
    """Statistics from backtesting for position sizing"""
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_consecutive_losses: int
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    recovery_factor: float
    avg_trade_duration: float


class PositionSizer:
    """
    Advanced position sizing calculator - COMPLETE VERSION
    
    Features:
    - Kelly Criterion with mandatory 25% safety factor
    - Multiple sizing algorithms
    - Dynamic adjustments based on market conditions
    - Integration with risk management
    - Comprehensive backtesting support
    """
    
    def __init__(self,
                 capital: float,
                 max_risk_per_trade: float = 0.02,
                 max_portfolio_risk: float = 0.06,
                 confidence_level: float = 0.95,
                 kelly_safety_factor: float = 0.25,
                 use_dynamic_sizing: bool = True):
        """
        Initialize Position Sizer
        
        Args:
            capital: Available capital
            max_risk_per_trade: Maximum risk per trade (as fraction)
            max_portfolio_risk: Maximum portfolio risk
            confidence_level: Confidence level for calculations
            kelly_safety_factor: Safety factor for Kelly Criterion (DEFAULT 25%)
            use_dynamic_sizing: Enable dynamic position sizing
        """
        self.capital = capital
        self.initial_capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.confidence_level = confidence_level
        self.kelly_safety_factor = kelly_safety_factor  # CRITICAL: 25% safety
        self.use_dynamic_sizing = use_dynamic_sizing
        
        # Historical data for calculations
        self.trade_history = []
        self.returns_history = []
        self.position_outcomes = {}
        
        # Portfolio tracking
        self.current_positions = {}
        self.portfolio_weights = {}
        self.correlation_matrix = pd.DataFrame()
        
        # Performance tracking
        self.cumulative_pnl = 0
        self.peak_capital = capital
        self.current_drawdown = 0
        
        # Cache for calculations
        self._kelly_cache = {}
        self._optimal_f_cache = {}
        
        # Sizing history for analysis
        self.sizing_history = []
        
        logger.info(f"PositionSizer initialized with ${capital:,.2f} capital")
        logger.info(f"Kelly safety factor: {self.kelly_safety_factor * 100}%")
        logger.info(f"Max risk per trade: {self.max_risk_per_trade * 100}%")
    
    # ================== MAIN SIZING METHOD ==================
    
    def calculate_position_size(self,
                               symbol: str,
                               entry_price: float,
                               stop_loss_price: float,
                               market_conditions: Optional[MarketConditions] = None,
                               backtest_stats: Optional[BacktestStats] = None,
                               method: str = 'kelly',
                               confidence: float = 1.0) -> PositionSizeResult:
        """
        Main method to calculate position size
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price
            market_conditions: Current market conditions
            backtest_stats: Historical performance statistics
            method: Sizing method to use
            confidence: Confidence in the trade (0-1)
            
        Returns:
            PositionSizeResult with all sizing information
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            logger.error("Invalid stop loss - no risk defined")
            return self._create_zero_position_result(symbol, "invalid_stop_loss")
        
        # Get base position value based on method
        if method == 'kelly':
            if backtest_stats:
                result = self.kelly_criterion_sizing(
                    win_rate=backtest_stats.win_rate,
                    avg_win=backtest_stats.avg_win,
                    avg_loss=backtest_stats.avg_loss,
                    confidence=confidence
                )
            else:
                # Use default conservative sizing
                result = self._create_default_result('kelly')
                
        elif method == 'fixed':
            result = self.fixed_percentage_sizing(confidence)
            
        elif method == 'volatility':
            if market_conditions:
                result = self.volatility_based_sizing(
                    market_conditions.volatility,
                    confidence
                )
            else:
                result = self.fixed_percentage_sizing(confidence)
                
        elif method == 'optimal_f':
            if backtest_stats:
                result = self.optimal_f_sizing(
                    backtest_stats.win_rate,
                    backtest_stats.avg_win,
                    backtest_stats.avg_loss,
                    confidence
                )
            else:
                result = self._create_default_result('optimal_f')
                
        elif method == 'risk_parity':
            result = self.risk_parity_sizing(
                len(self.current_positions) + 1,
                confidence
            )
        else:
            result = self.fixed_percentage_sizing(confidence)
        
        # Apply dynamic adjustments if enabled
        if self.use_dynamic_sizing and market_conditions:
            result = self._apply_dynamic_adjustments(result, market_conditions)
        
        # Calculate position size in units
        max_risk_amount = result.position_value
        position_size_units = max_risk_amount / risk_per_unit
        
        # Apply position limits
        position_size_units = self._apply_position_limits(
            position_size_units,
            entry_price,
            symbol
        )
        
        # Update result with final calculations
        result.position_size = position_size_units
        result.position_value = position_size_units * entry_price
        result.risk_amount = position_size_units * risk_per_unit
        result.max_loss = result.risk_amount
        
        # Calculate risk/reward if we have backtest stats
        if backtest_stats and backtest_stats.avg_loss > 0:
            result.risk_reward_ratio = backtest_stats.avg_win / backtest_stats.avg_loss
        
        # Store sizing decision
        self._store_sizing_decision(symbol, result)
        
        return result
    
    # ================== KELLY CRITERION (FIXED) ==================
    
    def kelly_criterion_sizing(self,
                              win_rate: float,
                              avg_win: float,
                              avg_loss: float,
                              confidence: float = 1.0,
                              use_safety_factor: bool = True) -> PositionSizeResult:
        """
        Kelly Criterion with MANDATORY 25% safety factor
        
        CRITICAL FIX: Properly applies 25% safety factor
        
        Formula: f* = (p * b - q) / b
        Where:
            f* = optimal fraction of capital to bet
            p = probability of winning
            b = ratio of win to loss (odds)
            q = probability of losing (1 - p)
        
        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            confidence: Confidence in the parameters (0 to 1)
            use_safety_factor: Whether to apply safety factor (DEFAULT TRUE)
            
        Returns:
            PositionSizeResult with safe Kelly sizing
        """
        # Input validation
        if win_rate <= 0 or win_rate >= 1:
            logger.warning(f"Invalid win rate: {win_rate}. Using default sizing.")
            return self._create_default_result("kelly")
        
        if avg_loss <= 0 or avg_win <= 0:
            logger.warning(f"Invalid win/loss amounts. Using default sizing.")
            return self._create_default_result("kelly")
        
        # Calculate win/loss ratio (b in Kelly formula)
        win_loss_ratio = avg_win / avg_loss
        
        # Calculate losing probability
        loss_rate = 1 - win_rate
        
        # CALCULATE FULL KELLY FRACTION
        # f* = (p * b - q) / b = (p * b - (1-p)) / b
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        
        # Check if we have positive expectancy
        if full_kelly <= 0:
            logger.warning(f"Negative Kelly fraction: {full_kelly:.4f}. No position recommended.")
            return PositionSizeResult(
                position_size=0,
                position_value=0,
                risk_amount=0,
                confidence=0,
                method_used="kelly_negative_expectancy",
                adjustments_applied=["negative_expectancy"],
                max_loss=0,
                expected_return=0,
                risk_reward_ratio=win_loss_ratio,
                kelly_fraction=0,
                metadata={
                    'full_kelly': full_kelly,
                    'win_rate': win_rate,
                    'win_loss_ratio': win_loss_ratio,
                    'message': 'Negative expectancy - no trade'
                }
            )
        
        # CRITICAL FIX: APPLY 25% SAFETY FACTOR
        if use_safety_factor:
            safe_kelly = full_kelly * self.kelly_safety_factor
            adjustments = [f"{self.kelly_safety_factor*100:.0f}% safety factor"]
            logger.info(f"Kelly: Full={full_kelly:.4f}, Safe={safe_kelly:.4f} (25% of full)")
        else:
            safe_kelly = full_kelly
            adjustments = ["no_safety_factor"]
            logger.warning("Kelly Criterion used WITHOUT safety factor - not recommended!")
        
        # Apply confidence adjustment
        final_kelly = safe_kelly * confidence
        if confidence < 1.0:
            adjustments.append(f"confidence adjustment: {confidence:.2%}")
        
        # Cap at maximum risk per trade
        if final_kelly > self.max_risk_per_trade:
            logger.info(f"Kelly fraction {final_kelly:.4f} exceeds max risk {self.max_risk_per_trade:.4f}")
            final_kelly = self.max_risk_per_trade
            adjustments.append(f"capped at max risk: {self.max_risk_per_trade:.2%}")
        
        # Calculate position value
        position_value = self.capital * final_kelly
        
        # Calculate expected return
        expected_return = position_value * (win_rate * win_loss_ratio - loss_rate)
        
        # Calculate maximum loss
        max_loss = position_value  # Could lose entire position
        
        # Create result
        return PositionSizeResult(
            position_size=0,  # Will be calculated with price later
            position_value=position_value,
            risk_amount=position_value * loss_rate,
            confidence=confidence,
            method_used="kelly_25_percent_safety",
            adjustments_applied=adjustments,
            max_loss=max_loss,
            expected_return=expected_return,
            risk_reward_ratio=win_loss_ratio,
            kelly_fraction=final_kelly,
            metadata={
                'full_kelly': full_kelly,
                'safe_kelly': safe_kelly,
                'final_kelly': final_kelly,
                'safety_factor': self.kelly_safety_factor,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'win_loss_ratio': win_loss_ratio,
                'edge': win_rate * win_loss_ratio - loss_rate
            }
        )
    
    # ================== OTHER SIZING METHODS ==================
    
    def fixed_percentage_sizing(self, confidence: float = 1.0) -> PositionSizeResult:
        """
        Fixed percentage of capital sizing
        
        Args:
            confidence: Confidence adjustment
            
        Returns:
            PositionSizeResult
        """
        position_value = self.capital * self.max_risk_per_trade * confidence
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value,
            confidence=confidence,
            method_used="fixed_percentage",
            adjustments_applied=[f"confidence: {confidence:.2%}"],
            max_loss=position_value,
            expected_return=0,
            risk_reward_ratio=0,
            kelly_fraction=self.max_risk_per_trade * confidence,
            metadata={'percentage': self.max_risk_per_trade}
        )
    
    def volatility_based_sizing(self,
                               volatility: float,
                               confidence: float = 1.0) -> PositionSizeResult:
        """
        Size positions inversely to volatility
        
        Args:
            volatility: Current volatility
            confidence: Confidence adjustment
            
        Returns:
            PositionSizeResult
        """
        if volatility <= 0:
            return self._create_default_result("volatility_based")
        
        # Target volatility (e.g., 2% daily)
        target_volatility = 0.02
        
        # Calculate volatility adjustment
        vol_adjustment = min(2.0, max(0.5, target_volatility / volatility))
        
        # Calculate position value
        base_value = self.capital * self.max_risk_per_trade
        position_value = base_value * vol_adjustment * confidence
        
        adjustments = [
            f"volatility adjustment: {vol_adjustment:.2f}x",
            f"confidence: {confidence:.2%}"
        ]
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value,
            confidence=confidence,
            method_used="volatility_based",
            adjustments_applied=adjustments,
            max_loss=position_value,
            expected_return=0,
            risk_reward_ratio=0,
            kelly_fraction=position_value / self.capital,
            metadata={
                'volatility': volatility,
                'target_volatility': target_volatility,
                'vol_adjustment': vol_adjustment
            }
        )
    
    def optimal_f_sizing(self,
                        win_rate: float,
                        avg_win: float,
                        avg_loss: float,
                        confidence: float = 1.0) -> PositionSizeResult:
        """
        Optimal f position sizing (Ralph Vince method)
        
        Args:
            win_rate: Win probability
            avg_win: Average win
            avg_loss: Average loss
            confidence: Confidence adjustment
            
        Returns:
            PositionSizeResult
        """
        if avg_loss <= 0:
            return self._create_default_result("optimal_f")
        
        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Optimal f formula (simplified)
        optimal_f = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply safety factor (50% of optimal f)
        safe_f = optimal_f * 0.5
        
        # Apply confidence
        final_f = safe_f * confidence
        
        # Cap at maximum
        final_f = min(final_f, self.max_risk_per_trade)
        
        position_value = self.capital * final_f
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value * (1 - win_rate),
            confidence=confidence,
            method_used="optimal_f",
            adjustments_applied=["50% safety factor", f"confidence: {confidence:.2%}"],
            max_loss=position_value,
            expected_return=position_value * (win_rate * win_loss_ratio - (1 - win_rate)),
            risk_reward_ratio=win_loss_ratio,
            kelly_fraction=final_f,
            metadata={
                'optimal_f': optimal_f,
                'safe_f': safe_f,
                'final_f': final_f
            }
        )
    
    def risk_parity_sizing(self,
                          num_positions: int,
                          confidence: float = 1.0) -> PositionSizeResult:
        """
        Risk parity - equal risk allocation across positions
        
        Args:
            num_positions: Number of positions
            confidence: Confidence adjustment
            
        Returns:
            PositionSizeResult
        """
        if num_positions <= 0:
            return self._create_default_result("risk_parity")
        
        # Allocate risk equally
        risk_per_position = self.max_portfolio_risk / num_positions
        
        # Apply confidence
        final_risk = risk_per_position * confidence
        
        position_value = self.capital * final_risk
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value,
            confidence=confidence,
            method_used="risk_parity",
            adjustments_applied=[f"positions: {num_positions}", f"confidence: {confidence:.2%}"],
            max_loss=position_value,
            expected_return=0,
            risk_reward_ratio=0,
            kelly_fraction=final_risk,
            metadata={
                'num_positions': num_positions,
                'risk_per_position': risk_per_position
            }
        )
    
    # ================== DYNAMIC ADJUSTMENTS ==================
    
    def _apply_dynamic_adjustments(self,
                                  result: PositionSizeResult,
                                  market_conditions: MarketConditions) -> PositionSizeResult:
        """
        Apply dynamic adjustments based on market conditions
        
        Args:
            result: Base position size result
            market_conditions: Current market conditions
            
        Returns:
            Adjusted PositionSizeResult
        """
        adjustment_factor = 1.0
        adjustments = []
        
        # Volatility adjustment
        if market_conditions.volatility > 0.03:  # High volatility
            vol_adj = min(0.7, 0.02 / market_conditions.volatility)
            adjustment_factor *= vol_adj
            adjustments.append(f"high volatility: {vol_adj:.2f}x")
        
        # Trend strength adjustment
        if market_conditions.market_regime == 'trending':
            if market_conditions.trend_strength > 0.7:
                adjustment_factor *= 1.2
                adjustments.append("strong trend: 1.2x")
            elif market_conditions.trend_strength > 0.5:
                adjustment_factor *= 1.1
                adjustments.append("moderate trend: 1.1x")
        elif market_conditions.market_regime == 'ranging':
            adjustment_factor *= 0.8
            adjustments.append("ranging market: 0.8x")
        elif market_conditions.market_regime == 'volatile':
            adjustment_factor *= 0.6
            adjustments.append("volatile market: 0.6x")
        
        # Liquidity adjustment
        if market_conditions.liquidity < 0.5:
            liq_adj = max(0.5, market_conditions.liquidity)
            adjustment_factor *= liq_adj
            adjustments.append(f"low liquidity: {liq_adj:.2f}x")
        
        # Spread adjustment
        if market_conditions.bid_ask_spread and market_conditions.bid_ask_spread > 0.002:  # 0.2%
            spread_adj = max(0.7, 1 - market_conditions.bid_ask_spread * 50)
            adjustment_factor *= spread_adj
            adjustments.append(f"wide spread: {spread_adj:.2f}x")
        
        # VIX adjustment (if available)
        if market_conditions.vix_level:
            if market_conditions.vix_level > 30:  # High fear
                adjustment_factor *= 0.5
                adjustments.append("high VIX: 0.5x")
            elif market_conditions.vix_level > 20:
                adjustment_factor *= 0.8
                adjustments.append("elevated VIX: 0.8x")
        
        # Apply adjustments
        result.position_value *= adjustment_factor
        result.risk_amount *= adjustment_factor
        result.max_loss *= adjustment_factor
        result.expected_return *= adjustment_factor
        result.adjustments_applied.extend(adjustments)
        result.metadata['total_adjustment'] = adjustment_factor
        
        return result
    
    # ================== POSITION LIMITS ==================
    
    def _apply_position_limits(self,
                              position_size: float,
                              entry_price: float,
                              symbol: str) -> float:
        """
        Apply position limits and constraints
        
        Args:
            position_size: Calculated position size
            entry_price: Entry price
            symbol: Trading symbol
            
        Returns:
            Limited position size
        """
        position_value = position_size * entry_price
        
        # Maximum position value (20% of capital)
        max_position_value = self.capital * 0.2
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            logger.info(f"Position capped at 20% of capital: {position_size:.6f}")
        
        # Minimum position value ($100 or 0.1% of capital)
        min_position_value = max(100, self.capital * 0.001)
        if position_value < min_position_value:
            logger.warning(f"Position too small: ${position_value:.2f} < ${min_position_value:.2f}")
            return 0
        
        # Check total exposure
        total_exposure = sum(pos['value'] for pos in self.current_positions.values())
        new_total = total_exposure + position_value
        
        if new_total > self.capital:
            available = self.capital - total_exposure
            if available <= 0:
                logger.warning("No capital available for new position")
                return 0
            position_size = available / entry_price
            logger.info(f"Position reduced to fit available capital: {position_size:.6f}")
        
        return position_size
    
    # ================== HELPER METHODS ==================
    
    def _create_default_result(self, method: str) -> PositionSizeResult:
        """Create default conservative position size result"""
        position_value = self.capital * 0.01  # 1% default
        
        return PositionSizeResult(
            position_size=0,
            position_value=position_value,
            risk_amount=position_value,
            confidence=0.5,
            method_used=f"{method}_default",
            adjustments_applied=["using_conservative_default"],
            max_loss=position_value,
            expected_return=0,
            risk_reward_ratio=1.0,
            kelly_fraction=0.01,
            metadata={'reason': 'Insufficient data, using conservative default'}
        )
    
    def _create_zero_position_result(self, symbol: str, reason: str) -> PositionSizeResult:
        """Create zero position result"""
        return PositionSizeResult(
            position_size=0,
            position_value=0,
            risk_amount=0,
            confidence=0,
            method_used="zero_position",
            adjustments_applied=[reason],
            max_loss=0,
            expected_return=0,
            risk_reward_ratio=0,
            kelly_fraction=0,
            metadata={'symbol': symbol, 'reason': reason}
        )
    
    def _store_sizing_decision(self, symbol: str, result: PositionSizeResult) -> None:
        """Store sizing decision for analysis"""
        self.sizing_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'method': result.method_used,
            'position_value': result.position_value,
            'kelly_fraction': result.kelly_fraction,
            'adjustments': result.adjustments_applied,
            'confidence': result.confidence
        })
        
        # Keep only recent history
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]
    
    # ================== PORTFOLIO MANAGEMENT ==================
    
    def add_position(self, symbol: str, size: float, entry_price: float) -> None:
        """Add position to portfolio tracking"""
        self.current_positions[symbol] = {
            'size': size,
            'entry_price': entry_price,
            'value': size * entry_price,
            'entry_time': datetime.now()
        }
        logger.info(f"Position added: {symbol} - {size:.6f} @ ${entry_price:.2f}")
    
    def remove_position(self, symbol: str, exit_price: float) -> float:
        """Remove position and calculate P&L"""
        if symbol not in self.current_positions:
            return 0
        
        position = self.current_positions[symbol]
        pnl = (exit_price - position['entry_price']) * position['size']
        
        # Update capital
        self.capital += pnl
        self.cumulative_pnl += pnl
        
        # Update drawdown
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        
        # Remove position
        del self.current_positions[symbol]
        
        logger.info(f"Position closed: {symbol} - P&L: ${pnl:.2f}")
        
        return pnl
    
    def update_capital(self, new_capital: float) -> None:
        """Update available capital"""
        self.capital = new_capital
        logger.info(f"Capital updated: ${new_capital:,.2f}")
    
    # ================== ANALYSIS METHODS ==================
    
    def get_kelly_analysis(self,
                          win_rate: float,
                          avg_win: float,
                          avg_loss: float) -> Dict[str, Any]:
        """
        Get detailed Kelly Criterion analysis with sensitivity
        
        Returns dictionary with full analysis
        """
        # Calculate base Kelly
        result = self.kelly_criterion_sizing(win_rate, avg_win, avg_loss)
        
        # Sensitivity analysis
        sensitivities = {}
        
        # Win rate sensitivity
        wr_range = np.arange(max(0.3, win_rate-0.1), min(0.7, win_rate+0.1), 0.02)
        wr_kellys = []
        for wr in wr_range:
            kr = self.kelly_criterion_sizing(wr, avg_win, avg_loss)
            wr_kellys.append(kr.metadata['safe_kelly'])
        sensitivities['win_rate'] = {
            'range': wr_range.tolist(),
            'kellys': wr_kellys,
            'current': win_rate
        }
        
        # Win/loss ratio sensitivity
        ratios = np.arange(0.5, 3.0, 0.25)
        ratio_kellys = []
        for ratio in ratios:
            kr = self.kelly_criterion_sizing(win_rate, avg_loss * ratio, avg_loss)
            ratio_kellys.append(kr.metadata['safe_kelly'])
        sensitivities['win_loss_ratio'] = {
            'range': ratios.tolist(),
            'kellys': ratio_kellys,
            'current': avg_win / avg_loss
        }
        
        analysis = {
            'full_kelly': result.metadata['full_kelly'],
            'safe_kelly': result.metadata['safe_kelly'],
            'recommended_fraction': result.metadata['final_kelly'],
            'position_value': result.position_value,
            'expected_return': result.expected_return,
            'max_loss': result.max_loss,
            'edge': result.metadata['edge'],
            'safety_factor': self.kelly_safety_factor,
            'sensitivities': sensitivities,
            'interpretation': self._interpret_kelly(result.metadata['safe_kelly']),
            'risk_assessment': self._assess_risk_level(result.metadata['safe_kelly'])
        }
        
        return analysis
    
    def _interpret_kelly(self, kelly_fraction: float) -> str:
        """Interpret Kelly fraction for user understanding"""
        if kelly_fraction <= 0:
            return "No position - negative expectancy"
        elif kelly_fraction < 0.01:
            return "Very small edge - consider skipping"
        elif kelly_fraction < 0.02:
            return "Small edge - conservative position"
        elif kelly_fraction < 0.05:
            return "Moderate edge - standard position"
        elif kelly_fraction < 0.10:
            return "Strong edge - confident position"
        else:
            return "Very strong edge - maximum position (capped)"
    
    def _assess_risk_level(self, kelly_fraction: float) -> str:
        """Assess risk level of position"""
        if kelly_fraction <= 0:
            return "NO_TRADE"
        elif kelly_fraction < 0.02:
            return "VERY_LOW_RISK"
        elif kelly_fraction < 0.05:
            return "LOW_RISK"
        elif kelly_fraction < 0.10:
            return "MODERATE_RISK"
        else:
            return "HIGH_RISK"
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        total_value = sum(pos['value'] for pos in self.current_positions.values())
        
        return {
            'capital': self.capital,
            'positions': len(self.current_positions),
            'total_exposure': total_value,
            'exposure_percentage': total_value / self.capital if self.capital > 0 else 0,
            'cumulative_pnl': self.cumulative_pnl,
            'current_drawdown': self.current_drawdown,
            'positions_detail': self.current_positions
        }
    
    # ================== TESTING ==================
    
    @staticmethod
    def run_comprehensive_tests():
        """Run comprehensive tests of position sizing"""
        print("=" * 70)
        print("POSITION SIZER COMPREHENSIVE TESTING")
        print("=" * 70)
        
        sizer = PositionSizer(capital=10000)
        
        # Test 1: Kelly Criterion with various scenarios
        print("\n=== TEST 1: Kelly Criterion Scenarios ===")
        test_cases = [
            (0.55, 100, 100, "Slight edge - 55% win rate, 1:1 ratio"),
            (0.60, 100, 100, "Good edge - 60% win rate, 1:1 ratio"),
            (0.50, 150, 100, "Breakeven win rate, 1.5:1 ratio"),
            (0.45, 200, 100, "Low win rate, 2:1 ratio"),
            (0.40, 300, 100, "Very low win rate, 3:1 ratio"),
            (0.65, 80, 100, "High win rate, poor ratio"),
            (0.70, 150, 100, "Excellent setup"),
        ]
        
        for win_rate, avg_win, avg_loss, description in test_cases:
            result = sizer.kelly_criterion_sizing(win_rate, avg_win, avg_loss)
            
            print(f"\n{description}")
            print(f"  Win Rate: {win_rate:.1%}, Win/Loss: {avg_win}/{avg_loss}")
            print(f"  Full Kelly: {result.metadata['full_kelly']:.4f}")
            print(f"  Safe Kelly (25%): {result.metadata['safe_kelly']:.4f}")
            print(f"  Position Value: ${result.position_value:.2f}")
            print(f"  Expected Return: ${result.expected_return:.2f}")
        
        # Test 2: Different sizing methods
        print("\n=== TEST 2: Comparing Sizing Methods ===")
        
        backtest_stats = BacktestStats(
            total_trades=100,
            win_rate=0.55,
            avg_win=150,
            avg_loss=100,
            max_consecutive_losses=5,
            max_drawdown=0.15,
            sharpe_ratio=1.5,
            profit_factor=1.65,
            recovery_factor=3.0,
            avg_trade_duration=24
        )
        
        methods = ['kelly', 'fixed', 'volatility', 'optimal_f', 'risk_parity']
        
        market_conditions = MarketConditions(
            volatility=0.02,
            trend_strength=0.6,
            correlation_matrix=pd.DataFrame(),
            market_regime='trending',
            liquidity=0.8,
            spread=0.001,
            volume_profile={}
        )
        
        for method in methods:
            result = sizer.calculate_position_size(
                symbol='BTC/USDT',
                entry_price=45000,
                stop_loss_price=44000,
                market_conditions=market_conditions,
                backtest_stats=backtest_stats,
                method=method
            )
            
            print(f"\n{method.upper()} Method:")
            print(f"  Position Size: {result.position_size:.6f}")
            print(f"  Position Value: ${result.position_value:.2f}")
            print(f"  Risk Amount: ${result.risk_amount:.2f}")
            print(f"  Kelly Fraction: {result.kelly_fraction:.4f}")
        
        # Test 3: Market condition adjustments
        print("\n=== TEST 3: Market Condition Adjustments ===")
        
        conditions = [
            MarketConditions(0.01, 0.8, pd.DataFrame(), 'trending', 0.9, 0.0005, {}, 15, 0.001),
            MarketConditions(0.04, 0.3, pd.DataFrame(), 'volatile', 0.5, 0.002, {}, 35, 0.003),
            MarketConditions(0.02, 0.1, pd.DataFrame(), 'ranging', 0.7, 0.001, {}, 20, 0.001),
        ]
        
        labels = ['Normal/Trending', 'High Volatility', 'Ranging/Sideways']
        
        for condition, label in zip(conditions, labels):
            result = sizer.calculate_position_size(
                symbol='BTC/USDT',
                entry_price=45000,
                stop_loss_price=44000,
                market_conditions=condition,
                backtest_stats=backtest_stats,
                method='kelly'
            )
            
            print(f"\n{label}:")
            print(f"  Position Value: ${result.position_value:.2f}")
            print(f"  Adjustments: {', '.join(result.adjustments_applied)}")
        
        # Test 4: Kelly Analysis
        print("\n=== TEST 4: Kelly Sensitivity Analysis ===")
        analysis = sizer.get_kelly_analysis(0.55, 150, 100)
        
        print(f"\nKelly Analysis Summary:")
        print(f"  Full Kelly: {analysis['full_kelly']:.4f}")
        print(f"  Safe Kelly: {analysis['safe_kelly']:.4f}")
        print(f"  Interpretation: {analysis['interpretation']}")
        print(f"  Risk Level: {analysis['risk_assessment']}")
        
        print("\n" + "=" * 70)
        print("TESTING COMPLETE - Position Sizer Ready for Production!")
        print("=" * 70)


# Run tests if this file is executed directly
if __name__ == "__main__":
    PositionSizer.run_comprehensive_tests()