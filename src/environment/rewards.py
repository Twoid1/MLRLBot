"""
Reward Functions Module
Different reward formulations for the trading environment
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any


class RewardCalculator:
    """
    Collection of reward functions for trading environment
    """
    
    @staticmethod
    def simple_returns(current_value: float, 
                      previous_value: float,
                      scaling: float = 1.0) -> float:
        """
        Simple percentage returns
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            scaling: Reward scaling factor
            
        Returns:
            Scaled return
        """
        if previous_value == 0:
            return 0
        
        returns = (current_value - previous_value) / previous_value
        return returns * scaling
    
    @staticmethod
    def log_returns(current_value: float,
                   previous_value: float,
                   scaling: float = 1.0) -> float:
        """
        Logarithmic returns (more stable for RL)
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            scaling: Reward scaling factor
            
        Returns:
            Scaled log return
        """
        if previous_value <= 0 or current_value <= 0:
            return 0
        
        log_return = np.log(current_value / previous_value)
        return log_return * scaling
    
    @staticmethod
    def risk_adjusted_returns(portfolio_values: List[float],
                            risk_free_rate: float = 0.02,
                            periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio as reward
        
        Args:
            portfolio_values: List of portfolio values
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sharpe ratio
        """
        if len(portfolio_values) < 2:
            return 0
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # Adjust risk-free rate to period
        period_rf = risk_free_rate / periods_per_year
        
        # Calculate Sharpe ratio
        excess_returns = returns - period_rf
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        
        return sharpe
    
    @staticmethod
    def profit_factor_reward(winning_trades: List[float],
                            losing_trades: List[float]) -> float:
        """
        Calculate profit factor as reward
        
        Args:
            winning_trades: List of winning trade PnLs
            losing_trades: List of losing trade PnLs
            
        Returns:
            Profit factor
        """
        if not winning_trades:
            return -1 if losing_trades else 0
        
        if not losing_trades:
            return 1
        
        total_wins = sum(winning_trades)
        total_losses = abs(sum(losing_trades))
        
        if total_losses == 0:
            return 1
        
        return total_wins / total_losses - 1  # Subtract 1 to center around 0
    
    @staticmethod
    def drawdown_adjusted_returns(current_value: float,
                                 previous_value: float,
                                 current_drawdown: float,
                                 max_drawdown: float,
                                 drawdown_penalty: float = 2.0) -> float:
        """
        Returns adjusted for drawdown risk
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum drawdown percentage
            drawdown_penalty: Penalty multiplier for drawdowns
            
        Returns:
            Drawdown-adjusted return
        """
        returns = (current_value - previous_value) / previous_value if previous_value > 0 else 0
        
        # Apply drawdown penalty
        dd_penalty = current_drawdown * drawdown_penalty
        
        # Extra penalty if approaching max drawdown
        if max_drawdown > 0.3:  # 30% drawdown threshold
            dd_penalty += (max_drawdown - 0.3) * drawdown_penalty * 2
        
        return returns - dd_penalty
    
    @staticmethod
    def composite_reward(current_value: float,
                        previous_value: float,
                        portfolio_history: List[float],
                        trades: List[Dict],
                        current_drawdown: float,
                        weights: Optional[Dict[str, float]] = None) -> float:
        """
        Composite reward combining multiple objectives
        
        Args:
            current_value: Current portfolio value
            previous_value: Previous portfolio value
            portfolio_history: Historical portfolio values
            trades: List of completed trades
            current_drawdown: Current drawdown percentage
            weights: Optional weight dictionary for components
            
        Returns:
            Composite reward value
        """
        if weights is None:
            weights = {
                'returns': 1.0,
                'sharpe': 0.5,
                'drawdown': 0.3,
                'win_rate': 0.2,
                'efficiency': 0.1
            }
        
        components = {}
        
        # Base returns
        components['returns'] = RewardCalculator.simple_returns(current_value, previous_value)
        
        # Risk-adjusted returns (if enough history)
        if len(portfolio_history) >= 20:
            components['sharpe'] = RewardCalculator.risk_adjusted_returns(portfolio_history[-20:]) * 0.01
        else:
            components['sharpe'] = 0
        
        # Drawdown penalty
        components['drawdown'] = -current_drawdown * 2
        
        # Win rate bonus (if we have trades)
        if trades:
            winning = sum(1 for t in trades if t.get('pnl') is not None and t.get('pnl', 0) > 0)
            total = sum(1 for t in trades if t.get('pnl') is not None)
            if total > 0:
                components['win_rate'] = (winning / total - 0.5) * 0.1
            else:
                components['win_rate'] = 0
        else:
            components['win_rate'] = 0
        
        # Trade efficiency
        if trades and portfolio_history:
            total_return = (portfolio_history[-1] / portfolio_history[0] - 1) if portfolio_history[0] > 0 else 0
            components['efficiency'] = RewardCalculator.trade_efficiency_reward(
                len(trades), total_return, target_trades=100
            )
        else:
            components['efficiency'] = 0
        
        # Calculate weighted sum
        total_weight = sum(weights.values())
        reward = sum(
            weights.get(key, 0) * value 
            for key, value in components.items()
        ) / total_weight
        
        return reward
    
    @staticmethod
    def sortino_ratio_reward(portfolio_values: List[float],
                           target_return: float = 0,
                           periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (penalizes downside volatility)
        
        Args:
            portfolio_values: List of portfolio values
            target_return: Target return threshold
            periods_per_year: Number of trading periods per year
            
        Returns:
            Sortino ratio
        """
        if len(portfolio_values) < 2:
            return 0
        
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        if len(returns) == 0:
            return 0
        
        # Calculate downside returns
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return 1  # No downside risk
        
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return 1
        
        # Calculate Sortino ratio
        excess_return = returns.mean() - target_return
        sortino = (excess_return / downside_std) * np.sqrt(periods_per_year)
        
        return sortino
    
    @staticmethod
    def calmar_ratio_reward(total_return: float,
                          max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (return to max drawdown)
        
        Args:
            total_return: Total portfolio return
            max_drawdown: Maximum drawdown
            
        Returns:
            Calmar ratio
        """
        if max_drawdown == 0:
            return total_return
        
        return total_return / abs(max_drawdown)
    
    @staticmethod
    def trade_efficiency_reward(num_trades: int,
                              total_return: float,
                              target_trades: int = 100) -> float:
        """
        Reward efficiency (high returns with fewer trades)
        
        Args:
            num_trades: Number of trades executed
            total_return: Total portfolio return
            target_trades: Target number of trades
            
        Returns:
            Efficiency reward
        """
        if num_trades == 0:
            return 0
        
        # Penalize too many or too few trades
        trade_penalty = abs(num_trades - target_trades) / target_trades
        
        # Return per trade
        return_per_trade = total_return / num_trades
        
        # Combine
        efficiency = return_per_trade * (1 - trade_penalty * 0.5)
        
        return efficiency
    
    @staticmethod
    def sparse_reward(episode_return: float,
                     episode_length: int,
                     success_threshold: float = 0.05) -> float:
        """
        Sparse reward given only at episode end
        
        Args:
            episode_return: Total episode return
            episode_length: Number of steps in episode
            success_threshold: Return threshold for success
            
        Returns:
            Sparse reward
        """
        if episode_return > success_threshold:
            # Success bonus
            return 1.0 + episode_return
        elif episode_return > 0:
            # Small positive return
            return episode_return
        else:
            # Penalty for losses
            return episode_return - 0.5
    
    @staticmethod
    def curiosity_reward(state_visits: Dict[str, int],
                        current_state: str,
                        exploration_bonus: float = 0.1) -> float:
        """
        Add exploration bonus for visiting new states
        
        Args:
            state_visits: Dictionary tracking state visit counts
            current_state: Current state hash
            exploration_bonus: Bonus for new states
            
        Returns:
            Curiosity reward
        """
        visit_count = state_visits.get(current_state, 0)
        
        if visit_count == 0:
            return exploration_bonus
        else:
            # Decay bonus with visit count
            return exploration_bonus / (1 + visit_count)


class RewardShaper:
    """
    Shape rewards for better learning
    """
    
    @staticmethod
    def normalize_reward(reward: float,
                        min_reward: float = -1,
                        max_reward: float = 1) -> float:
        """
        Normalize reward to specified range
        
        Args:
            reward: Raw reward
            min_reward: Minimum reward value
            max_reward: Maximum reward value
            
        Returns:
            Normalized reward
        """
        return np.clip(reward, min_reward, max_reward)
    
    @staticmethod
    def exponential_scaling(reward: float,
                          temperature: float = 1.0) -> float:
        """
        Apply exponential scaling to reward
        
        Args:
            reward: Raw reward
            temperature: Scaling temperature
            
        Returns:
            Scaled reward
        """
        return np.sign(reward) * (1 - np.exp(-abs(reward) / temperature))
    
    @staticmethod
    def potential_based_shaping(current_potential: float,
                              previous_potential: float,
                              gamma: float = 0.99) -> float:
        """
        Potential-based reward shaping (preserves optimal policy)
        
        Args:
            current_potential: Current state potential
            previous_potential: Previous state potential
            gamma: Discount factor
            
        Returns:
            Shaped reward
        """
        return gamma * current_potential - previous_potential
    
    @staticmethod
    def adaptive_scaling(rewards_history: List[float],
                        current_reward: float,
                        window: int = 100) -> float:
        """
        Adaptively scale rewards based on recent history
        
        Args:
            rewards_history: History of rewards
            current_reward: Current reward
            window: Window for statistics
            
        Returns:
            Adaptively scaled reward
        """
        if len(rewards_history) < window:
            return current_reward
        
        recent_rewards = rewards_history[-window:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        if std_reward == 0:
            return current_reward
        
        # Z-score normalization
        return (current_reward - mean_reward) / std_reward


# Example usage
if __name__ == "__main__":
    # Test reward functions
    calculator = RewardCalculator()
    
    # Test simple returns
    reward = calculator.simple_returns(10500, 10000)
    print(f"Simple Returns: {reward:.4f}")
    
    # Test Sharpe ratio
    portfolio = [10000, 10100, 10050, 10200, 10150, 10300]
    sharpe = calculator.risk_adjusted_returns(portfolio)
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Test composite reward
    trades = [
        {'pnl': 100}, {'pnl': -50}, {'pnl': 200},
        {'pnl': -30}, {'pnl': 150}
    ]
    composite = calculator.composite_reward(
        current_value=10500,
        previous_value=10000,
        portfolio_history=portfolio,
        trades=trades,
        current_drawdown=0.05
    )
    print(f"Composite Reward: {composite:.4f}")
    
    print("\n Reward functions ready!")