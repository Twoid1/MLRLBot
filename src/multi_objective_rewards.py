"""
Multi-Objective Reward System for Trade-Based RL

Instead of a single scalar reward, this system provides 5 separate reward signals:

1. PNL_QUALITY      - How good was the profit/loss amount?
2. HOLD_DURATION    - Did you hold long enough?
3. WIN_ACHIEVED     - Did you win the trade?
4. LOSS_CONTROL     - If you lost, did you minimize the damage?
5. RISK_REWARD      - Was the risk/reward ratio sensible?

Each objective is trained separately, allowing the agent to:
- See WHICH aspects of trading it's good/bad at
- Balance competing objectives (e.g., "hold longer" vs "take profit")
- Learn nuanced behavior that single-reward systems can't achieve

Author: Claude
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum


class RewardObjective(Enum):
    """The 5 reward objectives"""
    PNL_QUALITY = "pnl_quality"
    HOLD_DURATION = "hold_duration"
    WIN_ACHIEVED = "win_achieved"
    LOSS_CONTROL = "loss_control"
    RISK_REWARD = "risk_reward"


@dataclass
class MultiObjectiveConfig:
    """
    Configuration for Multi-Objective Reward System
    
    Each objective has its own parameters for calculating rewards.
    Weights determine how objectives are combined for action selection.
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OBJECTIVE WEIGHTS (for combining into action selection)
    # ═══════════════════════════════════════════════════════════════════════════
    # These determine the IMPORTANCE of each objective when choosing actions
    # Higher weight = more influence on decisions
    
    weight_pnl_quality: float = 0.40      # P&L is important but not everything
    weight_hold_duration: float = 0.05   # Holding matters a lot (our main issue!)
    weight_win_achieved: float = 0.15     # Win rate matters somewhat
    weight_loss_control: float = 0.20     # Cutting losers matters
    weight_risk_reward: float = 0.20      # Risk management matters
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PNL_QUALITY PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    # Rewards proportional to P&L, teaching "bigger wins and smaller losses"
    
    pnl_scale: float = 100.0              # Scale factor (1% = 1.0 reward)
    pnl_win_multiplier: float = 1.0       # Multiplier for winning P&L
    pnl_loss_multiplier: float = 1.0      # Multiplier for losing P&L (proportional!)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HOLD_DURATION PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    # Rewards for holding trades longer, REGARDLESS of P&L outcome
    
    hold_target_steps: int = 36           # Target hold duration (36 × 5m = 3 hours)
    hold_min_steps: int = 12              # Minimum before any bonus (1 hour)
    hold_max_reward: float = 2.0          # Maximum hold duration reward
    hold_too_short_penalty: float = -1.0  # Penalty for exiting too early
    hold_per_step_bonus: float = 0.05     # Bonus per step after minimum
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WIN_ACHIEVED PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    # Binary signal: did you win or lose?
    
    win_reward: float = 1.0               # Reward for winning trade
    loss_penalty: float = -0.5            # Penalty for losing trade
    breakeven_reward: float = 0.0         # Reward for breakeven (rare)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOSS_CONTROL PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    # For losing trades: rewards cutting early, penalizes holding to stop-loss
    
    stop_loss_pct: float = 0.03           # Stop-loss level (3%)
    loss_control_max_reward: float = 1.0  # Max reward for cutting at minimal loss
    loss_control_sl_penalty: float = -0.5 # Penalty for hitting stop-loss
    loss_control_scale: float = 1.0       # Scale factor for loss control rewards
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RISK_REWARD PARAMETERS
    # ═══════════════════════════════════════════════════════════════════════════
    # Rewards good risk/reward ratios
    
    take_profit_pct: float = 0.03         # Take-profit level (3%)
    rr_excellent_threshold: float = 2.0   # R:R > 2.0 is excellent
    rr_good_threshold: float = 1.0        # R:R > 1.0 is good
    rr_excellent_reward: float = 1.0      # Reward for excellent R:R
    rr_good_reward: float = 0.5           # Reward for good R:R
    rr_poor_penalty: float = -0.5         # Penalty for poor R:R
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL BONUSES
    # ═══════════════════════════════════════════════════════════════════════════
    
    take_profit_bonus: float = 1.5        # Bonus for hitting take-profit (to hold_duration)
    agent_exit_winner_bonus: float = 0.3  # Small bonus for agent exiting profitably
    
    def get_weights(self) -> Dict[str, float]:
        """Get objective weights as dictionary"""
        return {
            RewardObjective.PNL_QUALITY.value: self.weight_pnl_quality,
            RewardObjective.HOLD_DURATION.value: self.weight_hold_duration,
            RewardObjective.WIN_ACHIEVED.value: self.weight_win_achieved,
            RewardObjective.LOSS_CONTROL.value: self.weight_loss_control,
            RewardObjective.RISK_REWARD.value: self.weight_risk_reward,
        }
    
    def get_weights_tensor(self) -> list:
        """Get weights as list for tensor operations"""
        return [
            self.weight_pnl_quality,
            self.weight_hold_duration,
            self.weight_win_achieved,
            self.weight_loss_control,
            self.weight_risk_reward,
        ]


class MultiObjectiveRewardCalculator:
    """
    Calculates decomposed rewards across 5 objectives
    
    Usage:
        calculator = MultiObjectiveRewardCalculator(config)
        rewards = calculator.calculate(
            pnl_pct=0.015,          # 1.5% profit
            hold_duration=25,        # Held 25 steps
            exit_reason='agent',     # Agent decided to exit
            entry_price=100.0,
            exit_price=101.5
        )
        
        # rewards = {
        #     'pnl_quality': 1.5,
        #     'hold_duration': 0.65,
        #     'win_achieved': 1.0,
        #     'loss_control': 0.0,
        #     'risk_reward': 0.5,
        # }
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        self.config = config or MultiObjectiveConfig()
        
    def calculate(
        self,
        pnl_pct: float,
        hold_duration: int,
        exit_reason: str,
        entry_price: float,
        exit_price: float,
        max_favorable_excursion: Optional[float] = None,
        max_adverse_excursion: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate all 5 reward objectives
        
        Args:
            pnl_pct: Profit/loss as percentage (0.015 = 1.5%)
            hold_duration: Number of steps trade was held
            exit_reason: 'agent', 'stop_loss', 'take_profit', 'timeout'
            entry_price: Entry price
            exit_price: Exit price
            max_favorable_excursion: Best unrealized P&L during trade (optional)
            max_adverse_excursion: Worst unrealized P&L during trade (optional)
            
        Returns:
            Dictionary with reward for each objective
        """
        
        is_winner = pnl_pct > 0
        is_loser = pnl_pct < 0
        
        rewards = {
            RewardObjective.PNL_QUALITY.value: self._calc_pnl_quality(pnl_pct, is_winner),
            RewardObjective.HOLD_DURATION.value: self._calc_hold_duration(
                hold_duration, exit_reason, is_winner
            ),
            RewardObjective.WIN_ACHIEVED.value: self._calc_win_achieved(pnl_pct),
            RewardObjective.LOSS_CONTROL.value: self._calc_loss_control(
                pnl_pct, exit_reason, is_loser
            ),
            RewardObjective.RISK_REWARD.value: self._calc_risk_reward(
                pnl_pct, hold_duration, max_favorable_excursion, max_adverse_excursion
            ),
        }
        
        return rewards
    
    def _calc_pnl_quality(self, pnl_pct: float, is_winner: bool) -> float:
        """
        PNL_QUALITY: Proportional to actual P&L
        
        This teaches: "Bigger wins are better, bigger losses are worse"
        
        Examples:
            +2.0% → +2.0 reward
            +0.5% → +0.5 reward
            -0.5% → -0.5 reward
            -2.0% → -2.0 reward
        """
        if is_winner:
            return pnl_pct * self.config.pnl_scale * self.config.pnl_win_multiplier
        else:
            return pnl_pct * self.config.pnl_scale * self.config.pnl_loss_multiplier
    
    def _calc_hold_duration(
        self, 
        hold_duration: int, 
        exit_reason: str,
        is_winner: bool
    ) -> float:
        """
        HOLD_DURATION: Rewards holding regardless of P&L outcome
        
        This teaches: "Holding longer is good (to a point)"
        
        Examples:
            5 steps  → -1.0 (too short!)
            12 steps → 0.0 (minimum)
            24 steps → +0.6 (good)
            36 steps → +1.2 (target reached)
            50 steps → +1.7 (great)
            
        Special: Take-profit hit gets bonus
        """
        cfg = self.config
        
        # Too short - penalty
        if hold_duration < cfg.hold_min_steps:
            # Scale penalty based on how early
            early_ratio = hold_duration / cfg.hold_min_steps
            return cfg.hold_too_short_penalty * (1 - early_ratio)
        
        # Calculate hold bonus
        steps_beyond_min = hold_duration - cfg.hold_min_steps
        hold_bonus = steps_beyond_min * cfg.hold_per_step_bonus
        
        # Cap at maximum
        hold_bonus = min(hold_bonus, cfg.hold_max_reward)
        
        # Bonus for hitting take-profit (shows patience paid off)
        if exit_reason == 'take_profit':
            hold_bonus += cfg.take_profit_bonus
        
        return hold_bonus
    
    def _calc_win_achieved(self, pnl_pct: float) -> float:
        """
        WIN_ACHIEVED: Binary win/loss signal
        
        This teaches: "Winning is good, losing is bad"
        
        Examples:
            +anything → +1.0
            -anything → -0.5
            0 (rare)  → 0.0
        """
        if pnl_pct > 0.0001:  # Small threshold for "win"
            return self.config.win_reward
        elif pnl_pct < -0.0001:
            return self.config.loss_penalty
        else:
            return self.config.breakeven_reward
    
    def _calc_loss_control(
        self, 
        pnl_pct: float, 
        exit_reason: str,
        is_loser: bool
    ) -> float:
        """
        LOSS_CONTROL: For losers, rewards cutting early
        
        This teaches: "If you're going to lose, lose SMALL"
        
        Examples (for 3% stop-loss):
            Winner         → 0.0 (not applicable)
            Cut at -0.5%   → +0.83 (great cut!)
            Cut at -1.5%   → +0.5 (okay)
            Cut at -2.5%   → +0.17 (late)
            Hit stop-loss  → -0.5 (let it go too far)
        """
        if not is_loser:
            return 0.0  # Not applicable to winners
        
        cfg = self.config
        loss_pct = abs(pnl_pct)
        
        if exit_reason == 'stop_loss':
            # Hit stop-loss - bad loss control
            return cfg.loss_control_sl_penalty
        
        # Calculate how much of the stop-loss was avoided
        # If SL is 3% and we exited at -1%, we avoided 2% of loss
        avoided_pct = cfg.stop_loss_pct - loss_pct
        
        if avoided_pct > 0:
            # Reward proportional to how much was avoided
            # At -0% (cut immediately): avoided = 3%, reward = max
            # At -1.5%: avoided = 1.5%, reward = 0.5 * max
            # At -3%: avoided = 0%, reward = 0
            ratio = avoided_pct / cfg.stop_loss_pct
            return cfg.loss_control_max_reward * ratio * cfg.loss_control_scale
        else:
            # Loss exceeds stop-loss (shouldn't happen if SL works)
            return cfg.loss_control_sl_penalty
    
    def _calc_risk_reward(
        self,
        pnl_pct: float,
        hold_duration: int,
        max_favorable_excursion: Optional[float],
        max_adverse_excursion: Optional[float],
    ) -> float:
        """
        RISK_REWARD: Rewards good risk/reward ratios
        
        This teaches: "Structure trades with big wins, small losses"
        
        R:R = |profit| / |max_risk_taken|
        
        Examples:
            +3% with max drawdown -1% → R:R = 3.0 → +1.0 (excellent!)
            +1.5% with max drawdown -1.5% → R:R = 1.0 → +0.5 (good)
            +0.5% with max drawdown -2% → R:R = 0.25 → -0.5 (poor)
        """
        cfg = self.config
        
        # If we don't have excursion data, estimate from P&L
        if max_adverse_excursion is None:
            # Estimate: assume risk was proportional to result
            if pnl_pct > 0:
                # Winner - assume some drawdown occurred
                estimated_risk = min(abs(pnl_pct) * 0.5, cfg.stop_loss_pct)
            else:
                # Loser - risk was the loss itself
                estimated_risk = abs(pnl_pct)
            max_adverse_excursion = estimated_risk
        
        # Avoid division by zero
        risk = max(abs(max_adverse_excursion), 0.001)
        reward_taken = abs(pnl_pct)
        
        # Calculate R:R ratio
        if pnl_pct > 0:
            # For winners: R:R = profit / risk
            rr_ratio = reward_taken / risk
        else:
            # For losers: R:R is negative (inverted)
            rr_ratio = -risk / max(reward_taken, 0.001)
        
        # Score based on R:R
        if rr_ratio >= cfg.rr_excellent_threshold:
            return cfg.rr_excellent_reward
        elif rr_ratio >= cfg.rr_good_threshold:
            # Interpolate between good and excellent
            ratio = (rr_ratio - cfg.rr_good_threshold) / (cfg.rr_excellent_threshold - cfg.rr_good_threshold)
            return cfg.rr_good_reward + ratio * (cfg.rr_excellent_reward - cfg.rr_good_reward)
        elif rr_ratio > 0:
            # Below 1.0 but positive
            ratio = rr_ratio / cfg.rr_good_threshold
            return cfg.rr_poor_penalty + ratio * (cfg.rr_good_reward - cfg.rr_poor_penalty)
        else:
            # Negative R:R (loser)
            return cfg.rr_poor_penalty
    
    def get_total_reward(self, rewards: Dict[str, float]) -> float:
        """
        Combine objectives into single scalar (for comparison/logging)
        
        This is NOT used for training - each objective trains separately.
        Only for logging/analysis.
        """
        weights = self.config.get_weights()
        
        total = sum(
            rewards[obj] * weights[obj]
            for obj in rewards
        )
        
        return total
    
    def get_reward_summary(self, rewards: Dict[str, float]) -> str:
        """Pretty print reward breakdown"""
        lines = [
            "┌─────────────────────────────────────────┐",
            "│     MULTI-OBJECTIVE REWARD BREAKDOWN    │",
            "├─────────────────────────────────────────┤",
        ]
        
        weights = self.config.get_weights()
        
        for obj, value in rewards.items():
            weight = weights[obj]
            weighted = value * weight
            bar = "█" * int(abs(value) * 5)
            sign = "+" if value >= 0 else ""
            lines.append(
                f"│ {obj:15s} │ {sign}{value:6.2f} × {weight:.2f} = {weighted:+6.2f} │"
            )
        
        total = self.get_total_reward(rewards)
        lines.extend([
            "├─────────────────────────────────────────┤",
            f"│ {'TOTAL':15s} │ {total:+6.2f}                  │",
            "└─────────────────────────────────────────┘",
        ])
        
        return "\n".join(lines)


# Convenience function
def calculate_multi_objective_reward(
    pnl_pct: float,
    hold_duration: int,
    exit_reason: str,
    entry_price: float,
    exit_price: float,
    config: Optional[MultiObjectiveConfig] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to calculate multi-objective rewards
    
    Returns dict with 5 reward components.
    """
    calculator = MultiObjectiveRewardCalculator(config)
    return calculator.calculate(
        pnl_pct=pnl_pct,
        hold_duration=hold_duration,
        exit_reason=exit_reason,
        entry_price=entry_price,
        exit_price=exit_price,
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Create calculator with default config
    calc = MultiObjectiveRewardCalculator()
    
    print("=" * 60)
    print("EXAMPLE 1: Good winner held long")
    print("=" * 60)
    rewards = calc.calculate(
        pnl_pct=0.025,          # +2.5% profit
        hold_duration=30,        # Held 30 steps (2.5 hours)
        exit_reason='agent',
        entry_price=100.0,
        exit_price=102.5
    )
    print(calc.get_reward_summary(rewards))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Quick small winner (bad hold)")
    print("=" * 60)
    rewards = calc.calculate(
        pnl_pct=0.005,          # +0.5% profit
        hold_duration=5,         # Only 5 steps (25 min)
        exit_reason='agent',
        entry_price=100.0,
        exit_price=100.5
    )
    print(calc.get_reward_summary(rewards))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Cut loser early (good loss control)")
    print("=" * 60)
    rewards = calc.calculate(
        pnl_pct=-0.008,         # -0.8% loss
        hold_duration=15,        # Held 15 steps
        exit_reason='agent',     # Agent decided to cut
        entry_price=100.0,
        exit_price=99.2
    )
    print(calc.get_reward_summary(rewards))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Hit stop-loss (bad loss control)")
    print("=" * 60)
    rewards = calc.calculate(
        pnl_pct=-0.03,          # -3% loss (stop-loss)
        hold_duration=25,        # Held 25 steps
        exit_reason='stop_loss',
        entry_price=100.0,
        exit_price=97.0
    )
    print(calc.get_reward_summary(rewards))
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Hit take-profit (ideal trade!)")
    print("=" * 60)
    rewards = calc.calculate(
        pnl_pct=0.03,           # +3% profit (take-profit)
        hold_duration=40,        # Held 40 steps
        exit_reason='take_profit',
        entry_price=100.0,
        exit_price=103.0
    )
    print(calc.get_reward_summary(rewards))