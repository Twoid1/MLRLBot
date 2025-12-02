"""
Adaptive Exploration System for Trading RL
==========================================

Implements sophisticated exploration strategies:
1. Extended early exploration (front-loaded learning)
2. Performance-adaptive exploration (responds to learning progress)
3. Regime-aware boosting (market-adaptive)
4. Uncertainty-based adjustments

Author: AI Trading System
Version: 1.0
"""

import numpy as np
import logging
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class HybridExplorationStrategy:
    """
    Ultimate exploration strategy for trading RL
    
    Combines multiple adaptive mechanisms:
    - Extended early exploration (35% of training at high epsilon)
    - Performance-adaptive (increases when stuck, decreases when improving)
    - Regime-aware (boosts for new market conditions)
    - Uncertainty-aware (boosts when Q-values uncertain)
    """
    
    def __init__(self,
                 total_episodes: int = 3500,
                 # Phase 1: Extended early exploration
                 phase1_duration: float = 0.35,
                 phase1_epsilon_floor: float = 0.70,
                 # Phase 2: Performance-adaptive
                 phase2_base: float = 0.30,
                 phase2_min: float = 0.15,
                 phase2_max: float = 0.60,
                 adaptation_rate: float = 0.05,
                 # ✅ NEW: Configurable averaging windows
                 recent_window: int = 10,      # Average last 10 validations
                 comparison_window: int = 10,   # Compare to previous 10
                 min_samples: int = 15,         # Need 15+ before adapting
                 # Regime awareness
                 regime_boost: float = 0.20,
                 # Uncertainty
                 uncertainty_threshold: float = 10.0,
                 uncertainty_boost: float = 0.10):
        """
        Initialize hybrid exploration strategy
        
        Args:
            total_episodes: Total training episodes
            phase1_duration: Fraction of training for phase 1 (0.0-1.0)
            phase1_epsilon_floor: Minimum epsilon during phase 1
            phase2_base: Base epsilon for phase 2
            phase2_min: Minimum epsilon (never below this)
            phase2_max: Maximum epsilon (never above this)
            adaptation_rate: How fast to adapt epsilon
            performance_window: Window size for performance trend
            regime_boost: Epsilon boost for new regimes
            uncertainty_threshold: Q-value range threshold for uncertainty
            uncertainty_boost: Epsilon boost when uncertain
        """
        # Configuration
        self.total_episodes = total_episodes
        self.phase1_episodes = int(total_episodes * phase1_duration)
        self.phase1_epsilon_floor = phase1_epsilon_floor
        
        self.phase2_base = phase2_base
        self.phase2_min = phase2_min
        self.phase2_max = phase2_max
        self.adaptation_rate = adaptation_rate

        self.regime_boost = regime_boost
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_boost = uncertainty_boost

        # ✅ NEW: Moving average configuration
        self.recent_window = recent_window
        self.comparison_window = comparison_window
        self.min_samples = min_samples
        
        # ✅ NEW: Track moving averages explicitly
        self.recent_avg_history = []
        self.comparison_avg_history = []
        
        # State
        self.current_episode = 0
        self.epsilon = 1.0
        self.base_epsilon = 1.0  # Current base before bonuses
        
        # Performance tracking
        self.performance_history = []
        self.epsilon_history = []
        
        # Regime tracking
        self.regimes_seen = set()
        self.current_regime = None
        
        # Statistics
        self.total_regime_boosts = 0
        self.total_uncertainty_boosts = 0
        self.total_adaptations = 0
        
        logger.info(" Hybrid Exploration Strategy initialized")
        logger.info(f"  Phase 1: Episodes 0-{self.phase1_episodes} (floor: {phase1_epsilon_floor:.1%})")
        logger.info(f"  Phase 2: Episodes {self.phase1_episodes}-{total_episodes}")
        logger.info(f"  Adaptive range: {phase2_min:.1%} - {phase2_max:.1%}")
    
    def get_epsilon(self, 
                    episode: int,
                    state: Optional[np.ndarray] = None,
                    q_values: Optional[np.ndarray] = None) -> float:
        """
        Get epsilon for current step
        
        Args:
            episode: Current episode number
            state: Current state (for regime detection)
            q_values: Current Q-values (for uncertainty detection)
            
        Returns:
            Epsilon value (0.0-1.0)
        """
        self.current_episode = episode
        
        # ===== PHASE 1: Extended Early Exploration =====
        if episode < self.phase1_episodes:
            # Linear decay from 1.0 to phase1_epsilon_floor
            progress = episode / self.phase1_episodes
            self.base_epsilon = 1.0 - progress * (1.0 - self.phase1_epsilon_floor)
            phase = "PHASE_1"
            
        # ===== PHASE 2: Performance-Adaptive =====
        else:
            # Use adaptive epsilon (updated by performance)
            self.base_epsilon = self.epsilon
            phase = "PHASE_2"
        
        # Start with base
        final_epsilon = self.base_epsilon
        
        # ===== REGIME BOOST =====
        regime_bonus = 0.0
        if state is not None:
            regime = self._detect_regime(state)
            
            if regime != self.current_regime:
                self.current_regime = regime
                
                if regime not in self.regimes_seen:
                    regime_bonus = self.regime_boost
                    self.regimes_seen.add(regime)
                    self.total_regime_boosts += 1
                    
                    logger.info(f" New regime '{regime}' detected at episode {episode}! "
                              f"Boosting epsilon by +{regime_bonus:.0%}")
        
        # ===== UNCERTAINTY BOOST =====
        uncertainty_bonus = 0.0
        if q_values is not None:
            q_range = float(np.max(q_values) - np.min(q_values))
            
            if q_range < self.uncertainty_threshold:
                uncertainty_bonus = self.uncertainty_boost
                self.total_uncertainty_boosts += 1
        
        # ===== COMBINE =====
        final_epsilon = self.base_epsilon + regime_bonus + uncertainty_bonus
        
        # Clip to valid range
        final_epsilon = np.clip(final_epsilon, self.phase2_min, 0.95)
        
        # Log periodically
        if episode % 100 == 0 and episode > 0:
            logger.info(f"\n{'='*70}")
            logger.info(f" Exploration Update - Episode {episode}")
            logger.info(f"{'='*70}")
            logger.info(f"  Phase:              {phase}")
            logger.info(f"  Base epsilon:       {self.base_epsilon:.3f}")
            logger.info(f"  Regime bonus:       +{regime_bonus:.3f}")
            logger.info(f"  Uncertainty bonus:  +{uncertainty_bonus:.3f}")
            logger.info(f"  Final epsilon:      {final_epsilon:.3f}")
            logger.info(f"  Regimes discovered: {len(self.regimes_seen)}")
            logger.info(f"{'='*70}\n")
        
        # Save history
        self.epsilon_history.append(final_epsilon)
        
        return final_epsilon
    
    def update_from_performance(self, validation_reward: float) -> None:
        """
        Update epsilon based on MOVING AVERAGES of performance
        
        ✅ NEW: Uses rolling averages for stable adaptation
        - Compares recent average (last N validations)
        - To previous average (N validations before that)
        - Only adapts when trend is clear
        
        Args:
            validation_reward: Latest validation reward
        """
        self.performance_history.append(validation_reward)
        
        # Only adapt in Phase 2
        if self.current_episode < self.phase1_episodes:
            logger.debug(f"Phase 1: Not adapting yet (episode {self.current_episode})")
            return
        
        # Need enough history
        total_needed = self.recent_window + self.comparison_window
        if len(self.performance_history) < self.min_samples:
            logger.debug(f"Not enough samples: {len(self.performance_history)}/{self.min_samples}")
            return
        
        # ===== CALCULATE MOVING AVERAGES =====
        
        # Recent average (last N validations)
        recent_rewards = self.performance_history[-self.recent_window:]
        recent_avg = np.mean(recent_rewards)
        recent_std = np.std(recent_rewards)
        
        # Comparison average (N validations before recent)
        comparison_start = -(self.recent_window + self.comparison_window)
        comparison_end = -self.recent_window
        comparison_rewards = self.performance_history[comparison_start:comparison_end]
        comparison_avg = np.mean(comparison_rewards)
        comparison_std = np.std(comparison_rewards)
        
        # Save for tracking
        self.recent_avg_history.append(recent_avg)
        self.comparison_avg_history.append(comparison_avg)
        
        # ===== CALCULATE IMPROVEMENT IN AVERAGES =====
        
        avg_change = recent_avg - comparison_avg
        
        # Calculate improvement rate (percentage change)
        if abs(comparison_avg) > 0.01:
            improvement_rate = avg_change / abs(comparison_avg)
        else:
            improvement_rate = 0.0
        
        # ===== CHECK STATISTICAL SIGNIFICANCE (optional but good) =====
        
        # Simple check: is the change larger than combined noise?
        combined_noise = (recent_std + comparison_std) / 2
        signal_to_noise = abs(avg_change) / (combined_noise + 1e-6)
        
        # Only trust strong signals
        is_significant = signal_to_noise > 0.5  # Change is larger than half the noise
        
        # ===== DETERMINE EPSILON ADJUSTMENT =====
        
        adjustment = 0.0
        reason = ""
        confidence = "high" if is_significant else "low"
        
        if not is_significant:
            # Change too noisy - don't adapt
            adjustment = 0.0
            reason = " Change too noisy (low signal-to-noise)"
            
        elif improvement_rate > 0.15:
            # Strong improvement in average: Reduce exploration
            adjustment = -self.adaptation_rate * 1.5
            reason = " STRONG average improvement  Exploit learned strategy"
            
        elif improvement_rate > 0.05:
            # Moderate improvement: Slight reduction
            adjustment = -self.adaptation_rate * 0.8
            reason = " Moderate average improvement  Slight reduction"
            
        elif improvement_rate < -0.10:
            # Average degrading significantly: BOOST exploration!
            adjustment = self.adaptation_rate * 2.5
            reason = " AVERAGE DEGRADING  INCREASE EXPLORATION!"
            
        elif improvement_rate < -0.03:
            # Slight degradation: Increase exploration
            adjustment = self.adaptation_rate * 1.5
            reason = " Slight average degradation  Increase exploration"
            
        elif abs(improvement_rate) < 0.02:
            # Average plateaued: Increase to break out
            adjustment = self.adaptation_rate * 1.2
            reason = " Average plateaued  Increase to find new strategies"
            
        else:
            # Slight improvement: Maintain
            adjustment = 0.0
            reason = " Slight average improvement  Maintain current level"
        
        # ===== APPLY ADJUSTMENT =====
        
        old_epsilon = self.epsilon
        self.epsilon = np.clip(
            self.epsilon + adjustment,
            self.phase2_min,
            self.phase2_max
        )
        
        self.total_adaptations += 1
        
        # ===== DETAILED LOGGING =====
        
        logger.info(f"\n{'='*70}")
        logger.info(f" Performance-Based Adaptation #{self.total_adaptations}")
        logger.info(f"{'='*70}")
        logger.info(f"  MOVING AVERAGES:")
        logger.info(f"    Recent avg ({self.recent_window} vals):     {recent_avg:.2f} (+-{recent_std:.2f})")
        logger.info(f"    Previous avg ({self.comparison_window} vals): {comparison_avg:.2f} (+-{comparison_std:.2f})")
        logger.info(f"    Change in average:       {avg_change:+.2f}")
        logger.info(f"    Improvement rate:        {improvement_rate:+.1%}")
        logger.info(f"  SIGNAL QUALITY:")
        logger.info(f"    Signal-to-noise:         {signal_to_noise:.2f}")
        logger.info(f"    Confidence:              {confidence}")
        logger.info(f"  DECISION:")
        logger.info(f"    {reason}")
        logger.info(f"    Epsilon change:          {old_epsilon:.3f}  {self.epsilon:.3f} ({adjustment:+.3f})")
        logger.info(f"{'='*70}\n")
        
        # ===== ALERT IF MAJOR CHANGE =====
        
        if abs(adjustment) > 0.1:
            logger.warning(f"  LARGE EPSILON CHANGE: {adjustment:+.3f}")
            logger.warning(f"   This is a significant adaptation based on average performance trend")
        
        if improvement_rate < -0.10:
            logger.warning(f" PERFORMANCE ALERT: Average degraded by {abs(improvement_rate):.1%}")
            logger.warning(f"   Boosting exploration to recover")

    def get_performance_summary(self) -> Dict:
        """
        Get detailed performance summary with moving averages
        """
        if len(self.performance_history) < self.min_samples:
            return {
                'status': 'insufficient_data',
                'samples': len(self.performance_history),
                'needed': self.min_samples
            }
        
        # Calculate current moving averages
        recent_avg = np.mean(self.performance_history[-self.recent_window:])
        recent_std = np.std(self.performance_history[-self.recent_window:])
        
        if len(self.performance_history) >= self.recent_window + self.comparison_window:
            comparison_start = -(self.recent_window + self.comparison_window)
            comparison_end = -self.recent_window
            comparison_avg = np.mean(self.performance_history[comparison_start:comparison_end])
            
            improvement = recent_avg - comparison_avg
            improvement_rate = improvement / abs(comparison_avg) if abs(comparison_avg) > 0.01 else 0
        else:
            comparison_avg = None
            improvement = None
            improvement_rate = None
        
        # Overall trend
        if len(self.performance_history) >= 20:
            early_avg = np.mean(self.performance_history[:10])
            late_avg = np.mean(self.performance_history[-10:])
            total_improvement = late_avg - early_avg
            total_improvement_rate = total_improvement / abs(early_avg) if abs(early_avg) > 0.01 else 0
        else:
            early_avg = None
            late_avg = None
            total_improvement = None
            total_improvement_rate = None
        
        return {
            'status': 'active',
            'current_avg': recent_avg,
            'current_std': recent_std,
            'previous_avg': comparison_avg,
            'recent_improvement': improvement,
            'recent_improvement_rate': improvement_rate,
            'overall_start_avg': early_avg,
            'overall_current_avg': late_avg,
            'overall_improvement': total_improvement,
            'overall_improvement_rate': total_improvement_rate,
        }
    
    def _detect_regime(self, state: np.ndarray) -> str:
        """
        Detect market regime from state
        
        Args:
            state: Current state vector
            
        Returns:
            Regime name (string)
        """
        # Extract relevant features
        # Adjust indices based on your state structure
        try:
            # Assuming first few features are market indicators
            momentum = float(state[0]) if len(state) > 0 else 0.0
            volatility = float(state[1]) if len(state) > 1 else 0.01
            
            # Classify regime
            if momentum > 0.03:
                return 'bull_strong' if volatility < 0.03 else 'bull_volatile'
            elif momentum > 0.01:
                return 'bull_weak' if volatility < 0.03 else 'bull_choppy'
            elif momentum < -0.03:
                return 'bear_strong' if volatility < 0.03 else 'bear_volatile'
            elif momentum < -0.01:
                return 'bear_weak' if volatility < 0.03 else 'bear_choppy'
            elif volatility > 0.05:
                return 'sideways_volatile'
            elif volatility < 0.015:
                return 'sideways_calm'
            else:
                return 'sideways_normal'
                
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}, using 'unknown'")
            return 'unknown'
    
    def get_stats(self) -> Dict:
        """Get exploration statistics"""
        return {
            'current_episode': self.current_episode,
            'current_phase': 1 if self.current_episode < self.phase1_episodes else 2,
            'base_epsilon': self.base_epsilon,
            'regimes_discovered': len(self.regimes_seen),
            'total_regime_boosts': self.total_regime_boosts,
            'total_uncertainty_boosts': self.total_uncertainty_boosts,
            'total_adaptations': self.total_adaptations,
            'performance_trend': self._get_performance_trend(),
        }
    
    def _get_performance_trend(self) -> str:
        """Calculate recent performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = np.mean(self.performance_history[-5:])
        previous = np.mean(self.performance_history[-10:-5])
        
        if abs(previous) < 0.01:
            return "unknown"
        
        improvement = (recent - previous) / abs(previous)
        
        if improvement > 0.10:
            return "strong_improvement"
        elif improvement > 0.03:
            return "improving"
        elif improvement < -0.05:
            return "degrading"
        else:
            return "stable"
    
    def save_state(self) -> Dict:
        """Save exploration state for checkpointing"""
        return {
            'current_episode': self.current_episode,
            'epsilon': self.epsilon,
            'base_epsilon': self.base_epsilon,
            'performance_history': self.performance_history,
            'epsilon_history': self.epsilon_history,
            'regimes_seen': list(self.regimes_seen),
            'current_regime': self.current_regime,
            'total_regime_boosts': self.total_regime_boosts,
            'total_uncertainty_boosts': self.total_uncertainty_boosts,
            'total_adaptations': self.total_adaptations,
        }
    
    def load_state(self, state: Dict) -> None:
        """Load exploration state from checkpoint"""
        self.current_episode = state['current_episode']
        self.epsilon = state['epsilon']
        self.base_epsilon = state['base_epsilon']
        self.performance_history = state['performance_history']
        self.epsilon_history = state['epsilon_history']
        self.regimes_seen = set(state['regimes_seen'])
        self.current_regime = state['current_regime']
        self.total_regime_boosts = state['total_regime_boosts']
        self.total_uncertainty_boosts = state['total_uncertainty_boosts']
        self.total_adaptations = state['total_adaptations']
        
        logger.info(f" Loaded exploration state from episode {self.current_episode}")


class ExplorationMonitor:
    """
    Monitor and visualize exploration strategy with moving averages
    """
    
    def __init__(self, exploration_strategy: HybridExplorationStrategy):
        self.exploration = exploration_strategy
        
        self.episode_log = []
        self.epsilon_log = []
        self.reward_log = []
        self.regime_log = []
        
        # ✅ NEW: Track moving averages
        self.recent_avg_log = []
        self.comparison_avg_log = []
    
    def log(self, episode: int, epsilon: float, reward: float, regime: str = None):
        """Log exploration data with moving averages"""
        self.episode_log.append(episode)
        self.epsilon_log.append(epsilon)
        self.reward_log.append(reward)
        self.regime_log.append(regime)
        
        # ✅ NEW: Log moving averages
        if len(self.exploration.recent_avg_history) > 0:
            self.recent_avg_log.append(self.exploration.recent_avg_history[-1])
        
        if len(self.exploration.comparison_avg_history) > 0:
            self.comparison_avg_log.append(self.exploration.comparison_avg_history[-1])
    
    def plot(self, save_path: str = "exploration_analysis.png"):
        """Plot exploration analysis with moving averages"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 16))
            
            # ===== Plot 1: Epsilon over time =====
            ax1 = axes[0]
            ax1.plot(self.episode_log, self.epsilon_log, 
                    linewidth=2, color='orange', label='Epsilon')
            
            # Mark phase transition
            if len(self.episode_log) > 0:
                phase1_end = self.exploration.phase1_episodes
                ax1.axvline(phase1_end, color='red', linestyle='--', 
                           label=f'Phase Transition (ep {phase1_end})', alpha=0.7)
                ax1.axhline(self.exploration.phase2_min, color='green', 
                           linestyle=':', alpha=0.5, label=f'Min ({self.exploration.phase2_min:.0%})')
                ax1.axhline(self.exploration.phase2_max, color='red', 
                           linestyle=':', alpha=0.5, label=f'Max ({self.exploration.phase2_max:.0%})')
            
            ax1.set_ylabel('Epsilon', fontsize=12)
            ax1.set_title('Adaptive Exploration Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            ax1.set_ylim(0, 1.0)
            
            # ===== Plot 2: Raw rewards + Moving Averages =====
            ax2 = axes[1]
            
            # Raw rewards (light)
            ax2.plot(self.episode_log, self.reward_log, 
                    linewidth=1, color='lightblue', label='Raw Validation Reward', alpha=0.5)
            
            # Moving averages (bold)
            if len(self.recent_avg_log) > 0:
                # Only plot where we have data
                avg_episodes = self.episode_log[-len(self.recent_avg_log):]
                
                ax2.plot(avg_episodes, self.recent_avg_log, 
                        linewidth=3, color='blue', label=f'Recent Avg ({self.exploration.recent_window})', alpha=0.8)
                
                if len(self.comparison_avg_log) > 0:
                    comp_episodes = self.episode_log[-len(self.comparison_avg_log):]
                    ax2.plot(comp_episodes, self.comparison_avg_log, 
                            linewidth=3, color='navy', linestyle='--', 
                            label=f'Previous Avg ({self.exploration.comparison_window})', alpha=0.8)
            
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('Performance: Raw Rewards + Moving Averages', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # ===== Plot 3: Epsilon vs Moving Average (overlay) =====
            ax3 = axes[2]
            ax3_twin = ax3.twinx()
            
            # Recent moving average
            if len(self.recent_avg_log) > 0:
                avg_episodes = self.episode_log[-len(self.recent_avg_log):]
                ax3.plot(avg_episodes, self.recent_avg_log, 
                        label='Recent Avg Reward', color='blue', linewidth=3, alpha=0.8)
            
            # Epsilon
            ax3_twin.plot(self.episode_log, self.epsilon_log,
                         label='Epsilon', color='orange', linewidth=2, alpha=0.7)
            
            ax3.set_ylabel('Moving Avg Reward', color='blue', fontsize=12)
            ax3_twin.set_ylabel('Epsilon', color='orange', fontsize=12)
            ax3.set_title('Moving Average Performance vs Exploration', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
            
            # ===== Plot 4: Improvement Rate =====
            ax4 = axes[3]
            
            if len(self.recent_avg_log) > 0 and len(self.comparison_avg_log) > 0:
                # Calculate improvement rate
                improvement_rates = []
                improvement_episodes = []
                
                for i in range(min(len(self.recent_avg_log), len(self.comparison_avg_log))):
                    recent = self.recent_avg_log[i]
                    comparison = self.comparison_avg_log[i]
                    
                    if abs(comparison) > 0.01:
                        rate = (recent - comparison) / abs(comparison)
                        improvement_rates.append(rate * 100)  # Percentage
                        improvement_episodes.append(self.episode_log[-(len(self.recent_avg_log)-i)])
                
                # Plot improvement rate
                colors = ['green' if r > 0 else 'red' for r in improvement_rates]
                ax4.bar(improvement_episodes, improvement_rates, 
                       color=colors, alpha=0.6, width=20)
                
                # Zero line
                ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                
                ax4.set_xlabel('Episode', fontsize=12)
                ax4.set_ylabel('Improvement Rate (%)', fontsize=12)
                ax4.set_title('Moving Average Improvement Rate', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f" Exploration analysis saved to {save_path}")
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
        except Exception as e:
            logger.error(f"Failed to create plot: {e}")
    
    def print_summary(self):
        """Print exploration summary with moving averages"""
        if len(self.epsilon_log) == 0:
            logger.info("No exploration data logged yet")
            return
        
        stats = self.exploration.get_stats()
        perf_summary = self.exploration.get_performance_summary()
        
        print(f"\n{'='*70}")
        print(" EXPLORATION SUMMARY (Moving Average Based)")
        print(f"{'='*70}")
        print(f"  Current episode:        {stats['current_episode']}")
        print(f"  Current phase:          {stats['current_phase']}")
        print(f"  Current epsilon:        {stats['base_epsilon']:.3f}")
        print(f"  Average epsilon:        {np.mean(self.epsilon_log):.3f}")
        print(f"  Min/Max epsilon:        {np.min(self.epsilon_log):.3f} / {np.max(self.epsilon_log):.3f}")
        
        print(f"\n  MOVING AVERAGE PERFORMANCE:")
        if perf_summary['status'] == 'active':
            print(f"    Current avg:          {perf_summary['current_avg']:.2f} (+-{perf_summary['current_std']:.2f})")
            
            if perf_summary['previous_avg'] is not None:
                print(f"    Previous avg:         {perf_summary['previous_avg']:.2f}")
                print(f"    Recent change:        {perf_summary['recent_improvement']:+.2f} ({perf_summary['recent_improvement_rate']:+.1%})")
            
            if perf_summary['overall_improvement'] is not None:
                print(f"    Overall improvement:  {perf_summary['overall_improvement']:+.2f} ({perf_summary['overall_improvement_rate']:+.1%})")
                print(f"      From: {perf_summary['overall_start_avg']:.2f}")
                print(f"      To:   {perf_summary['overall_current_avg']:.2f}")
        else:
            print(f"    Status: {perf_summary['status']}")
            print(f"    Samples: {perf_summary['samples']}/{perf_summary['needed']}")
        
        print(f"\n  ADAPTATION STATISTICS:")
        print(f"    Regimes discovered:     {stats['regimes_discovered']}")
        print(f"    Regime boosts:          {stats['total_regime_boosts']}")
        print(f"    Uncertainty boosts:     {stats['total_uncertainty_boosts']}")
        print(f"    Performance adaptations: {stats['total_adaptations']}")
        print(f"    Trend:                  {stats['performance_trend']}")
        
        print(f"{'='*70}\n")