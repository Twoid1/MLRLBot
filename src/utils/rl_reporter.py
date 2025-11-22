"""
Enhanced RL Training Report Generator with Trade Details
Generates detailed analysis of RL agent performance including individual trade information
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import json


class RLTrainingReporter:
    """Generate comprehensive training reports for RL agent with trade details"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_full_report(self, 
                            episode_results: List[Dict],
                            env,
                            agent,
                            config: Dict,
                            save_path: str = None) -> str:
        """
        Generate complete training report
        
        Args:
            episode_results: List of episode statistics
            env: Trading environment (for final evaluation)
            agent: Trained RL agent
            config: Training configuration
            save_path: Where to save the report
            
        Returns:
            Formatted report string
        """
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 100)
        report_lines.append("RL TRAINING COMPLETE - COMPREHENSIVE PERFORMANCE REPORT")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 1. Training Configuration
        report_lines.extend(self._section_training_config(config))
        
        # 2. Training Progress Summary (NOW WITH VALIDATION)
        report_lines.extend(self._section_training_progress(episode_results))
        
        # 2.5 Validation Summary (NEW!)
        report_lines.extend(self._section_validation_summary(episode_results))
        
        # 3. Episode Performance Analysis
        report_lines.extend(self._section_episode_analysis(episode_results))
        
        # 4. Trading Behavior Analysis with Trade Details
        report_lines.extend(self._section_trading_behavior(episode_results))
        
        # 5. Learning Progress
        report_lines.extend(self._section_learning_progress(episode_results))
        
        # 6. Final Evaluation
        report_lines.extend(self._section_final_evaluation(env, agent))
        
        # 7. Best Episodes Breakdown with Top Trades
        report_lines.extend(self._section_best_episodes(episode_results))
        
        # 8. Top Profitable Trades Details (NEW!)
        report_lines.extend(self._section_top_trades(episode_results))
        
        # 9. Recommendations
        report_lines.extend(self._section_recommendations(episode_results, agent))
        
        # Combine report
        report = "\n".join(report_lines)
        
        # Save to file
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\n Report saved to: {save_path}")
        
        return report
    
    def _section_training_config(self, config: Dict) -> List[str]:
        """Training configuration section"""
        lines = []
        lines.append("=" * 100)
        lines.append("1. TRAINING CONFIGURATION")
        lines.append("=" * 100)
        lines.append(f"Episodes Trained:          {config.get('rl_episodes', 'N/A')}")
        lines.append(f"Steps per Episode:         {config.get('max_steps_per_episode', 'N/A')}")
        lines.append(f"Initial Balance:           ${config.get('initial_balance', 10000):,.2f}")
        lines.append(f"Fee Rate:                  {config.get('fee_rate', 0.0026)*100:.2f}%")
        lines.append(f"Network Architecture:      {config.get('rl_hidden_dims', [256, 256, 128])}")
        lines.append(f"Batch Size:                {config.get('rl_batch_size', 256)}")
        lines.append(f"Learning Rate:             {config.get('rl_learning_rate', 0.0001)}")
        lines.append(f"Gamma (Discount):          {config.get('rl_gamma', 0.99)}")
        lines.append(f"Double DQN:                {config.get('use_double_dqn', True)}")
        lines.append(f"Dueling DQN:               {config.get('use_dueling_dqn', True)}")
        lines.append("")
        return lines
    
    def _section_training_progress(self, episode_results: List[Dict]) -> List[str]:
        """
        Training progress summary with VALIDATION
        
        UPDATED: Now detects inconsistencies between rewards and portfolio values
        """
        lines = []
        lines.append("=" * 100)
        lines.append("2. TRAINING PROGRESS")
        lines.append("=" * 100)
        
        n = len(episode_results)
        
        # Overall statistics
        total_rewards = [e['total_reward'] for e in episode_results]
        portfolio_values = [e.get('portfolio_value', 0) for e in episode_results]
        
        lines.append("OVERALL PERFORMANCE:")
        lines.append(f"  Total Episodes:          {n}")
        lines.append(f"  Avg Reward:              {np.mean(total_rewards):.2f}")
        lines.append(f"  Best Reward:             {max(total_rewards):.2f} (Episode {np.argmax(total_rewards) + 1})")
        lines.append(f"  Worst Reward:            {min(total_rewards):.2f} (Episode {np.argmin(total_rewards) + 1})")
        lines.append(f"  Reward Std Dev:          {np.std(total_rewards):.2f}")
        lines.append(f"  Avg Final Portfolio:     ${np.mean(portfolio_values):,.2f}")
        lines.append(f"  Best Final Portfolio:    ${max(portfolio_values):,.2f}")
        lines.append("")
        
        # ========================================================================
        # NEW: VALIDATION - Check for inconsistencies
        # ========================================================================
        avg_reward = np.mean(total_rewards)
        avg_portfolio = np.mean(portfolio_values)
        
        # Get initial balance from config (default 10000)
        initial_balance = 10000  # You can pass this from config if available
        
        # Calculate expected relationship
        # If avg portfolio grew 20% (from $10k to $12k), avg reward should be ~+20
        # If scaled by 100, reward should be ~+20.0
        expected_avg_reward = ((avg_portfolio - initial_balance) / initial_balance) * 100
        
        # Check for major inconsistency (reward negative but portfolio positive)
        if avg_reward < -10 and avg_portfolio > initial_balance * 1.05:
            lines.append("=" * 100)
            lines.append("  CRITICAL WARNING: INCONSISTENCY DETECTED!")
            lines.append("=" * 100)
            lines.append(f"  Average Reward:          {avg_reward:.2f} (NEGATIVE)")
            lines.append(f"  Average Portfolio:       ${avg_portfolio:,.2f} (UP {((avg_portfolio/initial_balance - 1)*100):.1f}%)")
            lines.append(f"  Initial Balance:         ${initial_balance:,.2f}")
            lines.append("")
            lines.append("   DIAGNOSIS:")
            lines.append("     Rewards are negative but portfolios grew!")
            lines.append("     This indicates a bug in the reward calculation.")
            lines.append("")
            lines.append("   LIKELY CAUSES:")
            lines.append("     1. Reward function doesn't match portfolio growth")
            lines.append("     2. Step rewards don't accumulate correctly")
            lines.append("     3. Fee double-counting in reward calculation")
            lines.append("")
            lines.append("   RECOMMENDED FIX:")
            lines.append("     Apply Fix #1 from the verification report")
            lines.append("     (Simplify reward calculation)")
            lines.append("=" * 100)
            lines.append("")
        
        # Check for moderate inconsistency (rewards don't match portfolio change)
        elif abs(avg_reward - expected_avg_reward) > 50:
            lines.append("=" * 100)
            lines.append("  WARNING: Reward-Portfolio Mismatch")
            lines.append("=" * 100)
            lines.append(f"  Average Reward:          {avg_reward:.2f}")
            lines.append(f"  Expected Reward:         {expected_avg_reward:.2f}")
            lines.append(f"  Difference:              {abs(avg_reward - expected_avg_reward):.2f}")
            lines.append("")
            lines.append("   INTERPRETATION:")
            if avg_reward < expected_avg_reward:
                lines.append("     Rewards are LOWER than portfolio growth suggests")
                lines.append("     Agent may be penalized too heavily")
            else:
                lines.append("     Rewards are HIGHER than portfolio growth suggests")
                lines.append("     Check if fees are properly accounted for")
            lines.append("=" * 100)
            lines.append("")
        
        # Check for unrealistic portfolio growth
        max_portfolio = max(portfolio_values)
        if max_portfolio > initial_balance * 100:
            lines.append("=" * 100)
            lines.append("  WARNING: Unrealistic Portfolio Growth")
            lines.append("=" * 100)
            lines.append(f"  Best Portfolio:          ${max_portfolio:,.2f}")
            lines.append(f"  Initial Balance:         ${initial_balance:,.2f}")
            lines.append(f"  Growth:                  {(max_portfolio/initial_balance):.0f}x")
            lines.append("")
            lines.append("   DIAGNOSIS:")
            lines.append("     100x+ growth in training is unrealistic!")
            lines.append("     This indicates a bug in position sizing.")
            lines.append("")
            lines.append("   LIKELY CAUSES:")
            lines.append("     1. Position sizes not properly limited")
            lines.append("     2. Leverage bug allowing massive positions")
            lines.append("     3. Fee calculation error creating artificial profits")
            lines.append("")
            lines.append("   RECOMMENDED FIX:")
            lines.append("     Apply Fix #3 from the verification report")
            lines.append("     (Add position size validation)")
            lines.append("=" * 100)
            lines.append("")
        
        # All good - show positive confirmation
        else:
            lines.append(" CONSISTENCY CHECK:")
            lines.append(f"  Avg Reward:              {avg_reward:.2f}")
            lines.append(f"  Expected from Portfolio: {expected_avg_reward:.2f}")
            lines.append(f"  Difference:              {abs(avg_reward - expected_avg_reward):.2f}")
            
            if abs(avg_reward - expected_avg_reward) < 10:
                lines.append("  Status:                   EXCELLENT - Rewards match portfolio growth!")
            elif abs(avg_reward - expected_avg_reward) < 25:
                lines.append("  Status:                   GOOD - Minor variance is acceptable")
            else:
                lines.append("  Status:                    MODERATE - Some inconsistency present")
            lines.append("")
        
        return lines
    
    def _section_validation_summary(self, episode_results: List[Dict]) -> List[str]:
        """
        NEW SECTION: Comprehensive validation of training results
        
        Add this as a new section in your generate_full_report() method
        """
        lines = []
        lines.append("=" * 100)
        lines.append("2.5 VALIDATION SUMMARY")
        lines.append("=" * 100)
        
        # Collect all validation checks
        checks = {
            'reward_consistency': True,
            'portfolio_realistic': True,
            'position_sizes': True,
            'fee_calculation': True,
            'trade_counts': True
        }
        
        issues = []
        warnings = []
        
        # Get statistics
        total_rewards = [e['total_reward'] for e in episode_results]
        portfolio_values = [e.get('portfolio_value', 10000) for e in episode_results]
        initial_balance = 10000
        
        avg_reward = np.mean(total_rewards)
        avg_portfolio = np.mean(portfolio_values)
        max_portfolio = max(portfolio_values)
        
        # Check 1: Reward consistency
        expected_reward = ((avg_portfolio - initial_balance) / initial_balance) * 100
        if avg_reward < -10 and avg_portfolio > initial_balance * 1.05:
            checks['reward_consistency'] = False
            issues.append("Negative rewards despite positive portfolio growth")
        elif abs(avg_reward - expected_reward) > 50:
            warnings.append(f"Rewards don't match portfolio (off by {abs(avg_reward - expected_reward):.1f})")
        
        # Check 2: Portfolio realism
        if max_portfolio > initial_balance * 100:
            checks['portfolio_realistic'] = False
            issues.append(f"Unrealistic growth: {max_portfolio/initial_balance:.0f}x in training")
        
        # Check 3: Trade analysis
        all_trades = []
        for ep in episode_results:
            if 'trades' in ep:
                all_trades.extend([t for t in ep['trades'] if t.get('pnl') is not None])
        
        if all_trades:
            unrealistic_trades = [t for t in all_trades 
                                if t.get('pnl') is not None and t['pnl'] > initial_balance * 2]
            if unrealistic_trades:
                checks['position_sizes'] = False
                issues.append(f"{len(unrealistic_trades)} trades with unrealistic profits")
            
            # Check fee reasonableness
            total_pnl = sum(t['pnl'] for t in all_trades)
            total_fees = sum(t.get('fees', 0) for t in all_trades)
            if total_fees > abs(total_pnl) * 0.5:
                checks['fee_calculation'] = False
                issues.append(f"Fees too high: ${total_fees:,.0f} vs PnL ${total_pnl:,.0f}")
        
        # Display results
        lines.append("VALIDATION CHECKS:")
        for check_name, passed in checks.items():
            status = " PASS" if passed else " FAIL"
            check_label = check_name.replace('_', ' ').title()
            lines.append(f"  {check_label:.<40} {status}")
        lines.append("")
        
        if issues:
            lines.append(" CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")
            lines.append("")
        
        if warnings:
            lines.append("  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                lines.append(f"  {i}. {warning}")
            lines.append("")
        
        if not issues and not warnings:
            lines.append(" ALL VALIDATION CHECKS PASSED!")
            lines.append("   Training results appear consistent and realistic.")
            lines.append("")
        
        return lines
    
    def _section_episode_analysis(self, episode_results: List[Dict]) -> List[str]:
        """Episode performance analysis"""
        lines = []
        lines.append("=" * 100)
        lines.append("3. EPISODE PERFORMANCE ANALYSIS")
        lines.append("=" * 100)
        
        n = len(episode_results)
        
        # Split into phases
        first_25 = episode_results[:n//4]
        second_25 = episode_results[n//4:n//2]
        third_25 = episode_results[n//2:3*n//4]
        final_25 = episode_results[3*n//4:]
        
        def phase_stats(episodes, phase_name):
            rewards = [e['total_reward'] for e in episodes]
            portfolio = [e.get('portfolio_value', 0) for e in episodes]
            return [
                f"{phase_name}:",
                f"  Episodes:                {len(episodes)}",
                f"  Avg Reward:              {np.mean(rewards):.2f}",
                f"  Avg Portfolio:           ${np.mean(portfolio):,.2f}",
                f"  Best Reward:             {max(rewards):.2f}",
                f"  Worst Reward:            {min(rewards):.2f}",
                ""
            ]
        
        lines.extend(phase_stats(first_25, "Phase 1 (Early Learning)"))
        lines.extend(phase_stats(second_25, "Phase 2 (Exploration)"))
        lines.extend(phase_stats(third_25, "Phase 3 (Refinement)"))
        lines.extend(phase_stats(final_25, "Phase 4 (Convergence)"))
        
        # Learning trend
        first_10_avg = np.mean([e['total_reward'] for e in episode_results[:10]])
        last_10_avg = np.mean([e['total_reward'] for e in episode_results[-10:]])
        improvement = ((last_10_avg - first_10_avg) / abs(first_10_avg)) * 100 if first_10_avg != 0 else 0
        
        lines.append("LEARNING TREND:")
        lines.append(f"  First 10 Episodes Avg:   {first_10_avg:.2f}")
        lines.append(f"  Last 10 Episodes Avg:    {last_10_avg:.2f}")
        lines.append(f"  Improvement:             {improvement:+.1f}%")
        
        if improvement > 20:
            lines.append(f"  Status:                    Strong improvement - agent is learning well!")
        elif improvement > 0:
            lines.append(f"  Status:                    Positive improvement - agent is learning")
        else:
            lines.append(f"  Status:                    Limited improvement - may need more training")
        lines.append("")
        
        return lines
    
    def _section_trading_behavior(self, episode_results: List[Dict]) -> List[str]:
        """Analyze trading behavior"""
        lines = []
        lines.append("=" * 100)
        lines.append("4. TRADING BEHAVIOR ANALYSIS")
        lines.append("=" * 100)
        
        # Aggregate trading statistics across all episodes
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        total_loss = 0
        
        for episode in episode_results:
            # Get trading info from episode
            if 'num_trades' in episode:
                total_trades += episode.get('num_trades', 0)
            if 'winning_trades' in episode:
                winning_trades += episode.get('winning_trades', 0)
            if 'losing_trades' in episode:
                losing_trades += episode.get('losing_trades', 0)
        
        # Calculate statistics
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
        else:
            win_rate = 0
        
        lines.append("TRADING STATISTICS (Across All Episodes):")
        lines.append(f"  Total Trades:            {total_trades}")
        lines.append(f"  Winning Trades:          {winning_trades}")
        lines.append(f"  Losing Trades:           {losing_trades}")
        lines.append(f"  Win Rate:                {win_rate:.1f}%")
        lines.append(f"  Avg Trades per Episode:  {total_trades/len(episode_results):.1f}")
        lines.append("")
        
        # Profit factor
        lines.append("RISK/REWARD METRICS:")
        profitable_episodes = [e for e in episode_results if e['total_reward'] > 0]
        losing_episodes = [e for e in episode_results if e['total_reward'] <= 0]
        
        lines.append(f"  Profitable Episodes:     {len(profitable_episodes)} ({len(profitable_episodes)/len(episode_results)*100:.1f}%)")
        lines.append(f"  Losing Episodes:         {len(losing_episodes)} ({len(losing_episodes)/len(episode_results)*100:.1f}%)")
        
        if profitable_episodes:
            avg_profit = np.mean([e['total_reward'] for e in profitable_episodes])
            lines.append(f"  Avg Profit (when win):   {avg_profit:.2f}")
        
        if losing_episodes:
            avg_loss = np.mean([e['total_reward'] for e in losing_episodes])
            lines.append(f"  Avg Loss (when lose):    {avg_loss:.2f}")

        lines.append("TRADE DURATION ANALYSIS:")
        
        # Collect all completed trades with duration info
        all_durations = []
        winning_durations = []
        losing_durations = []
        
        for episode in episode_results:
            if 'trades' in episode and episode['trades']:
                for trade in episode['trades']:
                    # Only analyze SELL/COVER trades (they have duration)
                    if trade.get('action') in ['SELL', 'COVER', 'BUY_COVER'] and trade.get('duration') is not None:
                        duration = trade['duration']
                        all_durations.append(duration)
                        
                        # Classify by P&L
                        if trade.get('pnl', 0) > 0:
                            winning_durations.append(duration)
                        elif trade.get('pnl', 0) < 0:
                            losing_durations.append(duration)
        
        # Display statistics if we have data
        if all_durations:
            lines.append(f"  Total Closed Trades:     {len(all_durations)}")
            lines.append(f"  Avg Duration:            {np.mean(all_durations):.1f} bars")
            lines.append(f"  Median Duration:         {np.median(all_durations):.1f} bars")
            lines.append(f"  Min Duration:            {min(all_durations)} bars")
            lines.append(f"  Max Duration:            {max(all_durations)} bars")
            lines.append("")
            
            # Duration by outcome
            if winning_durations:
                lines.append(f"  Winning Trades:")
                lines.append(f"    Count:                 {len(winning_durations)}")
                lines.append(f"    Avg Duration:          {np.mean(winning_durations):.1f} bars")
                lines.append(f"    Median Duration:       {np.median(winning_durations):.1f} bars")
            
            if losing_durations:
                lines.append(f"  Losing Trades:")
                lines.append(f"    Count:                 {len(losing_durations)}")
                lines.append(f"    Avg Duration:          {np.mean(losing_durations):.1f} bars")
                lines.append(f"    Median Duration:       {np.median(losing_durations):.1f} bars")
            
            # Duration distribution
            lines.append("")
            lines.append("  Duration Distribution:")
            
            # Create bins: <20, 20-100, 100-200, 200+
            ultra_short = sum(1 for d in all_durations if d < 20)
            optimal = sum(1 for d in all_durations if 20 <= d <= 100)
            long_hold = sum(1 for d in all_durations if 100 < d <= 200)
            very_long = sum(1 for d in all_durations if d > 200)
            
            total_trades = len(all_durations)
            lines.append(f"    < 20 bars (ultra-short): {ultra_short} ({ultra_short/total_trades*100:.1f}%)")
            lines.append(f"    20-100 bars (optimal):   {optimal} ({optimal/total_trades*100:.1f}%)")
            lines.append(f"    100-200 bars (long):     {long_hold} ({long_hold/total_trades*100:.1f}%)")
            lines.append(f"    > 200 bars (very long):  {very_long} ({very_long/total_trades*100:.1f}%)")
            
            # Highlight if agent is holding positions optimally
            if optimal / total_trades > 0.5:
                lines.append("")
                lines.append("  âœ… Agent is holding positions in optimal range (20-100 bars)")
            elif ultra_short / total_trades > 0.5:
                lines.append("")
                lines.append("  âš ï¸ Agent is closing positions too quickly (mostly < 20 bars)")
            elif (long_hold + very_long) / total_trades > 0.5:
                lines.append("")
                lines.append("  âš ï¸ Agent is holding positions too long (mostly > 100 bars)")
        else:
            lines.append("  No completed trades with duration data")
        
        lines.append("")
        return lines
    
    def _section_learning_progress(self, episode_results: List[Dict]) -> List[str]:
        """Learning progress analysis"""
        lines = []
        lines.append("=" * 100)
        lines.append("5. LEARNING PROGRESS")
        lines.append("=" * 100)
        
        epsilons = [e.get('epsilon', 1.0) for e in episode_results]
        
        lines.append("EXPLORATION vs EXPLOITATION:")
        lines.append(f"  Starting Epsilon:        {epsilons[0]:.3f} (100% random actions)")
        lines.append(f"  Final Epsilon:           {epsilons[-1]:.3f} ({epsilons[-1]*100:.1f}% random actions)")
        lines.append(f"  Epsilon Decay:           {' Proper decay curve' if epsilons[0] > epsilons[-1] else ' No decay'}")
        lines.append("")
        
        lines.append("REWARD PROGRESSION (10-Episode Windows):")
        for i in range(0, len(episode_results), 10):
            window = episode_results[i:min(i+10, len(episode_results))]
            avg_reward = np.mean([e['total_reward'] for e in window])
            lines.append(f"  Episodes {i+1:3d}-{min(i+10, len(episode_results)):3d}:     Avg Reward = {avg_reward:8.2f}")
        lines.append("")
        
        return lines
    
    def _section_final_evaluation(self, env, agent) -> List[str]:
        """Final agent evaluation"""
        lines = []
        lines.append("=" * 100)
        lines.append("6. FINAL EVALUATION")
        lines.append("=" * 100)
        
        # Run one final evaluation episode
        try:
            obs = env.reset()
            done = False
            total_reward = 0
            actions_taken = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
            
            while not done:
                action = agent.act(obs, training=False)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if action == 0:
                    actions_taken['HOLD'] += 1
                elif action == 1:
                    actions_taken['BUY'] += 1
                else:
                    actions_taken['SELL'] += 1
                
                if truncated:
                    break
            
            lines.append("EVALUATION RUN (Trained Agent):")
            lines.append(f"  Total Reward:            {total_reward:.2f}")
            lines.append(f"  Final Portfolio Value:   ${info['portfolio_value']:,.2f}")
            lines.append(f"  Total Trades:            {info['num_trades']}")
            lines.append(f"  Win Rate:                {info['win_rate']*100:.1f}%")
            lines.append(f"  Max Drawdown:            {info['max_drawdown']*100:.2f}%")
            lines.append(f"  Sharpe Ratio:            {info['sharpe_ratio']:.2f}")
            lines.append("")
            
            lines.append("ACTION DISTRIBUTION:")
            total_actions = sum(actions_taken.values())
            for action, count in actions_taken.items():
                pct = (count / total_actions) * 100
                lines.append(f"  {action:6s}: {count:4d} times ({pct:5.1f}%)")
            
        except Exception as e:
            lines.append(f" Could not run final evaluation: {e}")
        
        lines.append("")
        return lines
    
    def _section_best_episodes(self, episode_results: List[Dict]) -> List[str]:
        """Best episodes breakdown"""
        lines = []
        lines.append("=" * 100)
        lines.append("7. TOP 5 BEST EPISODES")
        lines.append("=" * 100)
        
        # Sort by reward
        sorted_episodes = sorted(episode_results, key=lambda x: x['total_reward'], reverse=True)
        
        for i, episode in enumerate(sorted_episodes[:5], 1):
            lines.append(f"\n RANK #{i} - Episode {episode['episode']}:")
            lines.append(f"  Total Reward:            {episode['total_reward']:.2f}")
            lines.append(f"  Final Portfolio:         ${episode.get('portfolio_value', 0):,.2f}")
            lines.append(f"  Trades Executed:         {episode.get('num_trades', 0)}")
            lines.append(f"  Win Rate:                {episode.get('win_rate', 0)*100:.1f}%")
            lines.append(f"  Max Drawdown:            {episode.get('max_drawdown', 0)*100:.2f}%")
            lines.append(f"  Epsilon:                 {episode.get('epsilon', 0):.3f}")
        
        lines.append("")
        return lines
    
    def _section_top_trades(self, episode_results: List[Dict]) -> List[str]:
        """
        NEW SECTION: Display top profitable trades with asset and timestamp
        """
        lines = []
        lines.append("=" * 100)
        lines.append("8. TOP 10 MOST PROFITABLE TRADES")
        lines.append("=" * 100)
        
        # Collect all trades from all episodes
        all_trades = []
        for episode in episode_results:
            if 'trades' in episode and episode['trades']:
                episode_num = episode['episode']
                asset = episode.get('asset', episode.get('symbol', 'UNKNOWN'))
                
                for trade in episode['trades']:
                    if trade.get('pnl') is not None and trade['pnl'] != 0:
                        trade_info = {
                            'episode': episode_num,
                            'asset': asset,
                            'timestamp': trade.get('timestamp', 'N/A'),
                            'action': trade['action'],
                            'price': trade['price'],
                            'size': trade['size'],
                            'pnl': trade['pnl'],
                            'fees': trade.get('fees', 0)
                        }
                        all_trades.append(trade_info)
        
        if not all_trades:
            lines.append(" No completed trades found in episode data")
            lines.append("  (Note: Trades must close positions to have PnL)")
            lines.append("")
            return lines
        
        # Sort by PnL
        top_trades = sorted(all_trades, key=lambda x: x['pnl'], reverse=True)[:10]
        
        lines.append(f"\n Analyzed {len(all_trades)} completed trades across all episodes\n")
        
        for i, trade in enumerate(top_trades, 1):
            timestamp_str = trade['timestamp']
            if isinstance(timestamp_str, pd.Timestamp):
                timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(timestamp_str, datetime):
                timestamp_str = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp_str)
            
            lines.append(f"{'='*80}")
            lines.append(f"RANK #{i}")
            lines.append(f"{'='*80}")
            lines.append(f"  Episode:                 #{trade['episode']}")
            lines.append(f"  Asset:                   {trade['asset']}")
            lines.append(f"  Timestamp:               {timestamp_str}")
            lines.append(f"  Action:                  {trade['action']}")
            lines.append(f"  Price:                   ${trade['price']:,.2f}")
            lines.append(f"  Size:                    {trade['size']:.6f}")
            lines.append(f"  PnL:                     ${trade['pnl']:,.2f} ")
            lines.append(f"  Fees Paid:               ${trade['fees']:.2f}")
            lines.append(f"  Net Profit:              ${trade['pnl'] - trade['fees']:,.2f}")
            lines.append("")
        
        # Summary statistics
        total_pnl = sum(t['pnl'] for t in all_trades)
        total_fees = sum(t['fees'] for t in all_trades)
        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        losing_trades = [t for t in all_trades if t['pnl'] < 0]
        
        lines.append("\n" + "="*80)
        lines.append("TRADE STATISTICS SUMMARY")
        lines.append("="*80)
        lines.append(f"  Total Completed Trades:  {len(all_trades)}")
        lines.append(f"  Winning Trades:          {len(winning_trades)} ({len(winning_trades)/len(all_trades)*100:.1f}%)")
        lines.append(f"  Losing Trades:           {len(losing_trades)} ({len(losing_trades)/len(all_trades)*100:.1f}%)")
        lines.append(f"  Total PnL:               ${total_pnl:,.2f}")
        lines.append(f"  Total Fees:              ${total_fees:,.2f}")
        lines.append(f"  Net PnL:                 ${total_pnl - total_fees:,.2f}")
        
        if winning_trades:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            lines.append(f"  Avg Win Size:            ${avg_win:,.2f}")
        
        if losing_trades:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            lines.append(f"  Avg Loss Size:           ${avg_loss:,.2f}")
        
        if winning_trades and losing_trades:
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in losing_trades))
            lines.append(f"  Profit Factor:           {profit_factor:.2f}")
        
        lines.append("")
        return lines
    
    def _section_recommendations(self, episode_results: List[Dict], agent) -> List[str]:
        """Recommendations section"""
        lines = []
        lines.append("=" * 100)
        lines.append("9. RECOMMENDATIONS")
        lines.append("=" * 100)
        
        # Analyze performance
        recent_rewards = [e['total_reward'] for e in episode_results[-20:]]
        avg_recent_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        
        lines.append("TRAINING ANALYSIS:")
        
        # Check convergence
        if reward_std < 50:
            lines.append("   Agent appears to have converged")
            lines.append("   Consider deploying to paper trading")
        else:
            lines.append("   High variance in recent episodes")
            lines.append("   Consider training for more episodes")
        
        # Check profitability
        if avg_recent_reward > 0:
            lines.append("   Agent showing positive rewards")
            lines.append("   Good candidate for live testing")
        else:
            lines.append("   Agent showing negative rewards")
            lines.append("   Review reward function and features")
        
        # Check exploration
        final_epsilon = episode_results[-1].get('epsilon', 0)
        if final_epsilon < 0.1:
            lines.append("   Exploration has decayed properly")
        else:
            lines.append("   Epsilon still high - may need more episodes")
        
        lines.append("\nNEXT STEPS:")
        lines.append("  1. Review top performing trades and patterns")
        lines.append("  2. Test agent on different market conditions")
        lines.append("  3. Consider walk-forward validation")
        lines.append("  4. Start paper trading if performance is consistent")
        
        lines.append("")
        return lines


# Convenience function for quick reporting
def generate_rl_report(episode_results: List[Dict],
                       env,
                       agent,
                       config: Dict,
                       save_path: str = None) -> str:
    """
    Quick function to generate RL training report
    
    Args:
        episode_results: List of episode statistics
        env: Trading environment
        agent: Trained RL agent
        config: Training configuration
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    reporter = RLTrainingReporter()
    return reporter.generate_full_report(
        episode_results=episode_results,
        env=env,
        agent=agent,
        config=config,
        save_path=save_path
    )