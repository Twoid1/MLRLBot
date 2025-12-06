"""
OPTIMIZED RL Reporter with PERCENTILE ANALYSIS
================================================

Key Features:
1. 10-100x faster report generation
2. ⭐ NEW: Percentile breakdown (0-10%, 10-20%, ..., 90-100%)
3. Identifies where performance degrades
4. Easy-to-read training progression tables
5. Configurable detail levels

Usage:
    from rl_reporter_fast import FastRLTrainingReporter
    
    reporter = FastRLTrainingReporter()
    report = reporter.generate_full_report(
        episode_results=episode_results,
        env=env,
        agent=agent,
        config=config,
        detail_level='standard'  # 'summary', 'standard', or 'full'
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import heapq


class PercentileAnalyzer:
    """
    Analyzes training metrics across percentile buckets
    """
    
    def __init__(self, episode_results: List[Dict], num_buckets: int = 10):
        self.episode_results = episode_results
        self.num_buckets = num_buckets
        self.n = len(episode_results)
        self.bucket_size = max(1, self.n // num_buckets)
        self.buckets = self._create_buckets()
        
    def _create_buckets(self) -> List[List[Dict]]:
        """Split episodes into percentile buckets"""
        buckets = []
        for i in range(self.num_buckets):
            start_idx = i * self.bucket_size
            end_idx = start_idx + self.bucket_size if i < self.num_buckets - 1 else self.n
            buckets.append(self.episode_results[start_idx:end_idx])
        return buckets
    
    def get_bucket_label(self, bucket_idx: int) -> str:
        """Get human-readable label for bucket"""
        start_pct = bucket_idx * 10
        end_pct = (bucket_idx + 1) * 10
        return f"{start_pct:3d}-{end_pct:3d}%"
    
    def analyze_metric(self, metric_key: str, default: float = 0.0) -> Dict:
        """Analyze a single metric across all buckets"""
        bucket_avgs = []
        bucket_stds = []
        
        for bucket in self.buckets:
            values = [e.get(metric_key, default) for e in bucket]
            if values:
                bucket_avgs.append(np.mean(values))
                bucket_stds.append(np.std(values))
            else:
                bucket_avgs.append(default)
                bucket_stds.append(0)
        
        # Determine trend
        first_half_avg = np.mean(bucket_avgs[:5]) if len(bucket_avgs) >= 5 else np.mean(bucket_avgs)
        second_half_avg = np.mean(bucket_avgs[5:]) if len(bucket_avgs) > 5 else first_half_avg
        
        avg_std = np.mean(bucket_stds) if bucket_stds else 0
        overall_avg = np.mean(bucket_avgs) if bucket_avgs else 0
        
        if avg_std > abs(overall_avg) * 0.5 and overall_avg != 0:
            trend = 'volatile'
        elif second_half_avg > first_half_avg * 1.1:
            trend = 'improving'
        elif second_half_avg < first_half_avg * 0.9:
            trend = 'degrading'
        else:
            trend = 'stable'
        
        change_pct = ((second_half_avg - first_half_avg) / abs(first_half_avg) * 100) if first_half_avg != 0 else 0
        
        return {
            'bucket_avgs': bucket_avgs,
            'bucket_stds': bucket_stds,
            'overall_avg': overall_avg,
            'trend': trend,
            'best_bucket': int(np.argmax(bucket_avgs)) if bucket_avgs else 0,
            'worst_bucket': int(np.argmin(bucket_avgs)) if bucket_avgs else 0,
            'first_half_avg': first_half_avg,
            'second_half_avg': second_half_avg,
            'change_pct': change_pct
        }
    
    def analyze_trades(self) -> Dict:
        """Comprehensive trade analysis across buckets"""
        results = {
            'win_rates': [],
            'avg_wins': [],
            'avg_losses': [],
            'total_trades': [],
            'total_pnl': [],
            'avg_pnl_per_trade': [],
            'profit_factors': [],
            'avg_hold_duration': [],
        }
        
        for bucket in self.buckets:
            bucket_trades = []
            bucket_hold_durations = []
            
            for ep in bucket:
                if 'trades' in ep and ep['trades']:
                    for trade in ep['trades']:
                        if trade.get('pnl') is not None:
                            bucket_trades.append(trade)
                            if 'hold_duration' in trade:
                                bucket_hold_durations.append(trade['hold_duration'])
                
                if 'hold_steps' in ep:
                    bucket_hold_durations.append(ep['hold_steps'])
                
                # Also get episode-level PnL
                if 'pnl' in ep and ep['pnl'] is not None and not bucket_trades:
                    bucket_trades.append({'pnl': ep['pnl']})
            
            if bucket_trades:
                wins = [t['pnl'] for t in bucket_trades if t.get('pnl', 0) > 0]
                losses = [t['pnl'] for t in bucket_trades if t.get('pnl', 0) < 0]
                all_pnl = [t.get('pnl', 0) for t in bucket_trades]
                
                results['win_rates'].append(len(wins) / len(bucket_trades) * 100 if bucket_trades else 0)
                results['avg_wins'].append(np.mean(wins) if wins else 0)
                results['avg_losses'].append(np.mean(losses) if losses else 0)
                results['total_trades'].append(len(bucket_trades))
                results['total_pnl'].append(sum(all_pnl))
                results['avg_pnl_per_trade'].append(np.mean(all_pnl) if all_pnl else 0)
                
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses)) if losses else 1
                results['profit_factors'].append(total_wins / total_losses if total_losses > 0 else total_wins)
                
                results['avg_hold_duration'].append(np.mean(bucket_hold_durations) if bucket_hold_durations else 0)
            else:
                # Use episode-level stats
                win_rates = [ep.get('win_rate', 0) for ep in bucket]
                num_trades = [ep.get('num_trades', 0) for ep in bucket]
                
                results['win_rates'].append(np.mean(win_rates) * 100 if win_rates else 0)
                results['avg_wins'].append(0)
                results['avg_losses'].append(0)
                results['total_trades'].append(sum(num_trades))
                results['total_pnl'].append(0)
                results['avg_pnl_per_trade'].append(0)
                results['profit_factors'].append(0)
                results['avg_hold_duration'].append(np.mean(bucket_hold_durations) if bucket_hold_durations else 0)
        
        return results


class FastRLTrainingReporter:
    """Fast comprehensive training reports with percentile analysis"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_full_report(self, 
                            episode_results: List[Dict],
                            env=None,
                            agent=None,
                            config: Dict = None,
                            save_path: str = None,
                            detail_level: str = 'standard') -> str:
        """
        Generate training report with percentile analysis
        
        Args:
            episode_results: List of episode statistics
            env: Trading environment (optional)
            agent: Trained RL agent (optional)
            config: Training configuration
            save_path: Where to save the report
            detail_level: 'summary', 'standard', or 'full'
            
        Returns:
            Formatted report string
        """
        config = config or {}
        
        print("\n Generating training report with percentile analysis...")
        report_lines = []
        
        # Header
        report_lines.append("=" * 120)
        report_lines.append("RL TRAINING REPORT - WITH PERCENTILE ANALYSIS")
        report_lines.append("=" * 120)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Episodes: {len(episode_results)}")
        report_lines.append(f"Detail Level: {detail_level.upper()}")
        report_lines.append("")
        
        # Create analyzer
        analyzer = PercentileAnalyzer(episode_results)
        
        # 1. Quick Summary (always)
        print("   [1/8] Quick summary...")
        report_lines.extend(self._section_quick_summary(episode_results, analyzer))
        
        # 2. PERCENTILE BREAKDOWN - Main Feature!
        print("   [2/8] Percentile breakdown...")
        report_lines.extend(self._section_percentile_breakdown(analyzer))
        
        # 3. Training Configuration
        print("   [3/8] Configuration...")
        report_lines.extend(self._section_training_config(config))
        
        # 4. Reward Analysis
        print("   [4/8] Reward analysis...")
        report_lines.extend(self._section_reward_analysis(analyzer))
        
        # 5. Trade Analysis
        if detail_level in ['standard', 'full']:
            print("   [5/8] Trade analysis...")
            report_lines.extend(self._section_trade_analysis(analyzer))
        
        # 6. P&L Analysis
        if detail_level in ['standard', 'full']:
            print("   [6/8] P&L analysis...")
            report_lines.extend(self._section_pnl_analysis(analyzer))
        
        # 7. Degradation Detection
        print("   [7/8] Degradation detection...")
        report_lines.extend(self._section_degradation_detection(analyzer))
        
        # 8. Recommendations
        print("   [8/8] Recommendations...")
        report_lines.extend(self._section_recommendations(analyzer, episode_results))
        
        # 9. Raw Data (full only)
        if detail_level == 'full':
            report_lines.extend(self._section_raw_data_table(analyzer))
        
        # Combine report
        report = "\n".join(report_lines)
        
        # Save
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"\n Report saved to: {save_path}")
        
        return report
    
    def _section_quick_summary(self, episode_results: List[Dict], analyzer: PercentileAnalyzer) -> List[str]:
        """Quick overview"""
        lines = []
        lines.append("=" * 120)
        lines.append("1. QUICK SUMMARY")
        lines.append("=" * 120)
        
        rewards = [e.get('total_reward', 0) for e in episode_results]
        portfolios = [e.get('portfolio_value', 10000) for e in episode_results]
        
        lines.append(f"  Total Episodes:          {len(episode_results)}")
        lines.append(f"  Overall Avg Reward:      {np.mean(rewards):.2f}")
        lines.append(f"  Overall Avg Portfolio:   ${np.mean(portfolios):,.2f}")
        lines.append(f"  Best Reward:             {max(rewards):.2f}")
        lines.append(f"  Worst Reward:            {min(rewards):.2f}")
        lines.append("")
        
        reward_analysis = analyzer.analyze_metric('total_reward')
        
        trend_emoji = {
            'improving': ' IMPROVING',
            'degrading': ' DEGRADING', 
            'stable': '  STABLE',
            'volatile': ' VOLATILE'
        }
        
        lines.append(f"  Training Trend:          {trend_emoji.get(reward_analysis['trend'], 'UNKNOWN')}")
        lines.append(f"  First Half Avg:          {reward_analysis['first_half_avg']:.2f}")
        lines.append(f"  Second Half Avg:         {reward_analysis['second_half_avg']:.2f}")
        lines.append(f"  Change:                  {reward_analysis['change_pct']:+.1f}%")
        lines.append("")
        
        return lines
    
    def _section_percentile_breakdown(self, analyzer: PercentileAnalyzer) -> List[str]:
        """THE MAIN FEATURE: All metrics by training percentile"""
        lines = []
        lines.append("=" * 120)
        lines.append("2.  PERCENTILE BREAKDOWN - ALL METRICS BY TRAINING PROGRESS ")
        lines.append("=" * 120)
        lines.append("")
        lines.append("This shows how each metric changes as training progresses.")
        lines.append("Look for where values START TO DROP - that's when overfitting may begin!")
        lines.append("")
        
        # Collect metrics
        reward_data = analyzer.analyze_metric('total_reward')
        portfolio_data = analyzer.analyze_metric('portfolio_value', default=10000)
        trade_data = analyzer.analyze_trades()
        epsilon_data = analyzer.analyze_metric('epsilon', default=1.0)
        hold_data = analyzer.analyze_metric('hold_steps', default=0)
        
        # Header
        lines.append("=" * 120)
        lines.append(f"{'Bucket':<10} | {'Reward':>10} | {'Portfolio':>12} | {'Win Rate':>10} | {'Trades':>8} | "
                    f"{'Avg P&L':>10} | {'Hold':>8} | {'Epsilon':>8}")
        lines.append("=" * 120)
        
        for i in range(analyzer.num_buckets):
            label = analyzer.get_bucket_label(i)
            reward = reward_data['bucket_avgs'][i]
            portfolio = portfolio_data['bucket_avgs'][i]
            win_rate = trade_data['win_rates'][i] if i < len(trade_data['win_rates']) else 0
            trades = trade_data['total_trades'][i] if i < len(trade_data['total_trades']) else 0
            avg_pnl = trade_data['avg_pnl_per_trade'][i] if i < len(trade_data['avg_pnl_per_trade']) else 0
            hold = trade_data['avg_hold_duration'][i] if i < len(trade_data['avg_hold_duration']) else hold_data['bucket_avgs'][i]
            epsilon = epsilon_data['bucket_avgs'][i]
            
            # Highlight best/worst
            reward_marker = " Up" if i == reward_data['best_bucket'] else (" Down" if i == reward_data['worst_bucket'] else "  ")
            
            lines.append(f"{label:<10} | {reward:>8.2f}{reward_marker} | ${portfolio:>10,.0f} | {win_rate:>8.1f}% | "
                        f"{trades:>8.0f} | ${avg_pnl:>8.2f} | {hold:>8.1f} | {epsilon:>8.3f}")
        
        lines.append("=" * 120)
        lines.append("  Legend: Up = Best bucket, Down = Worst bucket")
        lines.append("=")
        
        return lines
    
    def _section_training_config(self, config: Dict) -> List[str]:
        """Training configuration"""
        lines = []
        lines.append("=" * 120)
        lines.append("3. TRAINING CONFIGURATION")
        lines.append("=" * 120)
        
        lines.append(f"  Episodes:                {config.get('rl_episodes', 'N/A')}")
        lines.append(f"  Initial Balance:         ${config.get('initial_balance', 10000):,.2f}")
        lines.append(f"  Fee Rate:                {config.get('fee_rate', 0.0026)*100:.2f}%")
        lines.append(f"  Network:                 {config.get('rl_hidden_dims', [256, 256, 128])}")
        lines.append(f"  Batch Size:              {config.get('rl_batch_size', 256)}")
        lines.append(f"  Learning Rate:           {config.get('rl_learning_rate', 0.0001)}")
        lines.append(f"  Gamma:                   {config.get('rl_gamma', 0.99)}")
        lines.append(f"  Multi-Objective:         {config.get('use_multi_objective', False)}")
        lines.append("")
        
        return lines
    
    def _section_reward_analysis(self, analyzer: PercentileAnalyzer) -> List[str]:
        """Reward analysis with visual bars"""
        lines = []
        lines.append("=" * 120)
        lines.append("4. REWARD PROGRESSION")
        lines.append("=" * 120)
        lines.append("")
        
        reward_data = analyzer.analyze_metric('total_reward')
        
        lines.append("REWARD BY BUCKET:")
        lines.append("=" * 80)
        
        max_reward = max(abs(r) for r in reward_data['bucket_avgs']) or 1
        
        for i, (avg, std) in enumerate(zip(reward_data['bucket_avgs'], reward_data['bucket_stds'])):
            label = analyzer.get_bucket_label(i)
            bar_len = int(abs(avg) / max_reward * 30)
            
            if avg >= 0:
                bar = "|" * bar_len
                marker = ""
            else:
                bar = "l" * bar_len
                marker = " (neg)"
            
            # Mark best/worst
            if i == reward_data['best_bucket']:
                marker = "  BEST"
            elif i == reward_data['worst_bucket']:
                marker = "  WORST"
            
            lines.append(f"  {label} │ {avg:>8.2f} ± {std:>6.2f} │ {bar}{marker}")
        
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"  Trend:           {reward_data['trend'].upper()}")
        lines.append(f"  First->Second:    {reward_data['change_pct']:+.1f}%")
        lines.append("")
        
        return lines
    
    def _section_trade_analysis(self, analyzer: PercentileAnalyzer) -> List[str]:
        """Trade analysis"""
        lines = []
        lines.append("=" * 120)
        lines.append("5. TRADE ANALYSIS")
        lines.append("=" * 120)
        lines.append("")
        
        trade_data = analyzer.analyze_trades()
        
        # Win rate progression
        lines.append("WIN RATE PROGRESSION:")
        lines.append("=" * 80)
        
        for i, wr in enumerate(trade_data['win_rates']):
            label = analyzer.get_bucket_label(i)
            bar_len = int(wr / 100 * 30)
            bar = "|" * bar_len
            lines.append(f"  {label} | {wr:>5.1f}% | {bar}")
        
        lines.append("=" * 80)
        lines.append("")
        
        # Win vs Loss
        lines.append("AVERAGE WIN vs LOSS:")
        lines.append("=" * 80)
        lines.append(f"  {'Bucket':<10} | {'Avg Win':>10} | {'Avg Loss':>10} | {'Ratio':>8} | {'Profit Factor':>12}")
        lines.append("=" * 80)
        
        for i in range(analyzer.num_buckets):
            label = analyzer.get_bucket_label(i)
            avg_win = trade_data['avg_wins'][i] if i < len(trade_data['avg_wins']) else 0
            avg_loss = trade_data['avg_losses'][i] if i < len(trade_data['avg_losses']) else 0
            pf = trade_data['profit_factors'][i] if i < len(trade_data['profit_factors']) else 0
            
            ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            ratio_str = f"{ratio:.2f}" if ratio < 100 else "infinity"
            
            lines.append(f"  {label:<10} | ${avg_win:>8.2f} | ${avg_loss:>8.2f} | {ratio_str:>8} | {pf:>12.2f}")
        
        lines.append("=" * 80)
        lines.append("")
        
        return lines
    
    def _section_pnl_analysis(self, analyzer: PercentileAnalyzer) -> List[str]:
        """P&L analysis"""
        lines = []
        lines.append("=" * 120)
        lines.append("6. P&L ANALYSIS")
        lines.append("=" * 120)
        lines.append("")
        
        trade_data = analyzer.analyze_trades()
        portfolio_data = analyzer.analyze_metric('portfolio_value', default=10000)
        
        lines.append("CUMULATIVE P&L:")
        lines.append("=" * 80)
        
        cumulative_pnl = 0
        for i, pnl in enumerate(trade_data['total_pnl']):
            label = analyzer.get_bucket_label(i)
            cumulative_pnl += pnl
            
            if pnl > 0:
                indicator = f"+${pnl:,.0f}"
            elif pnl < 0:
                indicator = f"-${abs(pnl):,.0f}"
            else:
                indicator = "$0"
            
            lines.append(f"  {label} | Period: {indicator:>10} | Cumulative: ${cumulative_pnl:>12,.0f}")
        
        lines.append("=" * 80)
        lines.append("")
        
        # Portfolio progression
        lines.append("PORTFOLIO VALUE:")
        lines.append("=" * 80)
        
        initial = 10000
        for i, avg in enumerate(portfolio_data['bucket_avgs']):
            label = analyzer.get_bucket_label(i)
            change_pct = (avg - initial) / initial * 100
            
            bar_len = int(abs(change_pct) / 5)
            if change_pct >= 0:
                bar = "|" * min(bar_len, 20)
            else:
                bar = "l" * min(bar_len, 20)
            
            lines.append(f"  {label} | ${avg:>10,.0f} | {change_pct:>+7.1f}% | {bar}")
        
        lines.append("=" * 80)
        lines.append("")
        
        return lines
    
    def _section_degradation_detection(self, analyzer: PercentileAnalyzer) -> List[str]:
        """Detect where performance degrades"""
        lines = []
        lines.append("=" * 120)
        lines.append("7.   DEGRADATION DETECTION")
        lines.append("=" * 120)
        lines.append("")
        
        reward_data = analyzer.analyze_metric('total_reward')
        rewards = reward_data['bucket_avgs']
        
        # Find peak
        peak_bucket = int(np.argmax(rewards))
        peak_reward = rewards[peak_bucket]
        
        if peak_bucket < len(rewards) - 1:
            post_peak_rewards = rewards[peak_bucket + 1:]
            post_peak_avg = np.mean(post_peak_rewards)
            degradation_pct = ((post_peak_avg - peak_reward) / abs(peak_reward) * 100) if peak_reward != 0 else 0
            
            lines.append("ANALYSIS:")
            lines.append("=" * 80)
            lines.append(f"  Peak Performance:        Bucket {peak_bucket + 1} ({analyzer.get_bucket_label(peak_bucket)})")
            lines.append(f"  Peak Avg Reward:         {peak_reward:.2f}")
            lines.append(f"  Post-Peak Avg:           {post_peak_avg:.2f}")
            lines.append(f"  Degradation:             {degradation_pct:.1f}%")
            lines.append("")
            
            if degradation_pct < -20:
                lines.append("    SIGNIFICANT DEGRADATION DETECTED!")
                lines.append(f"      Performance dropped {abs(degradation_pct):.1f}% after bucket {peak_bucket + 1}")
                
                optimal_episodes = int(len(analyzer.episode_results) * (peak_bucket + 1) / 10)
                lines.append(f"")
                lines.append(f"   RECOMMENDATION: Train for ~{optimal_episodes} episodes instead")
                lines.append(f"      (Currently at {len(analyzer.episode_results)} episodes)")
            elif degradation_pct < -10:
                lines.append("   MODERATE DEGRADATION:")
                lines.append(f"      Some performance loss ({abs(degradation_pct):.1f}%) in later training")
            else:
                lines.append("   NO SIGNIFICANT DEGRADATION")
                lines.append("      Performance remains stable throughout training")
        else:
            lines.append("  Peak at final bucket - still improving!")
            lines.append("  Consider training longer.")
        
        lines.append("=" * 80)
        lines.append("")
        
        return lines
    
    def _section_recommendations(self, analyzer: PercentileAnalyzer, episode_results: List[Dict]) -> List[str]:
        """Generate recommendations"""
        lines = []
        lines.append("=" * 120)
        lines.append("8. RECOMMENDATIONS")
        lines.append("=" * 120)
        lines.append("")
        
        reward_data = analyzer.analyze_metric('total_reward')
        trade_data = analyzer.analyze_trades()
        
        recommendations = []
        
        # Trend-based
        if reward_data['trend'] == 'degrading':
            peak_bucket = reward_data['best_bucket']
            optimal_pct = (peak_bucket + 1) * 10
            optimal_eps = int(len(episode_results) * optimal_pct / 100)
            recommendations.append(f"  REDUCE TRAINING: Performance degrades after {optimal_pct}%")
            recommendations.append(f"     Try {optimal_eps} episodes instead of {len(episode_results)}")
        
        if reward_data['bucket_avgs'][-1] > reward_data['bucket_avgs'][-2]:
            recommendations.append(" STILL IMPROVING: Consider more episodes")
        
        if reward_data['trend'] == 'volatile':
            recommendations.append(" HIGH VOLATILITY: Try reducing learning rate or increasing batch size")
        
        # Win rate
        win_rates = trade_data['win_rates']
        if win_rates:
            avg_wr = np.mean(win_rates)
            if avg_wr < 40:
                recommendations.append(f" LOW WIN RATE ({avg_wr:.1f}%): Review entry conditions")
            elif avg_wr > 55:
                recommendations.append(f" GOOD WIN RATE ({avg_wr:.1f}%)")
        
        # Profit factor
        pf_values = trade_data['profit_factors']
        if pf_values:
            avg_pf = np.mean([p for p in pf_values if p > 0])
            if avg_pf < 1.0:
                recommendations.append(f"  PROFIT FACTOR < 1 ({avg_pf:.2f}): Losing money on average")
            elif avg_pf > 1.5:
                recommendations.append(f" STRONG PROFIT FACTOR ({avg_pf:.2f})")
        
        if recommendations:
            for rec in recommendations:
                lines.append(f"  {rec}")
        else:
            lines.append("   Training looks healthy!")
        
        lines.append("")
        lines.append("NEXT STEPS:")
        lines.append("  1. If degradation detected -> Retrain with fewer episodes")
        lines.append("  2. Run validation -> python main.py backtest --walk-forward")
        lines.append("  3. Paper trade -> At least 30 days before live")
        lines.append("")
        
        return lines
    
    def _section_raw_data_table(self, analyzer: PercentileAnalyzer) -> List[str]:
        """Raw data for spreadsheet"""
        lines = []
        lines.append("=" * 120)
        lines.append("9. RAW DATA (Tab-Separated - Copy to Spreadsheet)")
        lines.append("=" * 120)
        lines.append("")
        
        reward_data = analyzer.analyze_metric('total_reward')
        portfolio_data = analyzer.analyze_metric('portfolio_value', default=10000)
        trade_data = analyzer.analyze_trades()
        epsilon_data = analyzer.analyze_metric('epsilon', default=1.0)
        
        headers = ["Bucket", "Reward", "Portfolio", "Win_Rate", "Trades", "Avg_PnL", "Profit_Factor", "Epsilon"]
        lines.append("\t".join(headers))
        
        for i in range(analyzer.num_buckets):
            row = [
                analyzer.get_bucket_label(i).strip(),
                f"{reward_data['bucket_avgs'][i]:.2f}",
                f"{portfolio_data['bucket_avgs'][i]:.0f}",
                f"{trade_data['win_rates'][i]:.1f}" if i < len(trade_data['win_rates']) else "0",
                f"{trade_data['total_trades'][i]}" if i < len(trade_data['total_trades']) else "0",
                f"{trade_data['avg_pnl_per_trade'][i]:.2f}" if i < len(trade_data['avg_pnl_per_trade']) else "0",
                f"{trade_data['profit_factors'][i]:.2f}" if i < len(trade_data['profit_factors']) else "0",
                f"{epsilon_data['bucket_avgs'][i]:.4f}"
            ]
            lines.append("\t".join(row))
        
        lines.append("")
        
        return lines


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_percentile_report(episode_results: List[Dict],
                               env=None,
                               agent=None,
                               config: Dict = None,
                               save_path: str = None,
                               detail_level: str = 'standard') -> str:
    """Quick function to generate report"""
    reporter = FastRLTrainingReporter()
    return reporter.generate_full_report(
        episode_results=episode_results,
        env=env,
        agent=agent,
        config=config,
        save_path=save_path,
        detail_level=detail_level
    )


def quick_percentile_summary(episode_results: List[Dict]) -> None:
    """Print quick percentile summary to console"""
    analyzer = PercentileAnalyzer(episode_results)
    
    print("\n" + "=" * 70)
    print("QUICK PERCENTILE SUMMARY")
    print("=" * 70)
    
    reward_data = analyzer.analyze_metric('total_reward')
    trade_data = analyzer.analyze_trades()
    
    print(f"\n{'Bucket':<12} {'Reward':>10} {'Win Rate':>10} {'P&L':>10}")
    print("-" * 45)
    
    for i in range(analyzer.num_buckets):
        label = analyzer.get_bucket_label(i)
        reward = reward_data['bucket_avgs'][i]
        win_rate = trade_data['win_rates'][i] if i < len(trade_data['win_rates']) else 0
        pnl = trade_data['total_pnl'][i] if i < len(trade_data['total_pnl']) else 0
        
        marker = " Up" if i == reward_data['best_bucket'] else (" Down" if i == reward_data['worst_bucket'] else "")
        print(f"{label:<12} {reward:>10.2f} {win_rate:>9.1f}% ${pnl:>8.0f}{marker}")
    
    print("-" * 45)
    print(f"\nTrend: {reward_data['trend'].upper()}")
    print(f"Best: {analyzer.get_bucket_label(reward_data['best_bucket'])}")
    print(f"Change: {reward_data['change_pct']:+.1f}%")
    print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Fast Reporter with Percentile Analysis...")
    
    np.random.seed(42)
    
    # Simulate training with degradation
    sample_results = []
    for i in range(150):
        # Learning curve with degradation at end
        base_reward = -50 + (i * 1.2)
        if i > 100:
            base_reward -= (i - 100) * 1.5  # Degradation after ~67%
        
        noise = np.random.normal(0, 15)
        pnl = base_reward * 10 + np.random.normal(0, 50)
        
        episode = {
            'episode': i + 1,
            'total_reward': base_reward + noise,
            'portfolio_value': 10000 + (base_reward * 30) + np.random.normal(0, 300),
            'win_rate': min(0.65, 0.35 + i * 0.002) + np.random.normal(0, 0.03),
            'num_trades': 1,
            'hold_steps': 25 + i * 0.3 + np.random.normal(0, 5),
            'epsilon': max(0.01, 1.0 - i * 0.007),
            'pnl': pnl,
            'trades': [{'pnl': pnl, 'hold_duration': 25 + i * 0.3}]
        }
        sample_results.append(episode)
    
    # Generate report
    report = generate_percentile_report(
        sample_results,
        save_path="test_percentile_report.txt",
        detail_level='full'
    )
    
    print(report)
    print("\n Test complete!")