"""
Explainability Integration Module
Integrates explainability into training and backtesting workflows
Adds --explain and --verbose flags to show agent thought process
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

from src.explainability import AgentExplainer, PolicyVisualizer

logger = logging.getLogger(__name__)


class ExplainableRL:
    """
    Wrapper that adds explainability to RL training and execution
    """
    
    def __init__(self, 
                 agent,
                 state_feature_names: List[str],
                 action_names: List[str] = ['Hold', 'Buy', 'Sell'],
                 explain_frequency: int = 100,
                 verbose: bool = False,
                 save_dir: Optional[str] = None):
        """
        Args:
            agent: The DQN agent to explain
            state_feature_names: Names of all state dimensions
            action_names: Names of actions
            explain_frequency: How often to print detailed explanations (every N steps)
            verbose: If True, print every decision
            save_dir: Directory to save explanation logs
        """
        self.agent = agent
        self.explainer = AgentExplainer(state_feature_names, action_names)
        self.visualizer = PolicyVisualizer(self.explainer)
        
        self.explain_frequency = explain_frequency
        self.verbose = verbose
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        self.step_counter = 0
        self.episode_counter = 0
        
        logger.info(f"Explainability enabled: freq={explain_frequency}, verbose={verbose}")
    
    def act_with_explanation(self, 
                            state: np.ndarray,
                            context: Optional[Dict] = None,
                            training: bool = True) -> Tuple[int, Dict]:
        """
        Agent acts and provides explanation
        
        Returns:
            action: The chosen action
            explanation: Dict with detailed explanation
        """
        # Get action and Q-values
        action = self.agent.act(state, training=training)
        
        # Get Q-values for explanation
        import torch
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
            q_values = self.agent.q_network(state_tensor).cpu().numpy()[0]
        
        # Generate explanation
        explanation = self.explainer.explain_decision(
            state=state,
            action=action,
            q_values=q_values,
            agent=self.agent,
            context=context
        )
        
        # Print if verbose or at frequency
        self.step_counter += 1
        if self.verbose or (self.step_counter % self.explain_frequency == 0):
            self._print_explanation(explanation, context, self.step_counter)
        
        return action, explanation
    
    def _print_explanation(self, explanation: Dict, context: Optional[Dict], step: int):
        """Print formatted explanation"""
        print("\n" + "="*100)
        print(f"STEP {step} - {explanation['timestamp']}")
        print("="*100)
        
        # Action and confidence
        print(f"\nACTION: {explanation['action_taken'].upper()}")
        print(f"   Q-Value: {explanation['confidence']:.4f}")
        print(f"   Risk: {explanation['risk_assessment']['risk_level']}")
        
        # Q-values comparison
        print(f"\n Q-VALUES COMPARISON:")
        q_vals = explanation['q_values']
        sorted_actions = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)
        for i, (action, q_val) in enumerate(sorted_actions, 1):
            marker = "P" if action == explanation['action_taken'] else f"{i}."
            bar_length = int(q_val * 50) if q_val > 0 else 0
            bar = "l" * bar_length
            print(f"   {marker} {action:10s}: {q_val:8.4f} {bar}")
        
        # Top features
        print(f"\n TOP 5 INFLUENTIAL FEATURES:")
        for i, (feature, attr) in enumerate(list(explanation['feature_attribution'].items())[:5], 1):
            importance_pct = attr['importance'] * 100
            bar_length = int(importance_pct / 2)
            bar = "l" * bar_length
            print(f"   {i}. {feature:45s} {importance_pct:6.2f}% {bar}")
            print(f"       Value: {attr['value']:8.4f}, Rank: #{attr['rank']}")
        
        # Reasoning
        print(f"\n DECISION REASONING:")
        reasoning_lines = explanation['decision_reasoning'].split('|')
        for line in reasoning_lines:
            print(f" ")
        
        # Alternative actions
        print(f"\n ACTIONS NOT TAKEN:")
        for action, details in explanation['alternative_actions'].items():
            diff = explanation['confidence'] - details['q_value']
            print(f"    {action}: Q={details['q_value']:.4f} (worse by {diff:.4f}, {diff/explanation['confidence']*100:.1f}%)")
        
        # Context info
        if context:
            print(f"\n MARKET CONTEXT:")
            if 'price' in context:
                print(f"   Price: ${context['price']:.2f}")
            if 'position' in context:
                pos_str = f"{context['position']:.4f} units" if context['position'] != 0 else "FLAT"
                print(f"   Position: {pos_str}")
            if 'balance' in context:
                print(f"   Balance: ${context['balance']:.2f}")
            if 'ml_prediction' in context:
                ml_pred = context['ml_prediction']
                pred_labels = ['DOWN', 'FLAT', 'UP']
                print(f"   ML Prediction: {pred_labels[np.argmax(ml_pred)]} ({ml_pred.max():.2%} conf)")
        
        # Risk assessment
        if 'warning' in explanation['risk_assessment']:
            print(f"\n  WARNING: {explanation['risk_assessment']['warning']}")
        
        print("="*100 + "\n")

    def save_decision_history(self, filepath: Optional[str] = None):
        """Save decision history to JSON file"""
        if filepath is None and self.save_dir:
            filepath = self.save_dir / "decision_history.json"
        
        if filepath:
            self.explainer.save_decision_history(str(filepath))
            logger.info(f"Decision history saved to: {filepath}")
    
    def episode_summary(self, episode_num: int, episode_reward: float, episode_steps: int):
        """Print summary at end of episode"""
        self.episode_counter += 1
        
        # Get recent patterns
        if len(self.explainer.decision_history) >= 100:
            patterns = self.explainer.analyze_policy_patterns(lookback=episode_steps)
            
            print("\n" + "" + "="*98 + "")
            print(f"  EPISODE {episode_num} SUMMARY" + " "*70 + "")
            print("" + "="*98 + "")
            
            print(f"  Reward: {episode_reward:10.2f}  |  Steps: {episode_steps:5d}  |  Avg Reward/Step: {episode_reward/episode_steps:8.4f}  ")
            
            # Action distribution
            print("" + "="*98 + "")
            print("  ACTION DISTRIBUTION:" + " "*77 + "")
            for action, stats in patterns['action_distribution'].items():
                pct = stats['percentage']
                bar_length = int(pct / 2)
                bar = "l" * bar_length
                print(f"    {action:8s}: {stats['count']:4d} times ({pct:5.1f}%) {bar:40s}  ")
            
            # Top features this episode
            if patterns['feature_importance_ranking']:
                print("" + "="*98 + "")
                print("  TOP 3 FEATURES THIS EPISODE:" + " "*66 + "")
                for i, feat_info in enumerate(patterns['feature_importance_ranking'][:3], 1):
                    feat_name = feat_info['feature'][:45]  # Truncate if too long
                    importance = feat_info['avg_importance']
                    print(f"    {i}. {feat_name:45s}  Importance: {importance:.4f}" + " "*20 + "")
            
            # Confidence stats
            print("" + "="*98 + "")
            conf_stats = patterns['decision_confidence']
            print(f"  CONFIDENCE: Avg={conf_stats['avg_confidence']:.3f}, Std={conf_stats['confidence_std']:.3f}, Trend={conf_stats['confidence_trend']:12s}  ")
            
            print("" + "="*98 + "\n")
    
    def save_episode_report(self, episode_num: int):
        """Save detailed report for this episode"""
        if not self.save_dir:
            return
        
        report_path = self.save_dir / f"episode_{episode_num:04d}_report.txt"
        report = self.explainer.generate_policy_report()
        
        with open(report_path, 'w') as f:
            f.write(f"EPISODE {episode_num} POLICY REPORT\n")
            f.write(f"{'='*80}\n\n")
            f.write(report)
        
        logger.info(f"Episode report saved to: {report_path}")
    
    def save_episode_visualizations(self, episode_num: int):
        """Save visualizations for this episode"""
        if not self.save_dir:
            return
        
        viz_dir = self.save_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Action distribution
            self.visualizer.plot_action_distribution(
                save_path=viz_dir / f"episode_{episode_num:04d}_actions.png"
            )
            
            # Confidence over time
            self.visualizer.plot_decision_confidence_over_time(
                save_path=viz_dir / f"episode_{episode_num:04d}_confidence.png"
            )
            
            # Feature importance
            if len(self.explainer.decision_history) > 100:
                self.visualizer.plot_feature_importance_over_time(
                    save_path=viz_dir / f"episode_{episode_num:04d}_features.png"
                )
            
            logger.info(f"Visualizations saved to: {viz_dir}")
        except Exception as e:
            logger.warning(f"Failed to save visualizations: {e}")
    
    def generate_final_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive final report"""
        if save_path is None and self.save_dir:
            save_path = self.save_dir / "final_policy_report.txt"
        
        report = self.explainer.generate_policy_report(save_path=save_path)
        
        # Add training summary
        summary = f"\n\n{'='*80}\n"
        summary += "TRAINING SUMMARY\n"
        summary += f"{'='*80}\n"
        summary += f"Total Episodes: {self.episode_counter}\n"
        summary += f"Total Steps: {self.step_counter}\n"
        summary += f"Decisions Analyzed: {len(self.explainer.decision_history)}\n"
        summary += f"{'='*80}\n"
        
        full_report = summary + report
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            logger.info(f"Final report saved to: {save_path}")
        
        return full_report


def create_explainable_training_loop(
    agent,
    env,
    n_episodes: int,
    state_feature_names: List[str],
    explain: bool = False,
    explain_frequency: int = 100,
    verbose: bool = False,
    save_dir: Optional[str] = None,
    ml_predictor = None
) -> Tuple:
    """
    Training loop with explainability
    
    Args:
        agent: DQN agent
        env: Trading environment
        n_episodes: Number of episodes
        state_feature_names: Names of state features
        explain: Enable explainability
        explain_frequency: How often to print explanations
        verbose: Print every decision
        save_dir: Where to save explanations
        ml_predictor: ML predictor for context
        
    Returns:
        episode_results: List of episode results
        explainer: ExplainableRL object (or None)
    """
    # Create explainer if requested
    explainer = None
    if explain:
        explainer = ExplainableRL(
            agent=agent,
            state_feature_names=state_feature_names,
            explain_frequency=explain_frequency,
            verbose=verbose,
            save_dir=save_dir
        )
        logger.info(" Explainability enabled")
    
    episode_results = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get context for explanation
            context = {
                'price': env.current_price if hasattr(env, 'current_price') else 0,
                'position': env.position if hasattr(env, 'position') else 0,
                'balance': env.balance if hasattr(env, 'balance') else 0,
            }
            
            # Add ML prediction if available
            if ml_predictor and hasattr(env, 'current_step'):
                try:
                    # Get features for current step
                    # This would need to be adapted based on your env structure
                    context['ml_prediction'] = np.array([0.33, 0.34, 0.33])  # Placeholder
                except:
                    pass
            
            # Act with or without explanation
            if explainer:
                action, explanation = explainer.act_with_explanation(
                    state, context=context, training=True
                )
            else:
                action = agent.act(state, training=True)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Remember and train
            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > 1000:
                agent.replay()
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if steps >= len(env.data) - 100:
                break
        
        # Episode summary
        episode_results.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': steps,
            'epsilon': agent.epsilon
        })
        
        if explainer:
            explainer.episode_summary(episode, episode_reward, steps)
            
            # Save periodic reports
            if (episode + 1) % 10 == 0:
                explainer.save_episode_report(episode)
                explainer.save_episode_visualizations(episode)
        
        # Regular logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean([r['reward'] for r in episode_results[-10:]])
            logger.info(f"Episode {episode+1}/{n_episodes} - Avg Reward (last 10): {avg_reward:.2f}")
        
        # Update target network
        if (episode + 1) % 10 == 0:
            agent.update_target_network()
    
    # Final report
    if explainer:
        explainer.generate_final_report()
    
    return episode_results, explainer


if __name__ == "__main__":
    # Test
    print("Explainability integration module loaded successfully")