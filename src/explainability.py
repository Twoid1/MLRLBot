"""
RL Agent Explainability System
Shows WHY the agent makes decisions and WHAT it learned
"""

import numpy as np
import pandas as pd
from collections import defaultdict, deque
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

class AgentExplainer:
    """
    Explains RL agent decisions in human-readable format
    Tracks feature importance, decision patterns, and policy behavior
    """
    
    def __init__(self, state_features: List[str], action_names: List[str]):
        """
        Args:
            state_features: Names of all state dimensions
            action_names: Names of actions (e.g., ['Hold', 'Buy', 'Sell'])
        """
        self.state_features = state_features
        self.action_names = action_names
        self.n_actions = len(action_names)
        
        # Decision history
        self.decision_history = []
        
        # Feature importance tracking
        self.feature_importance = defaultdict(lambda: {'count': 0, 'sum_impact': 0})
        self.recent_decisions = deque(maxlen=1000)
        
        # Policy patterns
        self.policy_rules = []
        self.state_action_pairs = defaultdict(int)
        
        # Performance tracking per decision type
        self.decision_outcomes = defaultdict(list)
        
    def explain_decision(self, 
                        state: np.ndarray,
                        action: int,
                        q_values: np.ndarray,
                        agent,
                        context: Optional[Dict] = None) -> Dict:
        """
        Explain a single decision in detail
        
        Returns comprehensive explanation including:
        - Top influential features
        - Q-values for all actions
        - Why this action was chosen
        - Alternative actions considered
        - Current market context
        """
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'action_taken': self.action_names[action],
            'action_index': action,
            'confidence': float(q_values[action]),
            'q_values': {self.action_names[i]: float(q_values[i]) 
                        for i in range(len(q_values))},
            'state_summary': self._summarize_state(state),
            'feature_attribution': self._calculate_feature_attribution(state, action, agent),
            'alternative_actions': self._analyze_alternatives(q_values, action),
            'decision_reasoning': self._generate_reasoning(state, action, q_values, context),
            'risk_assessment': self._assess_risk(state, action, q_values),
        }
        
        # Store for pattern analysis
        self.decision_history.append(explanation)
        self.recent_decisions.append((state, action, q_values))
        
        return explanation
    
    def _calculate_feature_attribution(self, 
                                      state: np.ndarray, 
                                      action: int,
                                      agent) -> Dict:
        """
        Calculate which state features most influenced this decision
        Uses gradient-based attribution
        """
        attributions = {}
        
        try:
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # â­ Auto-detect device and move tensor
            device = next(agent.q_network.parameters()).device
            state_tensor = state_tensor.to(device)
            state_tensor.requires_grad = True

            q_values = agent.q_network(state_tensor)
            
            # Get gradient for chosen action
            q_values[0, action].backward()
            
            # Attribution = gradient * input
            attribution = (state_tensor.grad[0] * state_tensor[0]).abs().detach().cpu().numpy()
            
            # Normalize
            if attribution.sum() > 0:
                attribution = attribution / attribution.sum()
            
            # Get top features
            top_indices = np.argsort(attribution)[-10:][::-1]
            
            for idx in top_indices:
                if idx < len(self.state_features):
                    attributions[self.state_features[idx]] = {
                        'importance': float(attribution[idx]),
                        'value': float(state[idx]),
                        'rank': int(np.where(np.argsort(attribution)[::-1] == idx)[0][0] + 1)
                    }
                    
                    # Update running importance
                    self.feature_importance[self.state_features[idx]]['count'] += 1
                    self.feature_importance[self.state_features[idx]]['sum_impact'] += float(attribution[idx])
                    
        except Exception as e:
            print(f"Attribution calculation failed: {e}")
            # Fallback: use simple variance-based importance
            attributions = self._fallback_attribution(state)
        
        return attributions
    
    def _fallback_attribution(self, state: np.ndarray) -> Dict:
        """Simple importance based on state magnitude"""
        abs_state = np.abs(state)
        if abs_state.sum() > 0:
            normalized = abs_state / abs_state.sum()
        else:
            normalized = np.ones_like(state) / len(state)
            
        top_indices = np.argsort(normalized)[-10:][::-1]
        
        attributions = {}
        for idx in top_indices:
            if idx < len(self.state_features):
                attributions[self.state_features[idx]] = {
                    'importance': float(normalized[idx]),
                    'value': float(state[idx]),
                    'rank': int(np.where(np.argsort(normalized)[::-1] == idx)[0][0] + 1)
                }
        
        return attributions
    
    def _summarize_state(self, state: np.ndarray) -> Dict:
        """Create human-readable summary of current state"""
        summary = {
            'state_vector_length': len(state),
            'non_zero_features': int(np.count_nonzero(state)),
            'state_magnitude': float(np.linalg.norm(state)),
            'extreme_values': {}
        }
        
        # Find extreme values
        abs_state = np.abs(state)
        if abs_state.max() > 0:
            max_idx = abs_state.argmax()
            min_idx = abs_state.argmin()
            
            if max_idx < len(self.state_features):
                summary['extreme_values']['maximum'] = {
                    'feature': self.state_features[max_idx],
                    'value': float(state[max_idx])
                }
            
            if min_idx < len(self.state_features):
                summary['extreme_values']['minimum'] = {
                    'feature': self.state_features[min_idx],
                    'value': float(state[min_idx])
                }
        
        return summary
    
    def _analyze_alternatives(self, q_values: np.ndarray, chosen_action: int) -> Dict:
        """Analyze actions that were NOT taken"""
        alternatives = {}
        
        sorted_indices = np.argsort(q_values)[::-1]
        
        for rank, action_idx in enumerate(sorted_indices):
            if action_idx != chosen_action:
                alternatives[self.action_names[action_idx]] = {
                    'q_value': float(q_values[action_idx]),
                    'rank': rank + 1,
                    'q_diff_from_chosen': float(q_values[chosen_action] - q_values[action_idx]),
                    'percentage_of_chosen': float((q_values[action_idx] / q_values[chosen_action] * 100) 
                                                 if q_values[chosen_action] != 0 else 0)
                }
        
        return alternatives
    
    def _generate_reasoning(self, 
                           state: np.ndarray, 
                           action: int,
                           q_values: np.ndarray,
                           context: Optional[Dict]) -> str:
        """
        Generate natural language explanation of the decision
        """
        reasoning_parts = []
        
        # Action decision
        action_name = self.action_names[action]
        confidence = q_values[action]
        
        q_diff = q_values.max() - q_values.min()
        if q_diff > 1.0:
            confidence_level = "very confident"
        elif q_diff > 0.5:
            confidence_level = "confident"
        elif q_diff > 0.1:
            confidence_level = "somewhat confident"
        else:
            confidence_level = "uncertain"
        
        reasoning_parts.append(
            f"Agent chose to {action_name} with {confidence_level} "
            f"(Q-value: {confidence:.3f}, Q-spread: {q_diff:.3f})"
        )
        
        # Feature-based reasoning
        if len(self.recent_decisions) > 10:
            recent_states = np.array([s for s, a, q in self.recent_decisions])
            current_deviation = np.abs(state - recent_states.mean(axis=0))
            unusual_features = np.where(current_deviation > recent_states.std(axis=0) * 2)[0]
            
            if len(unusual_features) > 0 and unusual_features[0] < len(self.state_features):
                top_unusual = unusual_features[0]
                reasoning_parts.append(
                    f"Notable: {self.state_features[top_unusual]} is unusual "
                    f"(value: {state[top_unusual]:.3f}, "
                    f"deviation: {current_deviation[top_unusual]:.3f})"
                )
        
        # Context-based reasoning
        if context:
            if 'price' in context and 'ml_prediction' in context:
                ml_pred = context['ml_prediction']
                pred_labels = ['DOWN', 'FLAT', 'UP']
                pred_action = pred_labels[np.argmax(ml_pred)]
                
                reasoning_parts.append(
                    f"ML Predictor suggests: {pred_action} "
                    f"(confidence: {ml_pred.max():.2%})"
                )
                
                # Check if RL agrees with ML
                rl_direction = ['DOWN', 'FLAT', 'UP'][action] if action < 3 else 'HOLD'
                if pred_action != rl_direction and action != 0:  # If not holding
                    reasoning_parts.append(
                        f"ï¸ RL agent disagreed with ML predictor "
                        f"(RL: {rl_direction}, ML: {pred_action})"
                    )
            
            if 'position' in context:
                if context['position'] > 0:
                    reasoning_parts.append(f"Currently holding position: {context['position']:.4f} units")
                else:
                    reasoning_parts.append("Currently flat (no position)")
        
        return " | ".join(reasoning_parts)
    
    def _assess_risk(self, 
                     state: np.ndarray, 
                     action: int, 
                     q_values: np.ndarray) -> Dict:
        """Assess risk of the decision"""
        risk_assessment = {
            'confidence_score': float(q_values[action] - q_values.mean()),
            'uncertainty': float(q_values.std()),
            'risk_level': 'unknown'
        }
        
        # Determine risk level
        if risk_assessment['uncertainty'] > 1.0:
            risk_assessment['risk_level'] = 'HIGH'
            risk_assessment['warning'] = "High uncertainty in Q-values"
        elif risk_assessment['uncertainty'] > 0.5:
            risk_assessment['risk_level'] = 'MEDIUM'
        else:
            risk_assessment['risk_level'] = 'LOW'
        
        # Check for conflicting signals
        if len(q_values) > 1:
            second_best = np.partition(q_values, -2)[-2]
            if abs(q_values[action] - second_best) < 0.1:
                risk_assessment['warning'] = "Very close decision - consider waiting"
                risk_assessment['risk_level'] = 'MEDIUM'
        
        return risk_assessment
    
    def analyze_policy_patterns(self, lookback: int = 1000) -> Dict:
        """
        Analyze patterns in recent decisions to understand the learned policy
        """
        if len(self.decision_history) < 10:
            return {'status': 'Insufficient data for pattern analysis'}
        
        recent = self.decision_history[-lookback:]
        
        patterns = {
            'action_distribution': self._get_action_distribution(recent),
            'feature_importance_ranking': self._get_feature_importance(),
            'decision_confidence': self._analyze_confidence(recent),
            'policy_rules': self._extract_policy_rules(recent),
            'common_patterns': self._find_common_patterns(recent),
            'risk_behavior': self._analyze_risk_behavior(recent)
        }
        
        return patterns
    
    def _get_action_distribution(self, decisions: List[Dict]) -> Dict:
        """Get distribution of actions taken"""
        action_counts = defaultdict(int)
        action_avg_confidence = defaultdict(list)
        
        for dec in decisions:
            action = dec['action_taken']
            action_counts[action] += 1
            action_avg_confidence[action].append(dec['confidence'])
        
        total = len(decisions)
        distribution = {}
        
        for action in self.action_names:
            count = action_counts.get(action, 0)
            distribution[action] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0,
                'avg_confidence': np.mean(action_avg_confidence[action]) if action_avg_confidence[action] else 0
            }
        
        return distribution
    
    def _get_feature_importance(self) -> List[Dict]:
        """Get overall feature importance ranking"""
        importance_list = []
        
        for feature, stats in self.feature_importance.items():
            if stats['count'] > 0:
                avg_impact = stats['sum_impact'] / stats['count']
                importance_list.append({
                    'feature': feature,
                    'avg_importance': avg_impact,
                    'appearance_count': stats['count']
                })
        
        # Sort by average importance
        importance_list.sort(key=lambda x: x['avg_importance'], reverse=True)
        
        return importance_list[:20]  # Top 20 features
    
    def _analyze_confidence(self, decisions: List[Dict]) -> Dict:
        """Analyze decision confidence over time"""
        confidences = [d['confidence'] for d in decisions]
        q_spreads = [max(d['q_values'].values()) - min(d['q_values'].values()) 
                     for d in decisions]
        
        return {
            'avg_confidence': float(np.mean(confidences)),
            'confidence_trend': 'increasing' if confidences[-10:] > confidences[:10] else 'decreasing',
            'avg_q_spread': float(np.mean(q_spreads)),
            'confidence_std': float(np.std(confidences)),
            'low_confidence_decisions': sum(1 for c in confidences if c < 0.1)
        }
    
    def _extract_policy_rules(self, decisions: List[Dict]) -> List[str]:
        """
        Extract human-readable policy rules
        """
        rules = []
        
        # Analyze feature -> action correlations
        if len(decisions) < 50:
            return ["Need more data to extract reliable rules"]
        
        # Group decisions by action
        action_groups = defaultdict(list)
        for dec in decisions:
            action_groups[dec['action_taken']].append(dec)
        
        # For each action, find common feature patterns
        for action, action_decisions in action_groups.items():
            if len(action_decisions) < 10:
                continue
            
            # Aggregate feature attributions
            feature_scores = defaultdict(list)
            for dec in action_decisions:
                for feature, attr in dec['feature_attribution'].items():
                    feature_scores[feature].append(attr['importance'])
            
            # Find consistently important features
            important_features = []
            for feature, scores in feature_scores.items():
                if np.mean(scores) > 0.1:  # Threshold for importance
                    important_features.append({
                        'feature': feature,
                        'avg_importance': np.mean(scores),
                        'consistency': 1 - np.std(scores)  # High consistency = low std
                    })
            
            important_features.sort(key=lambda x: x['avg_importance'], reverse=True)
            
            # Generate rule
            if important_features:
                top_features = important_features[:3]
                rule = f"When {action}: "
                feature_names = [f"{f['feature']} (imp: {f['avg_importance']:.2%})" 
                               for f in top_features]
                rule += "key factors are " + ", ".join(feature_names)
                rules.append(rule)
        
        return rules
    
    def _find_common_patterns(self, decisions: List[Dict]) -> List[str]:
        """Find common decision patterns"""
        patterns = []
        
        # Analyze action sequences
        if len(decisions) < 5:
            return patterns
        
        action_sequence = [d['action_taken'] for d in decisions]
        
        # Find consecutive action patterns
        consecutive_same = 1
        prev_action = action_sequence[0]
        max_consecutive = defaultdict(int)
        
        for action in action_sequence[1:]:
            if action == prev_action:
                consecutive_same += 1
            else:
                max_consecutive[prev_action] = max(max_consecutive[prev_action], consecutive_same)
                consecutive_same = 1
                prev_action = action
        
        for action, count in max_consecutive.items():
            if count > 3:
                patterns.append(f"Tends to stay in {action} for up to {count} consecutive steps")
        
        # Analyze action transitions
        transitions = defaultdict(int)
        for i in range(len(action_sequence) - 1):
            transition = f"{action_sequence[i]} -> {action_sequence[i+1]}"
            transitions[transition] += 1
        
        # Find most common transitions
        if transitions:
            most_common = max(transitions.items(), key=lambda x: x[1])
            if most_common[1] > 10:
                patterns.append(f"Most common transition: {most_common[0]} ({most_common[1]} times)")
        
        return patterns
    
    def _analyze_risk_behavior(self, decisions: List[Dict]) -> Dict:
        """Analyze risk-taking behavior"""
        risk_levels = [d['risk_assessment']['risk_level'] for d in decisions]
        
        risk_distribution = {
            'high_risk': risk_levels.count('HIGH'),
            'medium_risk': risk_levels.count('MEDIUM'),
            'low_risk': risk_levels.count('LOW')
        }
        
        # Check for risk warnings
        warnings = [d['risk_assessment'].get('warning', '') for d in decisions]
        warning_count = sum(1 for w in warnings if w)
        
        return {
            'risk_distribution': risk_distribution,
            'warning_count': warning_count,
            'risk_profile': 'aggressive' if risk_distribution['high_risk'] > len(decisions) * 0.3 
                           else 'conservative' if risk_distribution['low_risk'] > len(decisions) * 0.7
                           else 'balanced'
        }
    
    def generate_policy_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive human-readable report of the learned policy
        """
        if len(self.decision_history) < 50:
            return "Insufficient data for comprehensive report (need at least 50 decisions)"
        
        patterns = self.analyze_policy_patterns()
        
        report = []
        report.append("=" * 80)
        report.append("RL AGENT POLICY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Decisions Analyzed: {len(self.decision_history)}")
        report.append("")
        
        # Action Distribution
        report.append("1. ACTION DISTRIBUTION")
        report.append("-" * 80)
        for action, stats in patterns['action_distribution'].items():
            report.append(f"  {action:10s}: {stats['count']:4d} times ({stats['percentage']:5.1f}%) "
                         f"- Avg Confidence: {stats['avg_confidence']:.3f}")
        report.append("")
        
        # Feature Importance
        report.append("2. TOP INFLUENTIAL FEATURES")
        report.append("-" * 80)
        for i, feat in enumerate(patterns['feature_importance_ranking'][:10], 1):
            report.append(f"  {i:2d}. {feat['feature']:40s} - Importance: {feat['avg_importance']:.3f} "
                         f"(appeared {feat['appearance_count']} times)")
        report.append("")
        
        # Policy Rules
        report.append("3. LEARNED POLICY RULES")
        report.append("-" * 80)
        for rule in patterns['policy_rules']:
            report.append(f"  â€¢ {rule}")
        report.append("")
        
        # Decision Confidence
        report.append("4. DECISION CONFIDENCE ANALYSIS")
        report.append("-" * 80)
        conf_stats = patterns['decision_confidence']
        report.append(f"  Average Confidence: {conf_stats['avg_confidence']:.3f}")
        report.append(f"  Confidence Std Dev: {conf_stats['confidence_std']:.3f}")
        report.append(f"  Average Q-Spread: {conf_stats['avg_q_spread']:.3f}")
        report.append(f"  Low Confidence Decisions: {conf_stats['low_confidence_decisions']}")
        report.append(f"  Confidence Trend: {conf_stats['confidence_trend']}")
        report.append("")
        
        # Common Patterns
        report.append("5. BEHAVIORAL PATTERNS")
        report.append("-" * 80)
        for pattern in patterns['common_patterns']:
            report.append(f"  â€¢ {pattern}")
        report.append("")
        
        # Risk Behavior
        report.append("6. RISK BEHAVIOR")
        report.append("-" * 80)
        risk_stats = patterns['risk_behavior']
        report.append(f"  Risk Profile: {risk_stats['risk_profile'].upper()}")
        report.append(f"  High Risk Decisions: {risk_stats['risk_distribution']['high_risk']}")
        report.append(f"  Medium Risk Decisions: {risk_stats['risk_distribution']['medium_risk']}")
        report.append(f"  Low Risk Decisions: {risk_stats['risk_distribution']['low_risk']}")
        report.append(f"  Total Warnings: {risk_stats['warning_count']}")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {save_path}")
        
        return report_text
    
    def explain_live_decision(self, 
                            state: np.ndarray, 
                            action: int, 
                            q_values: np.ndarray,
                            agent,
                            context: Dict,
                            verbose: bool = True) -> Dict:
        """
        Real-time explanation during live trading
        """
        explanation = self.explain_decision(state, action, q_values, agent, context)
        
        if verbose:
            self._print_live_explanation(explanation, context)
        
        return explanation
    
    def _print_live_explanation(self, explanation: Dict, context: Dict):
        """Pretty print explanation for live monitoring"""
        print("\n" + "="*80)
        print(f" {explanation['timestamp']}")
        print("="*80)
        
        print(f"\n ACTION: {explanation['action_taken'].upper()}")
        print(f"   Confidence: {explanation['confidence']:.3f}")
        print(f"   Risk Level: {explanation['risk_assessment']['risk_level']}")
        
        print(f"\n Q-VALUES:")
        for action, q_val in explanation['q_values'].items():
            marker = "p" if action == explanation['action_taken'] else "  "
            print(f"   {marker} {action:10s}: {q_val:.3f}")
        
        print(f"\n TOP INFLUENTIAL FEATURES:")
        for i, (feature, attr) in enumerate(list(explanation['feature_attribution'].items())[:5], 1):
            print(f"   {i}. {feature:40s} - Importance: {attr['importance']:.3f}, Value: {attr['value']:.3f}")
        
        print(f"\n REASONING:")
        print(f"   {explanation['decision_reasoning']}")
        
        if 'warning' in explanation['risk_assessment']:
            print(f"\n  WARNING: {explanation['risk_assessment']['warning']}")
        
        print("="*80 + "\n")
    
    def save_decision_history(self, filepath: str):
        """Save all decisions to file"""
        # Custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(self.decision_history, f, indent=2, cls=NumpyEncoder)
        print(f"Decision history saved to: {filepath}")
    
    def load_decision_history(self, filepath: str):
        """Load decision history from file"""
        with open(filepath, 'r') as f:
            self.decision_history = json.load(f)
        print(f"Loaded {len(self.decision_history)} decisions from: {filepath}")


class PolicyVisualizer:
    """
    Visualize policy behavior and patterns
    """
    
    def __init__(self, explainer: AgentExplainer):
        self.explainer = explainer
    
    def plot_feature_importance_over_time(self, save_path: Optional[str] = None):
        """Plot how feature importance changes over time"""
        if len(self.explainer.decision_history) < 100:
            print("Need at least 100 decisions for temporal analysis")
            return
        
        # Split into time windows
        window_size = 100
        decisions = self.explainer.decision_history
        
        n_windows = len(decisions) // window_size
        if n_windows < 2:
            print("Need more data for temporal analysis")
            return
        
        # Track top features over time
        feature_evolution = defaultdict(list)
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            window_decisions = decisions[start_idx:end_idx]
            
            # Calculate feature importance in this window
            feature_scores = defaultdict(list)
            for dec in window_decisions:
                for feature, attr in dec['feature_attribution'].items():
                    feature_scores[feature].append(attr['importance'])
            
            # Average importance in this window
            for feature, scores in feature_scores.items():
                feature_evolution[feature].append(np.mean(scores))
        
        # Plot top 10 features
        plt.figure(figsize=(14, 8))
        
        # Get overall top features
        overall_importance = {feat: np.mean(vals) 
                             for feat, vals in feature_evolution.items()}
        top_features = sorted(overall_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        
        for feature, _ in top_features:
            if feature in feature_evolution:
                plt.plot(range(n_windows), feature_evolution[feature], 
                        label=feature, linewidth=2, marker='o')
        
        plt.xlabel('Time Window', fontsize=12)
        plt.ylabel('Average Feature Importance', fontsize=12)
        plt.title('Feature Importance Evolution Over Time', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_action_distribution(self, save_path: Optional[str] = None):
        """Plot distribution of actions taken"""
        patterns = self.explainer.analyze_policy_patterns()
        action_dist = patterns['action_distribution']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        actions = list(action_dist.keys())
        counts = [action_dist[a]['count'] for a in actions]
        colors = plt.cm.Set3(range(len(actions)))
        
        ax1.pie(counts, labels=actions, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Action Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart with confidence
        confidences = [action_dist[a]['avg_confidence'] for a in actions]
        bars = ax2.bar(actions, confidences, color=colors)
        ax2.set_ylabel('Average Confidence', fontsize=12)
        ax2.set_title('Average Confidence per Action', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, max(confidences) * 1.2])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_decision_confidence_over_time(self, save_path: Optional[str] = None):
        """Plot confidence and Q-spread over time"""
        if len(self.explainer.decision_history) < 10:
            print("Need more decisions")
            return
        
        decisions = self.explainer.decision_history
        
        confidences = [d['confidence'] for d in decisions]
        q_spreads = [max(d['q_values'].values()) - min(d['q_values'].values()) 
                    for d in decisions]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Confidence over time
        ax1.plot(confidences, linewidth=2, color='blue', alpha=0.7)
        ax1.axhline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.fill_between(range(len(confidences)), confidences, alpha=0.3)
        ax1.set_ylabel('Confidence (Q-Value)', fontsize=12)
        ax1.set_title('Decision Confidence Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-spread over time
        ax2.plot(q_spreads, linewidth=2, color='green', alpha=0.7)
        ax2.axhline(np.mean(q_spreads), color='red', linestyle='--',
                   label=f'Mean: {np.mean(q_spreads):.3f}')
        ax2.fill_between(range(len(q_spreads)), q_spreads, alpha=0.3)
        ax2.set_xlabel('Decision Number', fontsize=12)
        ax2.set_ylabel('Q-Spread (Max-Min)', fontsize=12)
        ax2.set_title('Q-Value Spread Over Time', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    state_features = [f'feature_{i}' for i in range(63)]
    action_names = ['Hold', 'Buy', 'Sell']
    
    explainer = AgentExplainer(state_features, action_names)
    
    # Simulate some decisions
    for _ in range(100):
        state = np.random.randn(63)
        q_values = np.random.rand(3)
        action = np.argmax(q_values)
        
        # Mock agent with simple Q-network
        class MockAgent:
            def __init__(self):
                self.q_network = None  # Would be real network
        
        agent = MockAgent()
        
        explanation = explainer.explain_decision(state, action, q_values, agent, None)
    
    # Generate report
    report = explainer.generate_policy_report()
    print(report)
    
    # Visualize
    visualizer = PolicyVisualizer(explainer)
    visualizer.plot_action_distribution()
    visualizer.plot_decision_confidence_over_time()