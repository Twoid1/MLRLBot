"""
Feature Selection Module
Advanced feature selection methods for trading bot
Reduces features from 100+ to top 50 most predictive
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, mutual_info_regression,
    RFE, RFECV, VarianceThreshold, chi2
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import joblib
from dataclasses import dataclass
import json


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results"""
    selected_features: List[str]
    feature_scores: pd.Series
    method: str
    n_features: int
    performance_score: float
    metadata: Dict[str, Any]


class FeatureSelector:
    """
    Advanced feature selection for trading systems
    Combines multiple selection methods to find optimal features
    """
    
    def __init__(self, 
                 n_features: int = 50,
                 task_type: str = 'classification',
                 random_state: int = 42):
        """
        Initialize Feature Selector
        
        Args:
            n_features: Target number of features to select
            task_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.task_type = task_type
        self.random_state = random_state
        self.selection_results = {}
        self.final_features = None
        self.feature_stability = None
        
    # ================== MAIN SELECTION PIPELINE ==================
    
    def select_features(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        methods: Optional[List[str]] = None) -> Dict[str, FeatureSelectionResult]:
        """
        Run multiple feature selection methods and combine results
        
        Args:
            X: Feature DataFrame
            y: Target variable
            methods: List of methods to use (default: all)
            
        Returns:
            Dictionary of selection results by method
        """
        if methods is None:
            methods = ['variance', 'correlation', 'mutual_info', 'random_forest', 
                      'lasso', 'recursive', 'boruta', 'stability']
        
        results = {}
        
        # Clean data
        X_clean, y_clean = self._clean_data(X, y)
        
        # Run each method
        for method in methods:
            print(f"Running {method} selection...")
            
            if method == 'variance':
                results[method] = self.variance_threshold_selection(X_clean, y_clean)
            elif method == 'correlation':
                results[method] = self.correlation_selection(X_clean, y_clean)
            elif method == 'mutual_info':
                results[method] = self.mutual_info_selection(X_clean, y_clean)
            elif method == 'random_forest':
                results[method] = self.random_forest_selection(X_clean, y_clean)
            elif method == 'lasso':
                results[method] = self.lasso_selection(X_clean, y_clean)
            elif method == 'recursive':
                results[method] = self.recursive_elimination(X_clean, y_clean)
            elif method == 'boruta':
                results[method] = self.boruta_selection(X_clean, y_clean)
            elif method == 'stability':
                results[method] = self.stability_selection(X_clean, y_clean)
        
        # Combine results
        self.selection_results = results
        self.final_features = self.combine_selections(results)
        
        return results
    
    # ================== VARIANCE-BASED SELECTION ==================
    
    def variance_threshold_selection(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    threshold: float = 0.01) -> FeatureSelectionResult:
        """
        Remove low-variance features
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Calculate variance scores
        variances = pd.Series(selector.variances_, index=X.columns)
        
        # If too many features, select top by variance
        if len(selected_features) > self.n_features:
            top_features = variances.nlargest(self.n_features).index.tolist()
            selected_features = top_features
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=variances,
            method='variance_threshold',
            n_features=len(selected_features),
            performance_score=np.mean(variances[selected_features]),
            metadata={'threshold': threshold}
        )
    
    # ================== CORRELATION-BASED SELECTION ==================
    
    def correlation_selection(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             correlation_threshold: float = 0.95) -> FeatureSelectionResult:
        """
        Remove highly correlated features and select by target correlation
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated feature pairs
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        for column in corr_matrix.columns:
            if column in to_drop:
                continue
            
            # Find features correlated with this one
            correlated = upper_tri[column][upper_tri[column] > correlation_threshold].index.tolist()
            
            if correlated:
                # Keep the one with higher correlation to target
                target_corrs = {column: abs(X[column].corr(y))}
                for feat in correlated:
                    target_corrs[feat] = abs(X[feat].corr(y))
                
                # Keep best, drop others
                best_feature = max(target_corrs, key=target_corrs.get)
                for feat in target_corrs:
                    if feat != best_feature:
                        to_drop.add(feat)
        
        # Remove highly correlated features
        selected_features = [f for f in X.columns if f not in to_drop]
        
        # Calculate correlation with target
        target_correlations = pd.Series(
            {col: abs(X[col].corr(y)) for col in selected_features}
        ).sort_values(ascending=False)
        
        # Select top N by target correlation
        if len(selected_features) > self.n_features:
            selected_features = target_correlations.head(self.n_features).index.tolist()
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=target_correlations,
            method='correlation',
            n_features=len(selected_features),
            performance_score=np.mean(target_correlations[selected_features]),
            metadata={'correlation_threshold': correlation_threshold}
        )
    
    # ================== MUTUAL INFORMATION SELECTION ==================
    
    def mutual_info_selection(self, 
                             X: pd.DataFrame, 
                             y: pd.Series) -> FeatureSelectionResult:
        """
        Select features using mutual information
        """
        if self.task_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Create scores series
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Select top features
        selected_features = mi_scores.head(self.n_features).index.tolist()
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=mi_scores,
            method='mutual_information',
            n_features=len(selected_features),
            performance_score=np.mean(mi_scores[selected_features]),
            metadata={'task_type': self.task_type}
        )
    
    # ================== RANDOM FOREST IMPORTANCE ==================
    
    def random_forest_selection(self, 
                               X: pd.DataFrame, 
                               y: pd.Series,
                               n_estimators: int = 100) -> FeatureSelectionResult:
        """
        Select features using Random Forest feature importance
        """
        if self.task_type == 'classification':
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Select top features
        selected_features = importance.head(self.n_features).index.tolist()
        
        # Calculate OOB score if available
        performance_score = np.mean(importance[selected_features])
        if hasattr(rf, 'oob_score_'):
            performance_score = rf.oob_score_
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=importance,
            method='random_forest',
            n_features=len(selected_features),
            performance_score=performance_score,
            metadata={'n_estimators': n_estimators}
        )
    
    # ================== LASSO SELECTION ==================
    
    def lasso_selection(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       cv: int = 5) -> FeatureSelectionResult:
        """
        Select features using Lasso regularization
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Use LassoCV to find optimal alpha
        lasso = LassoCV(cv=cv, random_state=self.random_state, n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # Get coefficients
        coefficients = pd.Series(np.abs(lasso.coef_), index=X.columns).sort_values(ascending=False)
        
        # Select non-zero coefficients
        selected_features = coefficients[coefficients > 0].index.tolist()
        
        # If too many, select top N
        if len(selected_features) > self.n_features:
            selected_features = coefficients.head(self.n_features).index.tolist()
        # If too few, add more based on coefficient magnitude
        elif len(selected_features) < self.n_features:
            selected_features = coefficients.head(self.n_features).index.tolist()
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=coefficients,
            method='lasso',
            n_features=len(selected_features),
            performance_score=lasso.score(X_scaled, y),
            metadata={'alpha': lasso.alpha_, 'cv': cv}
        )
    
    # ================== RECURSIVE FEATURE ELIMINATION ==================
    
    def recursive_elimination(self, 
                             X: pd.DataFrame, 
                             y: pd.Series,
                             step: int = 1) -> FeatureSelectionResult:
        """
        Recursive Feature Elimination with Cross-Validation
        """
        # Use appropriate estimator
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            estimator = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # RFE
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features,
            step=step
        )
        rfe.fit(X, y)
        
        # Get selected features
        selected_features = X.columns[rfe.support_].tolist()
        
        # Get ranking
        ranking = pd.Series(rfe.ranking_, index=X.columns)
        # Convert ranking to scores (inverse ranking)
        scores = pd.Series(1.0 / ranking, index=X.columns).sort_values(ascending=False)
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=scores,
            method='recursive_elimination',
            n_features=len(selected_features),
            performance_score=np.mean(scores[selected_features]),
            metadata={'step': step}
        )
    
    # ================== BORUTA-LIKE SELECTION ==================
    
    def boruta_selection(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        n_iterations: int = 100) -> FeatureSelectionResult:
        """
        Boruta-inspired feature selection
        Compare feature importance with shuffled features
        """
        # Create shadow features (shuffled versions)
        X_shadow = X.apply(np.random.permutation)
        X_shadow.columns = [f'shadow_{col}' for col in X_shadow.columns]
        
        # Combine original and shadow
        X_combined = pd.concat([X, X_shadow], axis=1)
        
        # Train Random Forest
        if self.task_type == 'classification':
            rf = RandomForestClassifier(
                n_estimators=n_iterations,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=n_iterations,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        rf.fit(X_combined, y)
        
        # Get importance
        importance = pd.Series(rf.feature_importances_, index=X_combined.columns)
        
        # Get max shadow importance
        shadow_importance = importance[[col for col in importance.index if 'shadow_' in col]]
        max_shadow = shadow_importance.max()
        
        # Select features better than best shadow
        original_importance = importance[[col for col in X.columns]]
        selected_mask = original_importance > max_shadow
        selected_features = original_importance[selected_mask].sort_values(ascending=False).index.tolist()
        
        # Adjust to target number
        if len(selected_features) > self.n_features:
            selected_features = original_importance.sort_values(ascending=False).head(self.n_features).index.tolist()
        elif len(selected_features) < self.n_features:
            selected_features = original_importance.sort_values(ascending=False).head(self.n_features).index.tolist()
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=original_importance,
            method='boruta',
            n_features=len(selected_features),
            performance_score=np.mean(original_importance[selected_features]),
            metadata={'n_iterations': n_iterations, 'max_shadow_importance': max_shadow}
        )
    
    # ================== STABILITY SELECTION ==================
    
    def stability_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           n_bootstrap: int = 50,
                           sample_fraction: float = 0.8) -> FeatureSelectionResult:
        """
        Stability selection using bootstrap sampling
        """
        n_samples = len(X)
        n_sample = int(n_samples * sample_fraction)
        
        # Store selection counts
        selection_counts = pd.Series(0, index=X.columns)
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_sample, replace=False)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            
            # Run feature selection on sample
            if self.task_type == 'classification':
                selector = SelectKBest(f_classif, k=min(self.n_features, len(X.columns)))
            else:
                selector = SelectKBest(mutual_info_regression, k=min(self.n_features, len(X.columns)))
            
            selector.fit(X_sample, y_sample)
            selected = X.columns[selector.get_support()]
            
            # Update counts
            selection_counts[selected] += 1
        
        # Calculate stability scores
        stability_scores = selection_counts / n_bootstrap
        stability_scores = stability_scores.sort_values(ascending=False)
        
        # Select stable features
        selected_features = stability_scores.head(self.n_features).index.tolist()
        
        # Store stability information
        self.feature_stability = stability_scores
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores=stability_scores,
            method='stability',
            n_features=len(selected_features),
            performance_score=np.mean(stability_scores[selected_features]),
            metadata={'n_bootstrap': n_bootstrap, 'sample_fraction': sample_fraction}
        )
    
    # ================== ENSEMBLE SELECTION ==================
    
    def combine_selections(self, 
                          results: Dict[str, FeatureSelectionResult],
                          weights: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Combine multiple selection methods using voting or weighted scoring
        
        Args:
            results: Dictionary of selection results
            weights: Optional weights for each method
            
        Returns:
            Final list of selected features
        """
        if not results:
            return []
        
        # Default weights
        if weights is None:
            weights = {
                'variance': 0.5,
                'correlation': 0.8,
                'mutual_info': 1.0,
                'random_forest': 1.2,
                'lasso': 0.9,
                'recursive': 1.0,
                'boruta': 1.1,
                'stability': 1.3
            }
        
        # Count votes for each feature
        feature_votes = pd.Series(0.0, index=self._get_all_features(results))
        
        for method, result in results.items():
            weight = weights.get(method, 1.0)
            
            # Add weighted votes
            for i, feature in enumerate(result.selected_features):
                # Give higher weight to top-ranked features
                rank_weight = 1.0 - (i / len(result.selected_features)) * 0.5
                feature_votes[feature] += weight * rank_weight
        
        # Sort by votes and select top N
        feature_votes = feature_votes.sort_values(ascending=False)
        final_features = feature_votes.head(self.n_features).index.tolist()
        
        print(f"\nFinal selection: {len(final_features)} features")
        print(f"Top 10 features by ensemble vote:")
        for feat in final_features[:10]:
            print(f"  - {feat}: {feature_votes[feat]:.3f}")
        
        return final_features
    
    # ================== MULTI-ASSET SELECTION ==================
    
    def multi_asset_selection(self,
                             asset_data: Dict[str, pd.DataFrame],
                             asset_targets: Dict[str, pd.Series]) -> List[str]:
        """
        Select features that work well across multiple assets
        
        Args:
            asset_data: Dictionary {asset_name: feature_df}
            asset_targets: Dictionary {asset_name: target_series}
            
        Returns:
            Features that perform well across all assets
        """
        asset_scores = {}
        
        for asset, X in asset_data.items():
            y = asset_targets[asset]
            
            # Run selection for this asset
            print(f"\nSelecting features for {asset}...")
            results = self.select_features(X, y, methods=['mutual_info', 'random_forest'])
            
            # Store scores
            for method, result in results.items():
                for feat in result.selected_features:
                    key = f"{asset}_{feat}"
                    if feat not in asset_scores:
                        asset_scores[feat] = []
                    asset_scores[feat].append(result.feature_scores.get(feat, 0))
        
        # Calculate average performance across assets
        avg_scores = {}
        for feat, scores in asset_scores.items():
            if len(scores) >= len(asset_data) * 0.5:  # Feature must work for at least 50% of assets
                avg_scores[feat] = np.mean(scores)
        
        # Sort and select
        avg_scores = pd.Series(avg_scores).sort_values(ascending=False)
        universal_features = avg_scores.head(self.n_features).index.tolist()
        
        print(f"\nUniversal features selected: {len(universal_features)}")
        
        return universal_features
    
    # ================== PERFORMANCE EVALUATION ==================
    
    def evaluate_features(self,
                         X: pd.DataFrame,
                         y: pd.Series,
                         features: List[str],
                         cv: int = 5) -> float:
        """
        Evaluate feature set performance using cross-validation
        
        Args:
            X: Feature DataFrame
            y: Target variable
            features: List of features to evaluate
            cv: Number of CV folds
            
        Returns:
            Cross-validation score
        """
        X_selected = X[features]
        
        # Choose estimator
        if self.task_type == 'classification':
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'roc_auc' if len(np.unique(y)) == 2 else 'accuracy'
        else:
            estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            scoring = 'r2'
        
        # Time series split for financial data
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Calculate scores
        scores = cross_val_score(estimator, X_selected, y, cv=tscv, scoring=scoring, n_jobs=-1)
        
        return np.mean(scores)
    
    # ================== HELPER METHODS ==================
    
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data by removing NaN and infinite values"""
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Find rows with any NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        
        return X[valid_mask], y[valid_mask]
    
    def _get_all_features(self, results: Dict[str, FeatureSelectionResult]) -> List[str]:
        """Get union of all selected features"""
        all_features = set()
        for result in results.values():
            all_features.update(result.selected_features)
        return list(all_features)
    
    # ================== SAVE/LOAD METHODS ==================
    
    def save_selection(self, filepath: str) -> None:
        """
        Save feature selection results
        
        Args:
            filepath: Path to save results
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'final_features': self.final_features,
            'n_features': self.n_features,
            'task_type': self.task_type,
            'selection_results': {}
        }
        
        # Convert results to serializable format
        for method, result in self.selection_results.items():
            save_data['selection_results'][method] = {
                'selected_features': result.selected_features,
                'feature_scores': result.feature_scores.to_dict(),
                'performance_score': result.performance_score,
                'metadata': result.metadata
            }
        
        # Save feature stability if available
        if self.feature_stability is not None:
            save_data['feature_stability'] = self.feature_stability.to_dict()
        
        # Save as JSON
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Feature selection saved to {path}")
    
    def load_selection(self, filepath: str) -> None:
        """
        Load feature selection results
        
        Args:
            filepath: Path to load results from
        """
        path = Path(filepath)
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        # Restore attributes
        self.final_features = save_data['final_features']
        self.n_features = save_data['n_features']
        self.task_type = save_data['task_type']
        
        # Restore feature stability
        if 'feature_stability' in save_data:
            self.feature_stability = pd.Series(save_data['feature_stability'])
        
        print(f"Feature selection loaded from {path}")
        print(f"Selected {len(self.final_features)} features")
    
    def get_feature_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive feature selection report
        
        Returns:
            DataFrame with feature selection statistics
        """
        if not self.selection_results:
            return pd.DataFrame()
        
        # Collect all features
        all_features = self._get_all_features(self.selection_results)
        
        # Create report DataFrame
        report = pd.DataFrame(index=all_features)
        
        # Add selection counts and scores for each method
        for method, result in self.selection_results.items():
            # Selection indicator
            report[f'{method}_selected'] = [
                1 if feat in result.selected_features else 0 
                for feat in all_features
            ]
            
            # Scores
            report[f'{method}_score'] = [
                result.feature_scores.get(feat, 0) 
                for feat in all_features
            ]
        
        # Add summary statistics
        selection_cols = [col for col in report.columns if '_selected' in col]
        report['total_selections'] = report[selection_cols].sum(axis=1)
        report['selection_rate'] = report['total_selections'] / len(selection_cols)
        
        # Add stability score if available
        if self.feature_stability is not None:
            report['stability_score'] = [
                self.feature_stability.get(feat, 0) 
                for feat in all_features
            ]
        
        # Add final selection indicator
        if self.final_features:
            report['final_selected'] = [
                1 if feat in self.final_features else 0 
                for feat in all_features
            ]
        
        # Sort by selection rate
        report = report.sort_values('selection_rate', ascending=False)
        
        return report