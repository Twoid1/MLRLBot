"""
Machine Learning Predictor Module
Implements XGBoost/LightGBM with triple-barrier labeling for multi-asset trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

# Import labeling classes from labeling module - NO DUPLICATION!
from .labeling import (
    TripleBarrierLabeler,
    FixedTimeLabeler,
    TrendLabeler,
    MetaLabeler,
    SampleWeights,
    LabelingConfig,
    LabelingPipeline
)

# Try to import gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Using LightGBM only.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Using XGBoost only.")

if not HAS_XGBOOST and not HAS_LIGHTGBM:
    raise ImportError("Please install either XGBoost or LightGBM: pip install xgboost lightgbm")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    timestamp: pd.Timestamp
    symbol: str
    timeframe: str
    prediction: int  # -1: down, 0: flat, 1: up
    probabilities: np.ndarray  # [P(down), P(flat), P(up)]
    confidence: float
    features_used: List[str]
    model_version: str


class MLPredictor:
    """
    Machine Learning predictor for multi-asset trading
    Uses XGBoost/LightGBM with labeling methods from labeling.py
    """
    
    def __init__(self,
                 model_type: str = 'xgboost',
                 task_type: str = 'multiclass',
                 n_jobs: int = -1,
                 random_state: int = 42,
                 labeling_config: Optional[LabelingConfig] = None):
        """
        Initialize ML Predictor
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            task_type: 'multiclass' for 3-class prediction
            n_jobs: Number of parallel jobs
            random_state: Random seed
            labeling_config: Configuration for labeling methods
        """
        self.model_type = model_type
        self.task_type = task_type
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Models storage
        self.models = {}  # Store multiple models for ensemble
        self.best_model = None
        self.feature_names = None
        self.selected_features = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize labeling configuration and pipeline
        self.labeling_config = labeling_config or LabelingConfig()
        self.labeling_pipeline = LabelingPipeline(self.labeling_config)
        
        # Individual labelers for flexibility
        self.triple_barrier_labeler = TripleBarrierLabeler()
        self.fixed_time_labeler = FixedTimeLabeler()
        self.trend_labeler = TrendLabeler()
        self.meta_labeler = MetaLabeler()
        self.sample_weights = SampleWeights()
        
        # Training history
        self.training_history = []
        self.validation_scores = []
        
        # Model parameters
        self.best_params = None
        
        logger.info(f"MLPredictor initialized with {model_type}")
    
    def prepare_features(self,
                        df: pd.DataFrame,
                        features_df: pd.DataFrame,
                        symbol: str,
                        timeframe: str) -> pd.DataFrame:
        """
        Prepare features for training/prediction
        
        Args:
            df: OHLCV DataFrame
            features_df: Calculated features
            symbol: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Prepared features DataFrame
        """
        # Combine features
        combined = pd.concat([features_df], axis=1)
        
        # Add symbol encoding
        symbol_map = {
            'BTC/USDT': 1, 'ETH/USDT': 2, 'BNB/USDT': 3,
            'ADA/USDT': 4, 'SOL/USDT': 5, 'XRP/USDT': 6,
            'DOT/USDT': 7, 'DOGE/USDT': 8, 'AVAX/USDT': 9,
            'MATIC/USDT': 10, 'LINK/USDT': 11, 'LTC/USDT': 12
        }
        combined['symbol_encoded'] = symbol_map.get(symbol, 0)
        
        # Add timeframe encoding
        timeframe_map = {
            '5m': 1, '15m': 2, '30m': 3,
            '1h': 4, '4h': 5, '1d': 6, '1w': 7
        }
        combined['timeframe_encoded'] = timeframe_map.get(timeframe, 0)
        
        # Add time-based features
        if isinstance(combined.index, pd.DatetimeIndex):
            combined['hour'] = combined.index.hour
            combined['day_of_week'] = combined.index.dayofweek
            combined['day_of_month'] = combined.index.day
            combined['month'] = combined.index.month
            combined['quarter'] = combined.index.quarter
            combined['year'] = combined.index.year
            
            # Add cyclical encoding for hour and day_of_week
            combined['hour_sin'] = np.sin(2 * np.pi * combined['hour'] / 24)
            combined['hour_cos'] = np.cos(2 * np.pi * combined['hour'] / 24)
            combined['dow_sin'] = np.sin(2 * np.pi * combined['day_of_week'] / 7)
            combined['dow_cos'] = np.cos(2 * np.pi * combined['day_of_week'] / 7)
        
        # Remove inf and fill NaN
        combined = combined.replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(method='ffill').fillna(0)
        
        return combined
    
    def create_labels(self,
                    df: pd.DataFrame,
                    method: str = 'triple_barrier',
                    **kwargs) -> pd.Series:
        """
        Create labels using the labeling pipeline
        
        Args:
            df: OHLCV DataFrame
            method: Labeling method ('triple_barrier', 'fixed_time', 'trend', 'simple_returns')
            **kwargs: Additional arguments for labeling
            
        Returns:
            Labels series (already converted to 0, 1, 2 for classification)
        """
        # Use labeling pipeline
        labels = self.labeling_pipeline.create_labels(df, method, **kwargs)
        
        # Convert labels to 0, 1, 2 for classification if needed
        # -1 -> 0 (down), 0 -> 1 (flat), 1 -> 2 (up)
        if labels.min() < 0:
            labels = labels + 1
        
        # Log label distribution
        label_counts = labels.value_counts()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        return labels
    
    def create_labels_with_weights(self,
                                  df: pd.DataFrame,
                                  method: str = 'triple_barrier',
                                  **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Create labels with sample weights
        
        Args:
            df: OHLCV DataFrame
            method: Labeling method
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (labels, weights)
        """
        labels, weights = self.labeling_pipeline.create_labels_with_weights(df, method, **kwargs)
        
        # Convert labels if needed
        if labels.min() < 0:
            labels = labels + 1
        
        return labels, weights
    
    def train(self,
             train_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
             val_data: Optional[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]] = None,
             feature_selection: bool = True,
             n_features: int = 50,
             optimize_params: bool = True,
             use_sample_weights: bool = False) -> Dict[str, Any]:
        """
        Train the ML model on multi-asset data
        
        Args:
            train_data: Dict {symbol: (ohlcv_df, features_df)}
            val_data: Optional validation data in same format
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            optimize_params: Whether to optimize hyperparameters
            use_sample_weights: Whether to use sample weights for training
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Prepare training data
        X_train_list = []
        y_train_list = []
        weights_list = []
        
        for symbol, (ohlcv_df, features_df) in train_data.items():
            # Prepare features
            X = self.prepare_features(ohlcv_df, features_df, symbol, '1h')
            
            # Create labels (with optional weights)
            if use_sample_weights:
                y, weights = self.create_labels_with_weights(ohlcv_df)
                weights_list.append(weights)
            else:
                y = self.create_labels(ohlcv_df)
            
            # Remove NaN rows
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            X_train_list.append(X)
            y_train_list.append(y)
        
        # Combine all data
        X_train = pd.concat(X_train_list, axis=0)
        y_train = pd.concat(y_train_list, axis=0)
        
        # Combine weights if using them
        sample_weights = None
        if use_sample_weights and weights_list:
            sample_weights = pd.concat(weights_list, axis=0)
            sample_weights = sample_weights[X_train.index].fillna(1.0)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Feature selection
        if feature_selection:
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(n_features, len(self.feature_names)))
            X_train = selector.fit_transform(X_train, y_train)
            self.selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
            logger.info(f"Selected {len(self.selected_features)} features")
        else:
            self.selected_features = self.feature_names
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        
        # Optimize hyperparameters
        if optimize_params:
            self.best_params = self._optimize_hyperparameters(X_train, y_train, sample_weights)
        else:
            self.best_params = self._get_default_params()
        
        # Train final model
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            self.best_model = xgb.XGBClassifier(
                **self.best_params,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.best_model = lgb.LGBMClassifier(
                **self.best_params,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            raise ValueError(f"Model type {self.model_type} not available")
        
        # Fit model
        if sample_weights is not None:
            self.best_model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            self.best_model.fit(X_train, y_train)
        
        # Validate if validation data provided
        results = {
            'train_samples': len(X_train),
            'features_used': len(self.selected_features),
            'model_type': self.model_type,
            'best_params': self.best_params,
            'used_sample_weights': use_sample_weights
        }
        
        if val_data:
            val_results = self.validate(val_data)
            results['validation'] = val_results
        
        # Calculate training metrics
        train_pred = self.best_model.predict(X_train)
        results['train_accuracy'] = accuracy_score(y_train, train_pred)
        results['train_f1'] = f1_score(y_train, train_pred, average='weighted')
        
        logger.info(f"Training complete. Accuracy: {results['train_accuracy']:.4f}")
        
        # Store training history
        self.training_history.append(results)
        
        return results
    
    def validate(self,
                val_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[str, float]:
        """
        Validate model on validation data
        
        Args:
            val_data: Validation data dict {symbol: (ohlcv_df, features_df)}
            
        Returns:
            Validation metrics
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare validation data
        X_val_list = []
        y_val_list = []
        
        for symbol, (ohlcv_df, features_df) in val_data.items():
            X = self.prepare_features(ohlcv_df, features_df, symbol, '1h')
            y = self.create_labels(ohlcv_df)
            
            # Select features
            if self.selected_features:
                X = X[self.selected_features]
            
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            X_val_list.append(X)
            y_val_list.append(y)
        
        X_val = pd.concat(X_val_list, axis=0)
        y_val = pd.concat(y_val_list, axis=0)
        
        # Scale features
        X_val = self.scaler.transform(X_val)
        
        # Predict
        y_pred = self.best_model.predict(X_val)
        y_pred_proba = self.best_model.predict_proba(X_val)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted'),
        }
        
        # Add per-class metrics
        for i, label in enumerate(['down', 'flat', 'up']):
            class_mask = (y_val == i)
            if class_mask.any():
                pred_mask = (y_pred == i)
                metrics[f'{label}_precision'] = precision_score(class_mask, pred_mask)
                metrics[f'{label}_recall'] = recall_score(class_mask, pred_mask)
                metrics[f'{label}_f1'] = f1_score(class_mask, pred_mask)
        
        self.validation_scores.append(metrics)
        
        return metrics
    
    def predict(self,
               ohlcv_df: pd.DataFrame,
               features_df: pd.DataFrame,
               symbol: str = 'BTC/USDT',
               timeframe: str = '1h') -> PredictionResult:
        """
        Make prediction for new data
        
        Args:
            ohlcv_df: OHLCV DataFrame
            features_df: Features DataFrame
            symbol: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Prediction result
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        X = self.prepare_features(ohlcv_df, features_df, symbol, timeframe)
        
        # Select features
        if self.selected_features:
            X = X[self.selected_features]
        
        # Get last row for prediction
        X_last = X.iloc[-1:].fillna(0)
        
        # Scale
        X_scaled = self.scaler.transform(X_last)
        
        # Predict
        prediction = self.best_model.predict(X_scaled)[0]
        probabilities = self.best_model.predict_proba(X_scaled)[0]
        
        # Convert prediction back to -1, 0, 1
        prediction_mapped = prediction - 1
        
        # Calculate confidence
        confidence = np.max(probabilities)
        
        return PredictionResult(
            timestamp=X.index[-1] if hasattr(X.index[-1], '__len__') else pd.Timestamp.now(),
            symbol=symbol,
            timeframe=timeframe,
            prediction=prediction_mapped,
            probabilities=probabilities,
            confidence=confidence,
            features_used=self.selected_features,
            model_version=f"{self.model_type}_v1"
        )
    
    def predict_batch(self,
                     data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> List[PredictionResult]:
        """
        Make predictions for multiple assets
        
        Args:
            data: Dict {symbol: (ohlcv_df, features_df)}
            
        Returns:
            List of prediction results
        """
        predictions = []
        
        for symbol, (ohlcv_df, features_df) in data.items():
            try:
                pred = self.predict(ohlcv_df, features_df, symbol)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
        
        return predictions
    
    def _optimize_hyperparameters(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 sample_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            sample_weights: Optional sample weights
            
        Returns:
            Best parameters dictionary
        """
        logger.info("Optimizing hyperparameters...")
        
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            model = xgb.XGBClassifier(
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            
        elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50],
                'subsample': [0.8, 1.0]
            }
            
            model = lgb.LGBMClassifier(
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbosity=-1
            )
        else:
            return self._get_default_params()
        
        # Time series split for CV
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='f1_weighted',
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters"""
        if self.model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'min_child_samples': 20
            }
        else:
            return {}
    
    def walk_forward_validation(self,
                              data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
                              n_splits: int = 5,
                              train_size: int = 252,
                              test_size: int = 63) -> pd.DataFrame:
        """
        Perform walk-forward validation
        
        Args:
            data: Dict {symbol: (ohlcv_df, features_df)}
            n_splits: Number of splits
            train_size: Training window size
            test_size: Test window size
            
        Returns:
            DataFrame with validation results
        """
        results = []
        
        # Combine all data
        all_data = []
        for symbol, (ohlcv_df, features_df) in data.items():
            X = self.prepare_features(ohlcv_df, features_df, symbol, '1h')
            y = self.create_labels(ohlcv_df)
            combined = pd.concat([X, y.rename('label')], axis=1)
            all_data.append(combined)
        
        combined_data = pd.concat(all_data, axis=0).sort_index()
        
        # Walk-forward splits
        for i in range(n_splits):
            start_idx = i * test_size
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            if test_end > len(combined_data):
                break
            
            # Split data
            train_data = combined_data.iloc[start_idx:train_end]
            test_data = combined_data.iloc[train_end:test_end]
            
            # Separate features and labels
            X_train = train_data.drop('label', axis=1)
            y_train = train_data['label']
            X_test = test_data.drop('label', axis=1)
            y_test = test_data['label']
            
            # Select features
            if self.selected_features:
                X_train = X_train[self.selected_features]
                X_test = X_test[self.selected_features]
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'xgboost' and HAS_XGBOOST:
                model = xgb.XGBClassifier(**self._get_default_params(), random_state=self.random_state)
            else:
                model = lgb.LGBMClassifier(**self._get_default_params(), random_state=self.random_state)
            
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            split_results = {
                'split': i,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted')
            }
            
            results.append(split_results)
            logger.info(f"Split {i}: Accuracy={split_results['accuracy']:.4f}, F1={split_results['f1']:.4f}")
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        else:
            importance = np.zeros(len(self.selected_features))
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model
        """
        if self.best_model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history,
            'validation_scores': self.validation_scores,
            'labeling_config': self.labeling_config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
        """
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.training_history = model_data['training_history']
        self.validation_scores = model_data['validation_scores']
        
        # Load labeling config if available
        if 'labeling_config' in model_data:
            self.labeling_config = model_data['labeling_config']
            self.labeling_pipeline = LabelingPipeline(self.labeling_config)
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_prediction_confidence(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Analyze prediction confidence
        
        Args:
            probabilities: Prediction probabilities [P(down), P(flat), P(up)]
            
        Returns:
            Confidence analysis
        """
        return {
            'max_prob': np.max(probabilities),
            'entropy': -np.sum(probabilities * np.log(probabilities + 1e-10)),
            'margin': probabilities.max() - np.sort(probabilities)[-2] if len(probabilities) > 1 else 0,
            'prediction_strength': (probabilities.max() - 0.33) / 0.67
        }


# Example usage
if __name__ == "__main__":
    print("=== ML Predictor Test ===\n")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1h')
    
    # Generate sample OHLCV data
    np.random.seed(42)
    sample_ohlcv = pd.DataFrame({
        'open': np.random.uniform(40000, 45000, len(dates)),
        'high': np.random.uniform(45000, 46000, len(dates)),
        'low': np.random.uniform(39000, 40000, len(dates)),
        'close': np.random.uniform(40000, 45000, len(dates)),
        'volume': np.random.uniform(100, 1000, len(dates))
    }, index=dates)
    
    # Generate sample features (would come from feature_engineer.py)
    sample_features = pd.DataFrame({
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.uniform(-100, 100, len(dates)),
        'bb_position': np.random.uniform(0, 1, len(dates)),
        'volume_ratio': np.random.uniform(0.5, 2, len(dates)),
        'momentum': np.random.uniform(-0.1, 0.1, len(dates))
    }, index=dates)
    
    # Initialize predictor
    predictor = MLPredictor(model_type='lightgbm' if HAS_LIGHTGBM else 'xgboost')
    
    # Test labeling
    print("1. Testing Labeling...")
    labels = predictor.create_labels(sample_ohlcv, method='triple_barrier')
    print(f"   Label distribution: {labels.value_counts().to_dict()}")
    print(f"   Label percentages: Down={((labels==0).sum()/len(labels)):.1%}, "
          f"Flat={((labels==1).sum()/len(labels)):.1%}, "
          f"Up={((labels==2).sum()/len(labels)):.1%}\n")
    
    # Prepare training data
    print("2. Preparing training data...")
    train_data = {
        'BTC/USDT': (sample_ohlcv[:6000], sample_features[:6000]),
        'ETH/USDT': (sample_ohlcv[:6000], sample_features[:6000])
    }
    val_data = {
        'BTC/USDT': (sample_ohlcv[6000:], sample_features[6000:]),
        'ETH/USDT': (sample_ohlcv[6000:], sample_features[6000:])
    }
    
    # Train model
    print("3. Training model...")
    results = predictor.train(
        train_data,
        val_data,
        feature_selection=True,
        n_features=10,
        optimize_params=False,
        use_sample_weights=True  # Test sample weights
    )
    print(f"   Training accuracy: {results['train_accuracy']:.4f}")
    print(f"   Training F1: {results['train_f1']:.4f}")
    if 'validation' in results:
        print(f"   Validation accuracy: {results['validation']['accuracy']:.4f}")
        print(f"   Validation F1: {results['validation']['f1']:.4f}\n")
    
    # Make prediction
    print("4. Making prediction...")
    prediction = predictor.predict(
        sample_ohlcv.iloc[-100:],
        sample_features.iloc[-100:],
        'BTC/USDT',
        '1h'
    )
    print(f"   Prediction: {['Down', 'Flat', 'Up'][prediction.prediction + 1]}")
    print(f"   Probabilities: Down={prediction.probabilities[0]:.3f}, "
          f"Flat={prediction.probabilities[1]:.3f}, "
          f"Up={prediction.probabilities[2]:.3f}")
    print(f"   Confidence: {prediction.confidence:.3f}\n")
    
    # Feature importance
    print("5. Top Features:")
    importance_df = predictor.get_feature_importance(top_n=5)
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    print("\n=== ML Predictor Ready ===")