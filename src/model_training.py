"""
Advanced Model Training Module for Fraud Detection

This module provides comprehensive model training capabilities including
hyperparameter optimization, ensemble methods, and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, f1_score, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
import shap
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Advanced fraud detection model training and evaluation class.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = 'fraud'):
        """
        Initialize the fraud detection model.
        
        Args:
            data: Preprocessed dataset
            target_column: Name of the target variable
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")
            
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.results = {}
        self.feature_selector = None
        self.scaler = None
        
        # Prepare features and target
        self.X = data.drop(columns=[target_column])
        self.y = data[target_column]
        
        logger.info(f"Initialized model with {len(self.X)} samples and {len(self.X.columns)} features")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets with stratification.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, 
            stratify=self.y
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, method: str = 'smote', random_state: int = 42):
        """
        Handle class imbalance using various techniques.
        
        Args:
            method: Balancing method ('smote', 'adasyn', 'undersample', 'none')
            random_state: Random state for reproducibility
        """
        if method == 'none':
            logger.info("Skipping class balancing")
            return self.X_train, self.y_train
            
        logger.info(f"Applying {method} for class balancing")
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            logger.warning(f"Unknown balancing method: {method}")
            return self.X_train, self.y_train
        
        X_resampled, y_resampled = sampler.fit_resample(self.X_train, self.y_train)
        
        logger.info(f"Original class distribution: {np.bincount(self.y_train)}")
        logger.info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def select_features(self, method: str = 'kbest', n_features: int = 50):
        """
        Perform feature selection.
        
        Args:
            method: Feature selection method ('kbest', 'rfe', 'none')
            n_features: Number of features to select
        """
        if method == 'none':
            logger.info("Skipping feature selection")
            return self.X_train, self.X_test
            
        logger.info(f"Applying {method} feature selection")
        
        if method == 'kbest':
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'rfe':
            # Use Random Forest for RFE
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_selector = RFE(estimator=base_model, n_features_to_select=n_features)
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return self.X_train, self.X_test
        
        X_train_selected = self.feature_selector.fit_transform(self.X_train, self.y_train)
        X_test_selected = self.feature_selector.transform(self.X_test)
        
        # Get selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_features = self.X_train.columns[self.feature_selector.get_support()]
            logger.info(f"Selected {len(selected_features)} features")
            logger.info(f"Top features: {list(selected_features[:10])}")
        
        return X_train_selected, X_test_selected
    
    def train_baseline_models(self, balance_method: str = 'smote', 
                            feature_selection: str = 'kbest') -> Dict[str, Any]:
        """
        Train baseline models for comparison with advanced preprocessing.
        
        Args:
            balance_method: Method for handling class imbalance
            feature_selection: Method for feature selection
            
        Returns:
            Dictionary with model results
        """
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(balance_method)
        
        # Feature selection
        X_train_selected, X_test_selected = self.select_features(feature_selection)
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                random_state=42, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6, 
                random_state=42, n_jobs=-1, verbose=-1
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train_selected, y_train_balanced)
                
                # Make predictions
                y_pred = model.predict(X_test_selected)
                y_prob = model.predict_proba(X_test_selected)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
                
                # Store results
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'metrics': metrics
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    if self.feature_selector is not None:
                        # Map feature importance to original features
                        importance = np.zeros(len(self.X.columns))
                        importance[self.feature_selector.get_support()] = model.feature_importances_
                        self.feature_importance[name] = dict(zip(self.X.columns, importance))
                    else:
                        self.feature_importance[name] = dict(zip(self.X.columns, model.feature_importances_))
                
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.models = results
        return results
    
    def hyperparameter_optimization(self, model_name: str = 'XGBoost', 
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using GridSearchCV.
        
        Args:
            model_name: Name of the model to optimize
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best model and parameters
        """
        logger.info(f"Performing hyperparameter optimization for {model_name}")
        
        # Handle class imbalance and feature selection
        X_train_balanced, y_train_balanced = self.handle_class_imbalance('smote')
        X_train_selected, _ = self.select_features('kbest')
        
        if model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        elif model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0, 0.1, 0.5]
            }
            model = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        
        else:
            logger.warning(f"Hyperparameter optimization not implemented for {model_name}")
            return {}
        
        # Perform grid search with stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_selected, y_train_balanced)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Evaluate best model
        _, X_test_selected = self.select_features('kbest')
        y_pred = best_model.predict(X_test_selected)
        y_prob = best_model.predict_proba(X_test_selected)[:, 1]
        metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
        
        results = {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'metrics': metrics,
            'cv_results': grid_search.cv_results_
        }
        
        self.best_model = best_model
        logger.info(f"Best {model_name} parameters: {best_params}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return results
    
    def create_ensemble_model(self, models_to_ensemble: List[str] = None,
                            voting_method: str = 'soft') -> Dict[str, Any]:
        """
        Create an ensemble model from multiple base models.
        
        Args:
            models_to_ensemble: List of model names to ensemble
            voting_method: Voting method ('soft', 'hard')
            
        Returns:
            Ensemble model results
        """
        if models_to_ensemble is None:
            models_to_ensemble = ['Random Forest', 'XGBoost', 'LightGBM']
        
        logger.info(f"Creating ensemble from: {models_to_ensemble}")
        
        # Get base models
        base_models = []
        for model_name in models_to_ensemble:
            if model_name in self.models:
                base_models.append((model_name.lower().replace(' ', '_'), 
                                  self.models[model_name]['model']))
        
        if len(base_models) < 2:
            logger.error("Need at least 2 models for ensemble")
            return {}
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=base_models,
            voting=voting_method,
            n_jobs=-1
        )
        
        # Train ensemble
        X_train_balanced, y_train_balanced = self.handle_class_imbalance('smote')
        X_train_selected, X_test_selected = self.select_features('kbest')
        
        ensemble.fit(X_train_selected, y_train_balanced)
        
        # Make predictions
        y_pred = ensemble.predict(X_test_selected)
        y_prob = ensemble.predict_proba(X_test_selected)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_prob)
        
        results = {
            'model': ensemble,
            'predictions': y_pred,
            'probabilities': y_prob,
            'metrics': metrics,
            'base_models': models_to_ensemble
        }
        
        logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob),
            'fraud_detection_rate': recall_score(y_true, y_pred, zero_division=0),
            'false_positive_rate': 1 - precision_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'balanced_accuracy': (metrics['recall'] + metrics['specificity']) / 2
        })
        
        return metrics
    
    def plot_feature_importance(self, model_name: str = 'XGBoost', top_n: int = 20) -> None:
        """
        Plot feature importance for a given model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to display
        """
        if model_name not in self.feature_importance:
            logger.warning(f"No feature importance available for {model_name}")
            return
        
        importance = self.feature_importance[model_name]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self) -> None:
        """
        Plot ROC curves for all trained models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, result in self.models.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curves(self) -> None:
        """
        Plot Precision-Recall curves for all trained models.
        """
        plt.figure(figsize=(10, 8))
        
        for name, result in self.models.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['probabilities'])
            ap = result['metrics']['average_precision']
            plt.plot(recall, precision, label=f'{name} (AP = {ap:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def explain_model_predictions(self, model_name: str = 'XGBoost', 
                                sample_size: int = 100) -> None:
        """
        Explain model predictions using SHAP values.
        
        Args:
            model_name: Name of the model to explain
            sample_size: Number of samples to use for explanation
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return
        
        model = self.models[model_name]['model']
        
        # Use a subset of test data for faster computation
        X_sample = self.X_test.iloc[:sample_size]
        
        try:
            # Create SHAP explainer
            if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Plot summary
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar")
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating SHAP explanation: {str(e)}")
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if model_name in self.models:
            model_data = {
                'model': self.models[model_name]['model'],
                'feature_selector': self.feature_selector,
                'scaler': self.scaler,
                'feature_columns': list(self.X.columns),
                'target_column': self.target_column,
                'training_date': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
        elif model_name == 'best' and self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'feature_selector': self.feature_selector,
                'scaler': self.scaler,
                'feature_columns': list(self.X.columns),
                'target_column': self.target_column,
                'training_date': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Best model saved to {filepath}")
        else:
            logger.error(f"Model {model_name} not found")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive model performance report.
        
        Returns:
            Dictionary containing the report
        """
        report = {
            'dataset_info': {
                'total_samples': len(self.data),
                'features': len(self.X.columns),
                'fraud_rate': self.y.mean(),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test)
            },
            'model_performance': {},
            'best_model': None,
            'recommendations': [],
            'training_summary': {
                'models_trained': len(self.models),
                'feature_selection_applied': self.feature_selector is not None,
                'class_balancing_applied': True  # Assuming SMOTE was used
            }
        }
        
        # Add model performance
        for name, result in self.models.items():
            report['model_performance'][name] = result['metrics']
        
        # Find best model
        if self.models:
            best_model = max(self.models.items(), 
                            key=lambda x: x[1]['metrics']['f1_score'])
            report['best_model'] = {
                'name': best_model[0],
                'metrics': best_model[1]['metrics']
            }
        
        # Generate recommendations
        fraud_rate = self.y.mean()
        if fraud_rate < 0.01:
            report['recommendations'].append("Consider using SMOTE for class balancing")
        
        if self.models:
            best_f1 = best_model[1]['metrics']['f1_score']
            if best_f1 < 0.8:
                report['recommendations'].append("Model performance could be improved with feature engineering")
            
            # Check for overfitting
            for name, result in self.models.items():
                if 'cv_results' in result:
                    cv_scores = result['cv_results']['mean_test_score']
                    test_score = result['metrics']['f1_score']
                    if test_score < np.mean(cv_scores) - 0.1:
                        report['recommendations'].append(f"{name} may be overfitting")
        
        return report


def main():
    """
    Main function to demonstrate the model training pipeline.
    """
    # Load and preprocess data
    from data_processing import DataPreprocessing
    from utils.utils import load_data
    
    # Load sample data (replace with your actual data path)
    try:
        data = load_data('data/processed/cleaned_fraud_data.csv')
    except FileNotFoundError:
        logger.warning("Sample data not found, creating synthetic data")
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'user_id': range(n_samples),
            'amount': np.random.exponential(100, n_samples),
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
            'merchant_category': np.random.choice(['retail', 'online', 'travel'], n_samples),
            'payment_method': np.random.choice(['credit_card', 'debit_card'], n_samples),
            'country': np.random.choice(['US', 'UK', 'CA'], n_samples),
            'device_type': np.random.choice(['mobile', 'desktop'], n_samples),
            'fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        })
    
    # Initialize preprocessing
    dp = DataPreprocessing(data)
    processed_data = (dp
                     .handle_missing_values(method='fill')
                     .remove_duplicates()
                     .correct_data_types()
                     .create_fraud_features())
    
    # Initialize model
    model = FraudDetectionModel(processed_data, 'fraud')
    
    # Split data
    model.split_data()
    
    # Train baseline models
    results = model.train_baseline_models(balance_method='smote', feature_selection='kbest')
    
    # Perform hyperparameter optimization
    best_model = model.hyperparameter_optimization('XGBoost')
    
    # Create ensemble
    ensemble_results = model.create_ensemble_model()
    
    # Generate plots
    model.plot_feature_importance()
    model.plot_roc_curves()
    model.plot_precision_recall_curves()
    
    # Generate report
    report = model.generate_report()
    print("Model Training Report:")
    print(report)
    
    # Save best model
    model.save_model('best', 'models/best_fraud_detector.pkl')


if __name__ == "__main__":
    main()
