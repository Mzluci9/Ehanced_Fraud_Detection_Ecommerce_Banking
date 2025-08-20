"""
Unit tests for model training module.

This module contains comprehensive tests for the FraudDetectionModel class
and related functionality to ensure model training reliability and performance.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import FraudDetectionModel


class TestFraudDetectionModel:
    """Test cases for FraudDetectionModel class."""
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud detection dataset for testing."""
        np.random.seed(42)
        
        data = {
            'user_id': range(100),
            'amount': np.random.uniform(10, 1000, 100),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'merchant_category': np.random.choice(['retail', 'online', 'travel'], 100),
            'payment_method': np.random.choice(['credit_card', 'debit_card'], 100),
            'country': np.random.choice(['US', 'UK', 'CA'], 100),
            'device_type': np.random.choice(['mobile', 'desktop'], 100),
            'fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self, sample_fraud_data):
        """Test FraudDetectionModel initialization."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        
        assert model.data.shape == sample_fraud_data.shape
        assert model.target_column == 'fraud'
        assert len(model.X.columns) == len(sample_fraud_data.columns) - 1
        assert len(model.y) == len(sample_fraud_data)
        assert isinstance(model.models, dict)
        assert isinstance(model.feature_importance, dict)
    
    def test_initialization_with_invalid_data(self):
        """Test initialization with invalid data."""
        with pytest.raises(ValueError, match="Input data cannot be None or empty"):
            FraudDetectionModel(None, 'fraud')
        
        with pytest.raises(ValueError, match="Input data cannot be None or empty"):
            FraudDetectionModel(pd.DataFrame(), 'fraud')
    
    def test_split_data(self, sample_fraud_data):
        """Test data splitting functionality."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        X_train, X_test, y_train, y_test = model.split_data(test_size=0.2)
        
        # Check split proportions
        assert len(X_train) == 80  # 80% of 100
        assert len(X_test) == 20   # 20% of 100
        assert len(y_train) == 80
        assert len(y_test) == 20
        
        # Check that features and target are properly separated
        assert 'fraud' not in X_train.columns
        assert 'fraud' not in X_test.columns
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_handle_class_imbalance(self, sample_fraud_data):
        """Test class imbalance handling."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Test SMOTE
        X_balanced, y_balanced = model.handle_class_imbalance('smote')
        assert len(X_balanced) == len(y_balanced)
        
        # Test no balancing
        X_orig, y_orig = model.handle_class_imbalance('none')
        assert len(X_orig) == len(model.X_train)
        assert len(y_orig) == len(model.y_train)
    
    def test_select_features(self, sample_fraud_data):
        """Test feature selection."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Test k-best feature selection
        X_train_selected, X_test_selected = model.select_features('kbest', n_features=5)
        assert X_train_selected.shape[1] == 5
        assert X_test_selected.shape[1] == 5
        
        # Test no feature selection
        X_train_orig, X_test_orig = model.select_features('none')
        assert X_train_orig.shape[1] == len(model.X_train.columns)
        assert X_test_orig.shape[1] == len(model.X_test.columns)
    
    @patch('model_training.xgb.XGBClassifier')
    @patch('model_training.lgb.LGBMClassifier')
    @patch('model_training.RandomForestClassifier')
    def test_train_baseline_models(self, mock_rf, mock_lgb, mock_xgb, sample_fraud_data):
        """Test baseline model training."""
        # Mock the models
        mock_rf.return_value.fit.return_value = None
        mock_rf.return_value.predict.return_value = np.array([0, 1, 0, 1] * 5)
        mock_rf.return_value.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]] * 10)
        mock_rf.return_value.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        
        mock_lgb.return_value.fit.return_value = None
        mock_lgb.return_value.predict.return_value = np.array([0, 1, 0, 1] * 5)
        mock_lgb.return_value.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]] * 10)
        mock_lgb.return_value.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        
        mock_xgb.return_value.fit.return_value = None
        mock_xgb.return_value.predict.return_value = np.array([0, 1, 0, 1] * 5)
        mock_xgb.return_value.predict_proba.return_value = np.array([[0.7, 0.3], [0.2, 0.8]] * 10)
        mock_xgb.return_value.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.4])
        
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        results = model.train_baseline_models(balance_method='none', feature_selection='none')
        
        # Check that models were trained
        assert len(results) > 0
        assert 'Random Forest' in results
        assert 'XGBoost' in results
        assert 'LightGBM' in results
        
        # Check that metrics were calculated
        for model_name, result in results.items():
            assert 'metrics' in result
            assert 'accuracy' in result['metrics']
            assert 'f1_score' in result['metrics']
    
    def test_calculate_metrics(self, sample_fraud_data):
        """Test metrics calculation."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        
        # Create sample predictions
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.1])
        
        metrics = model._calculate_metrics(y_true, y_pred, y_prob)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                          'roc_auc', 'average_precision', 'specificity']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_hyperparameter_optimization(self, sample_fraud_data):
        """Test hyperparameter optimization."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Test with a small parameter grid for faster execution
        with patch('model_training.GridSearchCV') as mock_grid_search:
            mock_grid_search.return_value.best_estimator_ = MagicMock()
            mock_grid_search.return_value.best_params_ = {'n_estimators': 100}
            mock_grid_search.return_value.best_score_ = 0.85
            mock_grid_search.return_value.cv_results_ = {}
            
            result = model.hyperparameter_optimization('Random Forest', cv_folds=2)
            
            assert 'best_model' in result
            assert 'best_params' in result
            assert 'best_score' in result
    
    def test_create_ensemble_model(self, sample_fraud_data):
        """Test ensemble model creation."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Mock some models first
        model.models = {
            'Random Forest': {
                'model': MagicMock(),
                'predictions': np.array([0, 1, 0, 1] * 5),
                'probabilities': np.array([0.1, 0.9, 0.2, 0.8] * 5),
                'metrics': {'f1_score': 0.8}
            },
            'XGBoost': {
                'model': MagicMock(),
                'predictions': np.array([0, 1, 0, 1] * 5),
                'probabilities': np.array([0.1, 0.9, 0.2, 0.8] * 5),
                'metrics': {'f1_score': 0.85}
            }
        }
        
        result = model.create_ensemble_model(['Random Forest', 'XGBoost'])
        
        assert 'model' in result
        assert 'base_models' in result
        assert len(result['base_models']) == 2
    
    def test_save_and_load_model(self, sample_fraud_data, tmp_path):
        """Test model saving and loading."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Mock a model
        model.models['Test Model'] = {
            'model': MagicMock(),
            'predictions': np.array([0, 1, 0, 1]),
            'probabilities': np.array([0.1, 0.9, 0.2, 0.8]),
            'metrics': {'f1_score': 0.8}
        }
        
        # Test saving
        save_path = tmp_path / "test_model.pkl"
        model.save_model('Test Model', str(save_path))
        
        # Check that file was created
        assert save_path.exists()
    
    def test_generate_report(self, sample_fraud_data):
        """Test report generation."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Mock some results
        model.models = {
            'Test Model': {
                'model': MagicMock(),
                'predictions': np.array([0, 1, 0, 1]),
                'probabilities': np.array([0.1, 0.9, 0.2, 0.8]),
                'metrics': {'f1_score': 0.8, 'accuracy': 0.75}
            }
        }
        
        report = model.generate_report()
        
        # Check report structure
        assert 'dataset_info' in report
        assert 'model_performance' in report
        assert 'recommendations' in report
        assert 'training_summary' in report
        
        # Check dataset info
        assert report['dataset_info']['total_samples'] == 100
        assert report['dataset_info']['features'] == 7  # 8 columns - 1 target
    
    def test_error_handling(self, sample_fraud_data):
        """Test error handling in various scenarios."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Test with invalid balancing method
        X_balanced, y_balanced = model.handle_class_imbalance('invalid_method')
        assert len(X_balanced) == len(model.X_train)  # Should fall back to original
        
        # Test with invalid feature selection method
        X_selected, _ = model.select_features('invalid_method')
        assert X_selected.shape[1] == len(model.X_train.columns)  # Should fall back to original


class TestModelTrainingIntegration:
    """Integration tests for model training pipeline."""
    
    def test_full_training_pipeline(self, sample_fraud_data):
        """Test complete model training pipeline."""
        # Initialize model
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        
        # Split data
        model.split_data(test_size=0.3)
        
        # Handle class imbalance
        X_balanced, y_balanced = model.handle_class_imbalance('smote')
        assert len(X_balanced) >= len(model.X_train)
        
        # Feature selection
        X_train_selected, X_test_selected = model.select_features('kbest', n_features=3)
        assert X_train_selected.shape[1] == 3
        assert X_test_selected.shape[1] == 3
        
        # Verify data integrity
        assert not np.isnan(X_train_selected).any()
        assert not np.isnan(X_test_selected).any()
    
    def test_model_performance_validation(self, sample_fraud_data):
        """Test that model performance metrics are reasonable."""
        model = FraudDetectionModel(sample_fraud_data, 'fraud')
        model.split_data()
        
        # Create a simple mock model for testing
        from sklearn.ensemble import RandomForestClassifier
        
        # Train a simple model
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(model.X_train, model.y_train)
        
        # Make predictions
        y_pred = rf_model.predict(model.X_test)
        y_prob = rf_model.predict_proba(model.X_test)[:, 1]
        
        # Calculate metrics
        metrics = model._calculate_metrics(model.y_test, y_pred, y_prob)
        
        # Validate metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['specificity'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
