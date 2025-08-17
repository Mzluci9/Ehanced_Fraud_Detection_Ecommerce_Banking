"""
Unit tests for data processing module.

This module contains comprehensive tests for the DataPreprocessing class
and related functionality to ensure data quality and processing reliability.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataPreprocessing


class TestDataPreprocessing:
    """Test cases for DataPreprocessing class."""
    
    @pytest.fixture
    def sample_fraud_data(self):
        """Create sample fraud detection dataset for testing."""
        np.random.seed(42)
        
        data = {
            'user_id': range(100),
            'signup_time': pd.date_range('2023-01-01', periods=100, freq='H'),
            'purchase_time': pd.date_range('2023-01-01', periods=100, freq='H'),
            'purchase_value': np.random.uniform(10, 1000, 100),
            'device_id': np.random.choice(['A', 'B', 'C'], 100),
            'source': np.random.choice(['SEO', 'Ads', 'Direct'], 100),
            'browser': np.random.choice(['Chrome', 'Safari', 'Firefox'], 100),
            'sex': np.random.choice(['M', 'F'], 100),
            'age': np.random.randint(18, 80, 100),
            'ip_address': np.random.randint(100000000, 999999999, 100),
            'class': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample dataset with missing values."""
        data = {
            'user_id': [1, 2, 3, 4, 5],
            'amount': [100, np.nan, 300, 400, np.nan],
            'category': ['A', 'B', np.nan, 'A', 'C'],
            'fraud': [0, 1, 0, 1, 0]
        }
        return pd.DataFrame(data)
    
    def test_initialization(self, sample_fraud_data):
        """Test DataPreprocessing initialization."""
        dp = DataPreprocessing(sample_fraud_data)
        
        assert dp.data.shape == sample_fraud_data.shape
        assert isinstance(dp.scalers, dict)
        assert isinstance(dp.label_encoders, dict)
        assert isinstance(dp.imputers, dict)
    
    def test_handle_missing_values_drop(self, sample_data_with_missing):
        """Test missing value handling with drop method."""
        dp = DataPreprocessing(sample_data_with_missing)
        result = dp.handle_missing_values(method='drop')
        
        # Should drop rows with any missing values
        assert len(result) == 2  # Only rows 1 and 4 have no missing values
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_fill(self, sample_data_with_missing):
        """Test missing value handling with fill method."""
        dp = DataPreprocessing(sample_data_with_missing)
        result = dp.handle_missing_values(method='fill')
        
        # Should fill missing values
        assert len(result) == len(sample_data_with_missing)
        assert result.isnull().sum().sum() == 0
    
    def test_handle_missing_values_fill_custom(self, sample_data_with_missing):
        """Test missing value handling with custom fill value."""
        dp = DataPreprocessing(sample_data_with_missing)
        result = dp.handle_missing_values(method='fill', fill_value=999)
        
        assert len(result) == len(sample_data_with_missing)
        assert result.isnull().sum().sum() == 0
        assert 999 in result['amount'].values
    
    def test_remove_duplicates(self, sample_fraud_data):
        """Test duplicate removal."""
        # Add duplicates
        sample_fraud_data_with_duplicates = pd.concat([
            sample_fraud_data, sample_fraud_data.iloc[:5]
        ])
        
        dp = DataPreprocessing(sample_fraud_data_with_duplicates)
        result = dp.remove_duplicates()
        
        assert len(result) == len(sample_fraud_data)
    
    def test_correct_data_types(self, sample_fraud_data):
        """Test data type correction."""
        dp = DataPreprocessing(sample_fraud_data)
        result = dp.correct_data_types()
        
        # Check datetime conversion
        assert pd.api.types.is_datetime64_any_dtype(result['signup_time'])
        assert pd.api.types.is_datetime64_any_dtype(result['purchase_time'])
        
        # Check numeric conversion for IP address
        assert pd.api.types.is_numeric_dtype(result['ip_address'])
    
    def test_normalize_and_scale(self, sample_fraud_data):
        """Test feature scaling."""
        dp = DataPreprocessing(sample_fraud_data)
        columns_to_scale = ['purchase_value', 'age']
        result = dp.normalize_and_scale(columns_to_scale)
        
        # Check that scaling was applied
        assert 'standard' in dp.scalers
        assert len(dp.scalers['standard'].mean_) == len(columns_to_scale)
        
        # Check that scaled columns have mean close to 0 and std close to 1
        for col in columns_to_scale:
            assert abs(result[col].mean()) < 1e-10
            assert abs(result[col].std() - 1) < 1e-10
    
    def test_encode_categorical_onehot(self, sample_fraud_data):
        """Test one-hot encoding."""
        dp = DataPreprocessing(sample_fraud_data)
        categorical_cols = ['source', 'browser']
        result = dp.encode_categorical(categorical_cols, method='onehot')
        
        # Check that new columns were created
        expected_new_cols = ['source_Ads', 'source_Direct', 'browser_Safari', 
                           'browser_Firefox']
        for col in expected_new_cols:
            assert col in result.columns
    
    def test_encode_categorical_label(self, sample_fraud_data):
        """Test label encoding."""
        dp = DataPreprocessing(sample_fraud_data)
        categorical_cols = ['source']
        result = dp.encode_categorical(categorical_cols, method='label')
        
        # Check that label encoder was stored
        assert 'source' in dp.label_encoders
        assert pd.api.types.is_numeric_dtype(result['source'])
    
    def test_encode_categorical_frequency(self, sample_fraud_data):
        """Test frequency encoding."""
        dp = DataPreprocessing(sample_fraud_data)
        categorical_cols = ['source']
        result = dp.encode_categorical(categorical_cols, method='frequency')
        
        # Check that frequency encoding was applied
        assert pd.api.types.is_numeric_dtype(result['source'])
        assert result['source'].min() > 0
    
    def test_create_fraud_features(self, sample_fraud_data):
        """Test fraud-specific feature creation."""
        dp = DataPreprocessing(sample_fraud_data)
        result = dp.create_fraud_features()
        
        # Check that new features were created
        expected_features = [
            'time_to_purchase_hours', 'signup_hour', 'purchase_hour',
            'signup_day_of_week', 'purchase_day_of_week',
            'signup_month', 'purchase_month'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check that time features are numeric
        assert pd.api.types.is_numeric_dtype(result['time_to_purchase_hours'])
        assert pd.api.types.is_numeric_dtype(result['signup_hour'])
    
    def test_validate_data(self, sample_fraud_data):
        """Test data validation."""
        dp = DataPreprocessing(sample_fraud_data)
        validation_results = dp.validate_data()
        
        # Check validation results structure
        assert 'total_rows' in validation_results
        assert 'total_columns' in validation_results
        assert 'missing_values' in validation_results
        assert 'duplicate_rows' in validation_results
        assert 'data_types' in validation_results
        
        # Check specific values
        assert validation_results['total_rows'] == 100
        assert validation_results['total_columns'] == 11
    
    def test_get_preprocessing_summary(self, sample_fraud_data):
        """Test preprocessing summary."""
        dp = DataPreprocessing(sample_fraud_data)
        
        # Apply some preprocessing
        dp.handle_missing_values()
        dp.normalize_and_scale(['purchase_value'])
        dp.encode_categorical(['source'], method='label')
        
        summary = dp.get_preprocessing_summary()
        
        # Check summary structure
        assert 'final_shape' in summary
        assert 'scalers_applied' in summary
        assert 'label_encoders_applied' in summary
        assert 'imputers_applied' in summary
        
        # Check that applied methods are recorded
        assert 'standard' in summary['scalers_applied']
        assert 'source' in summary['label_encoders_applied']


class TestDataProcessingIntegration:
    """Integration tests for data processing pipeline."""
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create test data
        data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'amount': [100, np.nan, 300, 400, 500],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'timestamp': pd.date_range('2023-01-01', periods=5),
            'fraud': [0, 1, 0, 1, 0]
        })
        
        # Apply full pipeline
        dp = DataPreprocessing(data)
        result = (dp
                 .handle_missing_values(method='fill')
                 .remove_duplicates()
                 .correct_data_types()
                 .normalize_and_scale(['amount'])
                 .encode_categorical(['category'], method='onehot')
                 .create_fraud_features())
        
        # Verify results
        assert len(result) == 5
        assert result.isnull().sum().sum() == 0
        assert 'category_A' in result.columns
        assert 'category_B' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
    
    def test_error_handling(self):
        """Test error handling in preprocessing."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        dp = DataPreprocessing(empty_df)
        
        # Should handle gracefully
        result = dp.handle_missing_values()
        assert len(result) == 0
        
        # Test with invalid method
        data = pd.DataFrame({'col': [1, 2, 3]})
        dp = DataPreprocessing(data)
        
        # Should handle invalid method gracefully
        result = dp.handle_missing_values(method='invalid_method')
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__])
