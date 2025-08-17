"""
Utility functions for fraud detection project.

This module provides helper functions for data loading, validation,
visualization, and common operations used throughout the fraud detection
pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Optional, Dict, Any, Tuple
import os
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats with error handling.
    
    Args:
        filepath (str): Path to the data file
        **kwargs: Additional arguments to pass to pandas read function
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_extension = filepath.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            data = pd.read_csv(filepath, **kwargs)
        elif file_extension in ['xlsx', 'xls']:
            data = pd.read_excel(filepath, **kwargs)
        elif file_extension == 'json':
            data = pd.read_json(filepath, **kwargs)
        elif file_extension == 'parquet':
            data = pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully loaded data from {filepath} with shape: "
                   f"{data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {str(e)}")
        raise


def save_data(data: pd.DataFrame, filepath: str, **kwargs) -> None:
    """
    Save data to various file formats.
    
    Args:
        data (pd.DataFrame): Data to save
        filepath (str): Path where to save the file
        **kwargs: Additional arguments to pass to pandas save function
    """
    file_extension = filepath.lower().split('.')[-1]
    
    try:
        if file_extension == 'csv':
            data.to_csv(filepath, index=False, **kwargs)
        elif file_extension in ['xlsx', 'xls']:
            data.to_excel(filepath, index=False, **kwargs)
        elif file_extension == 'json':
            data.to_json(filepath, orient='records', **kwargs)
        elif file_extension == 'parquet':
            data.to_parquet(filepath, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Successfully saved data to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {str(e)}")
        raise


def validate_fraud_data(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate fraud detection dataset for common issues.
    
    Args:
        data (pd.DataFrame): Dataset to validate
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_results = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'data_types': data.dtypes.to_dict(),
        'fraud_rate': None,
        'warnings': []
    }
    
    # Check for fraud column
    fraud_columns = ['class', 'Class', 'fraud', 'is_fraud', 'target']
    fraud_col = None
    for col in fraud_columns:
        if col in data.columns:
            fraud_col = col
            break
    
    if fraud_col:
        fraud_rate = data[fraud_col].value_counts(normalize=True)
        validation_results['fraud_rate'] = fraud_rate.to_dict()
        
        # Check for class imbalance
        if len(fraud_rate) == 2:
            min_class_rate = fraud_rate.min()
            if min_class_rate < 0.01:
                validation_results['warnings'].append(
                    f"Severe class imbalance detected: {min_class_rate:.3f}")
            elif min_class_rate < 0.1:
                validation_results['warnings'].append(
                    f"Class imbalance detected: {min_class_rate:.3f}")
    else:
        validation_results['warnings'].append("No fraud column found")
    
    # Check for required columns in fraud detection
    required_cols = ['user_id', 'amount', 'timestamp']
    missing_required = [col for col in required_cols if col not in data.columns]
    if missing_required:
        validation_results['warnings'].append(
            f"Missing common fraud detection columns: {missing_required}")
    
    return validation_results


def plot_fraud_distribution(data: pd.DataFrame, fraud_col: str = 'class',
                           figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create comprehensive fraud distribution plots.
    
    Args:
        data (pd.DataFrame): Dataset containing fraud information
        fraud_col (str): Name of the fraud column
        figsize (Tuple[int, int]): Figure size
    """
    if fraud_col not in data.columns:
        logger.warning(f"Fraud column '{fraud_col}' not found in data")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Fraud Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Fraud vs Non-fraud count
    fraud_counts = data[fraud_col].value_counts()
    axes[0, 0].pie(fraud_counts.values, labels=fraud_counts.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Fraud vs Non-Fraud Distribution')
    
    # 2. Fraud rate over time (if timestamp available)
    time_cols = ['timestamp', 'purchase_time', 'transaction_time']
    time_col = None
    for col in time_cols:
        if col in data.columns:
            time_col = col
            break
    
    if time_col:
        data[time_col] = pd.to_datetime(data[time_col])
        fraud_by_time = data.groupby(data[time_col].dt.date)[fraud_col].mean()
        axes[0, 1].plot(fraud_by_time.index, fraud_by_time.values)
        axes[0, 1].set_title('Fraud Rate Over Time')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Amount distribution by fraud status
    amount_cols = ['amount', 'purchase_value', 'transaction_amount']
    amount_col = None
    for col in amount_cols:
        if col in data.columns:
            amount_col = col
            break
    
    if amount_col:
        for fraud_status in data[fraud_col].unique():
            subset = data[data[fraud_col] == fraud_status][amount_col]
            axes[1, 0].hist(subset, alpha=0.7, label=f'Fraud: {fraud_status}')
        axes[1, 0].set_title('Amount Distribution by Fraud Status')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Amount')
        axes[1, 0].set_ylabel('Frequency')
    
    # 4. Fraud rate by categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        # Use the first categorical column
        cat_col = categorical_cols[0]
        fraud_by_cat = data.groupby(cat_col)[fraud_col].mean().sort_values(
            ascending=False)
        fraud_by_cat.head(10).plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title(f'Fraud Rate by {cat_col}')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def create_interactive_fraud_dashboard(data: pd.DataFrame, 
                                     fraud_col: str = 'class') -> go.Figure:
    """
    Create an interactive Plotly dashboard for fraud analysis.
    
    Args:
        data (pd.DataFrame): Dataset for analysis
        fraud_col (str): Name of the fraud column
        
    Returns:
        go.Figure: Interactive dashboard figure
    """
    if fraud_col not in data.columns:
        logger.warning(f"Fraud column '{fraud_col}' not found in data")
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Fraud Distribution', 'Amount vs Fraud Status',
                       'Fraud Rate by Category', 'Time Series Analysis'),
        specs=[[{"type": "pie"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Pie chart for fraud distribution
    fraud_counts = data[fraud_col].value_counts()
    fig.add_trace(
        go.Pie(labels=fraud_counts.index, values=fraud_counts.values,
               name="Fraud Distribution"),
        row=1, col=1
    )
    
    # 2. Box plot for amount distribution
    amount_cols = ['amount', 'purchase_value', 'transaction_amount']
    amount_col = None
    for col in amount_cols:
        if col in data.columns:
            amount_col = col
            break
    
    if amount_col:
        for fraud_status in data[fraud_col].unique():
            subset = data[data[fraud_col] == fraud_status][amount_col]
            fig.add_trace(
                go.Box(y=subset, name=f'Fraud: {fraud_status}'),
                row=1, col=2
            )
    
    # 3. Bar chart for categorical analysis
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_col = categorical_cols[0]
        fraud_by_cat = data.groupby(cat_col)[fraud_col].mean().sort_values(
            ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=fraud_by_cat.index, y=fraud_by_cat.values,
                   name=f'Fraud Rate by {cat_col}'),
            row=2, col=1
        )
    
    # 4. Time series analysis
    time_cols = ['timestamp', 'purchase_time', 'transaction_time']
    time_col = None
    for col in time_cols:
        if col in data.columns:
            time_col = col
            break
    
    if time_col:
        data[time_col] = pd.to_datetime(data[time_col])
        fraud_by_time = data.groupby(data[time_col].dt.date)[fraud_col].mean()
        fig.add_trace(
            go.Scatter(x=fraud_by_time.index, y=fraud_by_time.values,
                      mode='lines+markers', name='Fraud Rate Over Time'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Fraud Detection Dashboard")
    return fig


def calculate_fraud_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive fraud detection metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (Optional[np.ndarray]): Predicted probabilities
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, roc_auc_score, confusion_matrix)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics.update({
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    })
    
    return metrics


def print_fraud_analysis_summary(data: pd.DataFrame, 
                               fraud_col: str = 'class') -> None:
    """
    Print a comprehensive summary of fraud analysis.
    
    Args:
        data (pd.DataFrame): Dataset to analyze
        fraud_col (str): Name of the fraud column
    """
    print("=" * 60)
    print("FRAUD DETECTION ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Overview:")
    print(f"- Total records: {len(data):,}")
    print(f"- Total features: {len(data.columns)}")
    
    if fraud_col in data.columns:
        fraud_counts = data[fraud_col].value_counts()
        fraud_rate = fraud_counts[1] / len(data) if 1 in fraud_counts else 0
        
        print(f"\nFraud Analysis:")
        print(f"- Fraud cases: {fraud_counts.get(1, 0):,}")
        print(f"- Non-fraud cases: {fraud_counts.get(0, 0):,}")
        print(f"- Fraud rate: {fraud_rate:.3%}")
        
        if fraud_rate < 0.01:
            print("- ⚠️  Severe class imbalance detected!")
        elif fraud_rate < 0.1:
            print("- ⚠️  Class imbalance detected")
        else:
            print("- ✅ Balanced dataset")
    
    # Data quality check
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\nData Quality Issues:")
        print(f"- Missing values: {missing_data.sum():,}")
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  • {col}: {missing:,} ({missing/len(data):.1%})")
    else:
        print(f"\n✅ No missing values detected")
    
    print("\n" + "=" * 60)
