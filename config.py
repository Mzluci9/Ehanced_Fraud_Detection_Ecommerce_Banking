"""
Configuration file for Fraud Detection Project

This module contains all configuration settings for the fraud detection
system, including data paths, model parameters, API settings, and more.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Data processing configuration
DATA_PROCESSING_CONFIG = {
    "missing_value_method": "fill",  # 'drop', 'fill', 'interpolate', 'knn'
    "missing_value_strategy": "mean",  # 'mean', 'median', 'most_frequent'
    "knn_neighbors": 5,
    "scaling_method": "standard",  # 'standard', 'robust', 'minmax'
    "categorical_encoding": "onehot",  # 'onehot', 'label', 'frequency', 'target'
    "feature_selection_method": "kbest",  # 'kbest', 'rfe', 'none'
    "n_features_to_select": 50,
    "remove_duplicates": True,
    "correct_data_types": True,
    "create_fraud_features": True
}

# Model training configuration
MODEL_TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "class_balancing": "smote",  # 'smote', 'adasyn', 'undersample', 'none'
    "feature_selection": "kbest",
    "n_features": 50,
    "models_to_train": [
        "Random Forest",
        "XGBoost", 
        "LightGBM",
        "Gradient Boosting",
        "Logistic Regression"
    ],
    "ensemble_method": "soft",  # 'soft', 'hard'
    "save_best_model": True,
    "save_all_models": False
}

# Hyperparameter grids for optimization
HYPERPARAMETER_GRIDS = {
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5]
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.1, 0.2],
        "num_leaves": [31, 50, 100],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5]
    }
}

# Model evaluation configuration
EVALUATION_CONFIG = {
    "metrics": [
        "accuracy", "precision", "recall", "f1_score", 
        "roc_auc", "average_precision", "specificity"
    ],
    "threshold_optimization": True,
    "business_metrics": {
        "fraud_detection_rate": 0.95,  # Target fraud detection rate
        "false_positive_rate": 0.02,   # Maximum acceptable FPR
        "cost_matrix": {
            "true_negative": 0,      # Cost of correct non-fraud prediction
            "false_positive": 10,    # Cost of false alarm
            "false_negative": 100,   # Cost of missed fraud
            "true_positive": -5      # Benefit of correct fraud detection
        }
    }
}

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False,
    "workers": 4,
    "max_request_size": 100 * 1024 * 1024,  # 100MB
    "timeout": 30,
    "cors_origins": ["*"],
    "rate_limit": {
        "requests_per_minute": 100,
        "burst_size": 20
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "host": "localhost",
    "port": 8501,
    "debug": False,
    "theme": "light",
    "page_title": "Fraud Detection Dashboard",
    "page_icon": "🕵️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_handler": {
        "enabled": True,
        "filename": LOGS_DIR / "fraud_detection.log",
        "max_bytes": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5
    },
    "console_handler": {
        "enabled": True
    }
}

# Database configuration (for future use)
DATABASE_CONFIG = {
    "enabled": False,
    "type": "postgresql",  # 'postgresql', 'mysql', 'sqlite'
    "host": "localhost",
    "port": 5432,
    "database": "fraud_detection",
    "username": "fraud_user",
    "password": "secure_password",
    "pool_size": 10,
    "max_overflow": 20
}

# Monitoring and alerting configuration
MONITORING_CONFIG = {
    "enabled": True,
    "metrics_collection": True,
    "performance_thresholds": {
        "accuracy_min": 0.85,
        "f1_score_min": 0.80,
        "response_time_max": 1000,  # milliseconds
        "error_rate_max": 0.01
    },
    "alerts": {
        "email": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "",
            "password": "",
            "recipients": []
        },
        "slack": {
            "enabled": False,
            "webhook_url": "",
            "channel": "#fraud-alerts"
        }
    }
}

# Security configuration
SECURITY_CONFIG = {
    "api_key_required": True,
    "rate_limiting": True,
    "input_validation": True,
    "data_encryption": False,
    "allowed_file_types": [".csv", ".json", ".parquet"],
    "max_file_size": 50 * 1024 * 1024,  # 50MB
}

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "temporal_features": True,
    "geographic_features": True,
    "behavioral_features": True,
    "risk_scoring": True,
    "user_profiling": True,
    "transaction_patterns": True,
    "device_fingerprinting": True
}

# Model deployment configuration
DEPLOYMENT_CONFIG = {
    "model_versioning": True,
    "a_b_testing": False,
    "canary_deployment": False,
    "rollback_threshold": 0.05,  # Performance degradation threshold
    "model_refresh_frequency": "weekly",  # 'daily', 'weekly', 'monthly'
    "performance_monitoring": True
}

# Environment-specific configurations
ENVIRONMENTS = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "save_models": False,
        "use_sample_data": True
    },
    "testing": {
        "debug": False,
        "log_level": "INFO",
        "save_models": True,
        "use_sample_data": True
    },
    "production": {
        "debug": False,
        "log_level": "WARNING",
        "save_models": True,
        "use_sample_data": False,
        "security": {
            "api_key_required": True,
            "rate_limiting": True,
            "input_validation": True
        }
    }
}

# Get current environment
CURRENT_ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration for the current environment.
    
    Returns:
        Dictionary containing all configuration settings
    """
    config = {
        "project_root": PROJECT_ROOT,
        "data_processing": DATA_PROCESSING_CONFIG,
        "model_training": MODEL_TRAINING_CONFIG,
        "hyperparameter_grids": HYPERPARAMETER_GRIDS,
        "evaluation": EVALUATION_CONFIG,
        "api": API_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "logging": LOGGING_CONFIG,
        "database": DATABASE_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "security": SECURITY_CONFIG,
        "feature_engineering": FEATURE_ENGINEERING_CONFIG,
        "deployment": DEPLOYMENT_CONFIG
    }
    
    # Override with environment-specific settings
    if CURRENT_ENVIRONMENT in ENVIRONMENTS:
        env_config = ENVIRONMENTS[CURRENT_ENVIRONMENT]
        for key, value in env_config.items():
            if key in config:
                if isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    return config

def get_model_path(model_name: str) -> Path:
    """
    Get the path for a specific model file.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the model file
    """
    return MODELS_DIR / f"{model_name}.pkl"

def get_data_path(data_type: str, filename: str) -> Path:
    """
    Get the path for a specific data file.
    
    Args:
        data_type: Type of data ('raw' or 'processed')
        filename: Name of the file
        
    Returns:
        Path to the data file
    """
    if data_type == "raw":
        return RAW_DATA_DIR / filename
    elif data_type == "processed":
        return PROCESSED_DATA_DIR / filename
    else:
        raise ValueError(f"Invalid data type: {data_type}")

def get_log_path() -> Path:
    """
    Get the path for log files.
    
    Returns:
        Path to the log directory
    """
    return LOGS_DIR

def get_report_path() -> Path:
    """
    Get the path for report files.
    
    Returns:
        Path to the reports directory
    """
    return REPORTS_DIR

# Export main configuration
CONFIG = get_config()
