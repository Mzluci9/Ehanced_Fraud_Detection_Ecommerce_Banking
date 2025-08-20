#!/usr/bin/env python3
"""
Deployment script for Fraud Detection System

This script handles the complete deployment pipeline including:
- Data preprocessing
- Model training and evaluation
- API deployment
- Dashboard deployment
- Testing and validation
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import CONFIG, get_model_path, get_data_path
from src.data_processing import DataPreprocessing
from src.model_training import FraudDetectionModel
from src.utils.utils import load_data, save_data


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, CONFIG['logging']['level']),
        format=CONFIG['logging']['format'],
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(CONFIG['logging']['file_handler']['filename'])
        ]
    )
    return logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger = logging.getLogger(__name__)
    logger.info("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm',
        'streamlit', 'fastapi', 'uvicorn', 'plotly', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Please install missing packages: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed.")
    return True


def preprocess_data(data_path: str = None) -> str:
    """
    Preprocess the data for model training.
    
    Args:
        data_path: Path to the raw data file
        
    Returns:
        Path to the processed data file
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    # Load data
    if data_path is None:
        # Try to find data file
        possible_paths = [
            get_data_path("raw", "Fraud_Data.csv"),
            get_data_path("raw", "creditcard.csv"),
            get_data_path("raw", "fraud_data.csv")
        ]
        
        for path in possible_paths:
            if path.exists():
                data_path = str(path)
                break
        else:
            logger.warning("No data file found, creating sample data...")
            return create_sample_data()
    
    try:
        data = load_data(data_path)
        logger.info(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return create_sample_data()
    
    # Initialize preprocessing
    dp = DataPreprocessing(data)
    
    # Apply preprocessing pipeline
    processed_data = (dp
                     .handle_missing_values(
                         method=CONFIG['data_processing']['missing_value_method'],
                         strategy=CONFIG['data_processing']['missing_value_strategy']
                     )
                     .remove_duplicates()
                     .correct_data_types()
                     .create_fraud_features())
    
    # Save processed data
    output_path = get_data_path("processed", "cleaned_fraud_data.csv")
    save_data(processed_data, str(output_path))
    
    logger.info(f"Data preprocessing completed. Saved to: {output_path}")
    return str(output_path)


def create_sample_data() -> str:
    """Create sample data for demonstration."""
    logger = logging.getLogger(__name__)
    logger.info("Creating sample fraud detection data...")
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'user_id': range(n_samples),
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
        'amount': np.random.exponential(100, n_samples),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'merchant_category': np.random.choice(['retail', 'online', 'travel', 'food'], n_samples),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'bank_transfer'], n_samples),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], n_samples),
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        'ip_address': [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Save sample data
    output_path = get_data_path("processed", "sample_fraud_data.csv")
    save_data(df, str(output_path))
    
    logger.info(f"Sample data created and saved to: {output_path}")
    return str(output_path)


def train_models(data_path: str) -> Dict[str, Any]:
    """
    Train fraud detection models.
    
    Args:
        data_path: Path to the processed data
        
    Returns:
        Dictionary containing training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Load processed data
    data = load_data(data_path)
    
    # Initialize model
    model = FraudDetectionModel(data, 'fraud')
    
    # Split data
    model.split_data(
        test_size=CONFIG['model_training']['test_size'],
        random_state=CONFIG['model_training']['random_state']
    )
    
    # Train baseline models
    results = model.train_baseline_models(
        balance_method=CONFIG['model_training']['class_balancing'],
        feature_selection=CONFIG['model_training']['feature_selection']
    )
    
    # Perform hyperparameter optimization
    best_model_result = model.hyperparameter_optimization(
        'XGBoost', 
        cv_folds=CONFIG['model_training']['cv_folds']
    )
    
    # Create ensemble model
    ensemble_result = model.create_ensemble_model(
        voting_method=CONFIG['model_training']['ensemble_method']
    )
    
    # Save best model
    if CONFIG['model_training']['save_best_model']:
        model.save_model('best', str(get_model_path('best_fraud_detector')))
    
    # Generate report
    report = model.generate_report()
    
    logger.info("Model training completed successfully!")
    logger.info(f"Best model F1 score: {report['best_model']['metrics']['f1_score']:.4f}")
    
    return {
        'results': results,
        'best_model': best_model_result,
        'ensemble': ensemble_result,
        'report': report
    }


def run_tests():
    """Run the test suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running tests...")
    
    try:
        result = subprocess.run(
            ['pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        if result.returncode == 0:
            logger.info("All tests passed!")
            return True
        else:
            logger.error(f"Tests failed: {result.stdout}")
            logger.error(f"Test errors: {result.stderr}")
            return False
            
    except FileNotFoundError:
        logger.error("pytest not found. Please install pytest: pip install pytest")
        return False


def start_api():
    """Start the FastAPI server."""
    logger = logging.getLogger(__name__)
    logger.info("Starting API server...")
    
    api_config = CONFIG['api']
    host = api_config['host']
    port = api_config['port']
    
    try:
        # Start the API server
        cmd = [
            'uvicorn', 'api.app:app',
            '--host', host,
            '--port', str(port),
            '--reload' if api_config['debug'] else '--workers', str(api_config['workers'])
        ]
        
        logger.info(f"Starting API server on {host}:{port}")
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"Error starting API server: {e}")


def start_dashboard():
    """Start the Streamlit dashboard."""
    logger = logging.getLogger(__name__)
    logger.info("Starting dashboard...")
    
    dashboard_config = CONFIG['dashboard']
    host = dashboard_config['host']
    port = dashboard_config['port']
    
    try:
        # Start the dashboard
        cmd = [
            'streamlit', 'run', 'dashboard/app.py',
            '--server.port', str(port),
            '--server.address', host
        ]
        
        logger.info(f"Starting dashboard on {host}:{port}")
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")


def validate_deployment():
    """Validate the deployment by running health checks."""
    logger = logging.getLogger(__name__)
    logger.info("Validating deployment...")
    
    import requests
    import time
    
    # Check API health
    api_config = CONFIG['api']
    api_url = f"http://{api_config['host']}:{api_config['port']}/health"
    
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            logger.info("API health check passed")
        else:
            logger.warning(f"API health check failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"API health check failed: {e}")
    
    # Check dashboard
    dashboard_config = CONFIG['dashboard']
    dashboard_url = f"http://{dashboard_config['host']}:{dashboard_config['port']}"
    
    try:
        response = requests.get(dashboard_url, timeout=10)
        if response.status_code == 200:
            logger.info("Dashboard health check passed")
        else:
            logger.warning(f"Dashboard health check failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Dashboard health check failed: {e}")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Fraud Detection System")
    parser.add_argument('--data', help='Path to raw data file')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--dashboard', action='store_true', help='Start dashboard')
    parser.add_argument('--deploy', action='store_true', help='Full deployment')
    parser.add_argument('--validate', action='store_true', help='Validate deployment')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Fraud Detection System deployment...")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    try:
        if args.deploy or args.train:
            # Preprocess data
            data_path = preprocess_data(args.data)
            
            # Train models
            training_results = train_models(data_path)
            
            # Save training report
            import json
            report_path = CONFIG['project_root'] / 'reports' / 'training_report.json'
            with open(report_path, 'w') as f:
                json.dump(training_results['report'], f, indent=2, default=str)
            
            logger.info(f"Training report saved to: {report_path}")
        
        if args.deploy or args.test:
            # Run tests
            if not run_tests():
                logger.error("Tests failed. Deployment aborted.")
                sys.exit(1)
        
        if args.deploy or args.api:
            # Start API server
            start_api()
        
        if args.deploy or args.dashboard:
            # Start dashboard
            start_dashboard()
        
        if args.deploy or args.validate:
            # Validate deployment
            validate_deployment()
        
        if not any([args.train, args.test, args.api, args.dashboard, args.deploy, args.validate]):
            # Default: full deployment
            logger.info("No specific action specified, running full deployment...")
            
            # Preprocess data
            data_path = preprocess_data(args.data)
            
            # Train models
            training_results = train_models(data_path)
            
            # Run tests
            if not run_tests():
                logger.error("Tests failed. Deployment aborted.")
                sys.exit(1)
            
            # Start services
            logger.info("Deployment completed successfully!")
            logger.info("Starting services...")
            
            # Start API and dashboard in separate processes
            import threading
            
            api_thread = threading.Thread(target=start_api)
            dashboard_thread = threading.Thread(target=start_dashboard)
            
            api_thread.start()
            time.sleep(2)  # Give API time to start
            
            dashboard_thread.start()
            
            # Wait for user interruption
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down services...")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
