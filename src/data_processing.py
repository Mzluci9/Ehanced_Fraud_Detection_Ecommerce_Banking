"""
Enhanced Data Processing Module for Fraud Detection

This module provides comprehensive data preprocessing capabilities
specifically designed for fraud detection in e-commerce and banking
transactions. It includes methods for handling missing values, feature
engineering, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from typing import List, Optional, Dict, Any
import warnings
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataPreprocessing:
    """
    Enhanced data preprocessing class for fraud detection datasets.

    This class provides methods to clean, transform, and prepare data for
    machine learning models in fraud detection scenarios.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataPreprocessing class.

        Args:
            data (pd.DataFrame): Input dataset to be processed
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty")

        self.data = data.copy()
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.feature_importance = {}
        self.processing_history = []

        logger.info(
            f"Initialized DataPreprocessing with dataset shape: "
            f"{self.data.shape}"
        )

    def handle_missing_values(
        self,
        method: str = "drop",
        fill_value: Optional[Any] = None,
        strategy: str = "mean",
        k_neighbors: int = 5,
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset with advanced strategies.

        Args:
            method (str): Method to handle missing values
                ('drop', 'fill', 'interpolate', 'knn')
            fill_value: Value to fill missing data with (if method='fill')
            strategy (str): Strategy for imputation
                ('mean', 'median', 'most_frequent')
            k_neighbors (int): Number of neighbors for KNN imputation

        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        logger.info(f"Handling missing values using method: {method}")

        if method == "drop":
            initial_rows = len(self.data)
            self.data = self.data.dropna()
            dropped_rows = initial_rows - len(self.data)
            logger.info(
                f"Dropped {dropped_rows} rows with missing values"
            )
            self.processing_history.append(
                f"Dropped {dropped_rows} rows with missing values"
            )

        elif method == "fill":
            if fill_value is not None:
                self.data = self.data.fillna(fill_value)
                logger.info(
                    f"Filled missing values with: {fill_value}"
                )
            else:
                # Use advanced imputation strategies
                numeric_cols = self.data.select_dtypes(
                    include=[np.number]
                ).columns
                categorical_cols = self.data.select_dtypes(
                    include=["object"]
                ).columns

                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy=strategy)
                    self.data[numeric_cols] = imputer.fit_transform(
                        self.data[numeric_cols]
                    )
                    self.imputers['numeric'] = imputer
                    logger.info(
                        "Applied %s imputation to %d numeric columns",
                        strategy,
                        len(numeric_cols),
                    )

                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        mode_value = (
                            self.data[col].mode()[0]
                            if len(self.data[col].mode()) > 0
                            else "Unknown"
                        )
                        self.data[col] = self.data[col].fillna(mode_value)
                    logger.info(
                        "Applied mode imputation to %d categorical columns",
                        len(categorical_cols),
                    )

        elif method == "knn":
            # Use KNN imputation for better accuracy
            imputer = KNNImputer(n_neighbors=k_neighbors)
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.data[numeric_cols] = imputer.fit_transform(
                    self.data[numeric_cols]
                )
                self.imputers['knn'] = imputer
                logger.info(
                    "Applied KNN imputation to %d numeric columns",
                    len(numeric_cols),
                )

        elif method == "interpolate":
            self.data = self.data.interpolate(method="linear")
            logger.info("Applied linear interpolation to missing values")

        else:
            logger.warning(
                f"Unknown method '{method}', using drop as fallback"
            )
            self.data = self.data.dropna()

        return self.data

    def remove_duplicates(
        self, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Args:
            subset (Optional[List[str]]): Columns to consider for
                duplicate detection

        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset)
        removed_rows = initial_rows - len(self.data)
        logger.info(f"Removed {removed_rows} duplicate rows")
        self.processing_history.append(
            f"Removed {removed_rows} duplicate rows"
        )
        return self.data

    def correct_data_types(self) -> pd.DataFrame:
        """
        Convert data types to appropriate formats for fraud detection.

        Returns:
            pd.DataFrame: Dataset with corrected data types
        """
        # Convert datetime columns
        datetime_columns = [
            "signup_time",
            "purchase_time",
            "transaction_time",
            "timestamp",
        ]
        for col in datetime_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_datetime(
                        self.data[col], errors='coerce'
                    )
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")

        # Convert IP addresses to numeric if they're strings
        if "ip_address" in self.data.columns:
            if self.data["ip_address"].dtype == "object":
                try:
                    # Handle IP addresses more robustly
                    self.data["ip_address"] = self.data["ip_address"].apply(
                        lambda x: (
                            self._ip_to_numeric(x) if pd.notna(x) else np.nan
                        )
                    )
                    logger.info("Converted ip_address to numeric")
                except Exception as e:
                    logger.warning(
                        f"Could not convert ip_address to numeric: {e}"
                    )

        # Convert boolean columns
        boolean_columns = ["is_fraud", "fraud", "class"]
        for col in boolean_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = self.data[col].astype(int)
                    logger.info(f"Converted {col} to integer")
                except Exception as e:
                    logger.warning(
                        f"Could not convert {col} to integer: {e}"
                    )

        return self.data

    def _ip_to_numeric(self, ip_str: str) -> int:
        """Convert IP address string to numeric value."""
        try:
            if pd.isna(ip_str):
                return np.nan

            # Handle different IP formats
            if isinstance(ip_str, (int, float)):
                return int(ip_str)

            # Remove any non-numeric characters except dots
            ip_str = re.sub(r'[^\d.]', '', str(ip_str))

            # Split by dots and convert to numeric
            parts = ip_str.split('.')
            if len(parts) == 4:
                return sum(
                    int(part) * (256 ** (3 - i))
                    for i, part in enumerate(parts)
                )
            else:
                return int(ip_str) if ip_str.isdigit() else np.nan
        except Exception:
            return np.nan

    def normalize_and_scale(
        self, columns: List[str], method: str = "standard"
    ) -> pd.DataFrame:
        """
        Normalize and scale specified columns using various scaling methods.

        Args:
            columns (List[str]): List of column names to scale
            method (str): Scaling method ('standard', 'robust',
                'minmax')

        Returns:
            pd.DataFrame: Dataset with scaled columns
        """
        if not columns:
            logger.warning("No columns specified for scaling")
            return self.data

        if method == "standard":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            logger.warning(
                f"Unknown scaling method '{method}', using standard"
            )
        scaler = StandardScaler()

        # Only scale columns that exist and are numeric
        valid_columns = [
            col
            for col in columns
            if (
                col in self.data.columns
                and pd.api.types.is_numeric_dtype(self.data[col])
            )
        ]

        if not valid_columns:
            logger.warning("No valid numeric columns found for scaling")
            return self.data

        self.data[valid_columns] = scaler.fit_transform(
            self.data[valid_columns]
        )
        self.scalers[method] = scaler
        logger.info(
            "Scaled %d columns using %s scaler",
            len(valid_columns),
            method,
        )
        self.processing_history.append(
            f"Scaled {len(valid_columns)} columns using {method}"
        )
        return self.data

    def encode_categorical(
        self, columns: List[str], method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical variables using specified method.

        Args:
            columns (List[str]): List of categorical column names
            method (str): Encoding method ('onehot', 'label',
                'frequency', 'target')

        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        if not columns:
            logger.warning("No categorical columns specified for encoding")
            return self.data

        # Only encode columns that exist
        valid_columns = [col for col in columns if col in self.data.columns]
        if not valid_columns:
            logger.warning("No valid categorical columns found for encoding")
            return self.data

        if method == "onehot":
            self.data = pd.get_dummies(
                self.data, columns=valid_columns, drop_first=True
            )
            logger.info(
                "One-hot encoded %d categorical columns",
                len(valid_columns),
            )

        elif method == "label":
            for col in valid_columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(
                    self.data[col].astype(str)
                )
                self.label_encoders[col] = le
            logger.info(
                "Label encoded %d categorical columns",
                len(valid_columns),
            )

        elif method == "frequency":
            for col in valid_columns:
                freq_map = self.data[col].value_counts().to_dict()
                self.data[col] = self.data[col].map(freq_map)
            logger.info(
                "Frequency encoded %d categorical columns",
                len(valid_columns),
            )

        elif method == "target":
            # Target encoding for high-cardinality categorical variables
            fraud_col = self._get_fraud_column()
            if fraud_col:
                for col in valid_columns:
                    target_means = self.data.groupby(col)[fraud_col].mean()
                    self.data[f"{col}_target_encoded"] = self.data[col].map(
                        target_means
                    )
                logger.info(
                    "Target encoded %d categorical columns",
                    len(valid_columns),
                )

        self.processing_history.append(
            f"Encoded {len(valid_columns)} columns using {method}"
        )
        return self.data

    def _get_fraud_column(self) -> Optional[str]:
        """Get the fraud column name from the dataset."""
        fraud_columns = ['class', 'Class', 'fraud', 'is_fraud', 'target']
        for col in fraud_columns:
            if col in self.data.columns:
                return col
        return None

    def create_fraud_features(self) -> pd.DataFrame:
        """
        Create fraud-specific features for enhanced detection.

        Returns:
            pd.DataFrame: Dataset with additional fraud detection features
        """
        logger.info("Creating fraud-specific features")

        # Time-based features
        if (
            "signup_time" in self.data.columns
            and "purchase_time" in self.data.columns
        ):
            self.data["signup_time"] = pd.to_datetime(self.data["signup_time"])
            self.data["purchase_time"] = pd.to_datetime(
                self.data["purchase_time"]
            )

            # Time difference between signup and purchase
            self.data["time_to_purchase_hours"] = (
                self.data["purchase_time"] - self.data["signup_time"]
            ).dt.total_seconds() / 3600

            # Hour of day features
            self.data["signup_hour"] = self.data["signup_time"].dt.hour
            self.data["purchase_hour"] = self.data["purchase_time"].dt.hour

            # Day of week features
            self.data["signup_day_of_week"] = self.data[
                "signup_time"
            ].dt.dayofweek
            self.data["purchase_day_of_week"] = self.data[
                "purchase_time"
            ].dt.dayofweek

            # Month features
            self.data["signup_month"] = self.data["signup_time"].dt.month
            self.data["purchase_month"] = self.data["purchase_time"].dt.month

            # Weekend vs weekday
            self.data["signup_is_weekend"] = (
                self.data["signup_time"].dt.dayofweek >= 5
            )
            self.data["purchase_is_weekend"] = (
                self.data["purchase_time"].dt.dayofweek >= 5
            )

        # Device and browser features
        if "device_id" in self.data.columns:
            self.data["device_id_count"] = self.data.groupby("device_id")[
                "device_id"
            ].transform("count")

        if "browser" in self.data.columns:
            self.data["browser_count"] = self.data.groupby("browser")[
                "browser"
            ].transform("count")

        # Source features
        if "source" in self.data.columns:
            self.data["source_count"] = (
                self.data.groupby("source")["source"].transform("count")
            )

        # Age-based features
        if "age" in self.data.columns:
            self.data["age_group"] = pd.cut(
                self.data["age"],
                bins=[0, 25, 35, 50, 100],
                labels=["18-25", "26-35", "36-50", "50+"],
            )

        # Amount-based features
        amount_cols = ['amount', 'purchase_value', 'transaction_amount']
        for col in amount_cols:
            if col in self.data.columns:
                self.data[f"{col}_log"] = np.log1p(self.data[col])
                self.data[f"{col}_squared"] = self.data[col] ** 2
                break

        # User behavior features
        if "user_id" in self.data.columns:
            user_stats = (
                self.data.groupby("user_id").agg(
                    {
                        'amount': ['count', 'mean', 'std', 'min', 'max'],
                        'purchase_time': ['min', 'max'],
                    }
                ).reset_index()
            )

            user_stats.columns = [
                'user_id',
                'user_transaction_count',
                'user_avg_amount',
                'user_amount_std',
                'user_min_amount',
                'user_max_amount',
                'user_first_purchase',
                'user_last_purchase',
            ]

            self.data = self.data.merge(user_stats, on='user_id', how='left')

        # Risk scoring features
        self._create_risk_features()

        logger.info("Fraud-specific features created successfully")
        self.processing_history.append("Created fraud-specific features")
        return self.data

    def _create_risk_features(self):
        """Create risk-based features for fraud detection."""
        # High-value transaction flag
        amount_cols = ['amount', 'purchase_value', 'transaction_amount']
        for col in amount_cols:
            if col in self.data.columns:
                threshold = self.data[col].quantile(0.95)
                self.data[f"{col}_high_value"] = self.data[col] > threshold
                break

        # Unusual time patterns
        if "purchase_hour" in self.data.columns:
            self.data["unusual_hour"] = (
                (self.data["purchase_hour"] < 6)
                | (self.data["purchase_hour"] > 23)
            )

        # Rapid successive transactions
        if (
            "user_id" in self.data.columns
            and "purchase_time" in self.data.columns
        ):
            self.data = self.data.sort_values(['user_id', 'purchase_time'])
            self.data['time_since_last_transaction'] = (
                self.data.groupby('user_id')['purchase_time']
                .diff()
                .dt.total_seconds()
            )
            # 5 minutes threshold
            self.data['rapid_transaction'] = (
                self.data['time_since_last_transaction'] < 300
            )

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the dataset for common data quality issues.

        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "duplicate_rows": self.data.duplicated().sum(),
            "data_types": self.data.dtypes.to_dict(),
            "warnings": [],
            "recommendations": []
        }

        # Check for fraud column
        fraud_col = self._get_fraud_column()
        if fraud_col:
            fraud_rate = self.data[fraud_col].value_counts(normalize=True)
            validation_results['fraud_rate'] = fraud_rate.to_dict()

            # Check for class imbalance
            if len(fraud_rate) == 2:
                min_class_rate = fraud_rate.min()
                if min_class_rate < 0.01:
                    validation_results['warnings'].append(
                        (
                            f"Severe class imbalance detected: "
                            f"{min_class_rate:.3f}"
                        )
                    )
                    validation_results['recommendations'].append(
                        "Consider using SMOTE or other balancing techniques"
                    )
                elif min_class_rate < 0.1:
                    validation_results['warnings'].append(
                        f"Class imbalance detected: {min_class_rate:.3f}"
                    )
        else:
            validation_results['warnings'].append("No fraud column found")

        # Check for outliers
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (
                    (
                        self.data[col] < (Q1 - 1.5 * IQR)
                    )
                    | (
                        self.data[col] > (Q3 + 1.5 * IQR)
                    )
                ).sum()
                if outliers > 0:
                    validation_results['warnings'].append(
                        f"Outliers detected in {col}: {outliers} values"
                    )

        logger.info("Data validation completed")
        return validation_results

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all preprocessing steps applied.

        Returns:
            Dict[str, Any]: Summary of preprocessing operations
        """
        summary = {
            "final_shape": self.data.shape,
            "scalers_applied": list(self.scalers.keys()),
            "label_encoders_applied": list(self.label_encoders.keys()),
            "imputers_applied": list(self.imputers.keys()),
            "processing_history": self.processing_history,
            "feature_count": len(self.data.columns)
        }

        return summary

    def save_preprocessing_pipeline(self, filepath: str) -> None:
        """
        Save the preprocessing pipeline for later use.

        Args:
            filepath (str): Path to save the pipeline
        """
        import joblib

        pipeline = {
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'imputers': self.imputers,
            'feature_columns': list(self.data.columns),
        }

        joblib.dump(pipeline, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")

    def load_preprocessing_pipeline(self, filepath: str) -> None:
        """
        Load a saved preprocessing pipeline.

        Args:
            filepath (str): Path to the saved pipeline
        """
        import joblib

        pipeline = joblib.load(filepath)
        self.scalers = pipeline['scalers']
        self.label_encoders = pipeline['label_encoders']
        self.imputers = pipeline['imputers']
        logger.info(f"Preprocessing pipeline loaded from {filepath}")
