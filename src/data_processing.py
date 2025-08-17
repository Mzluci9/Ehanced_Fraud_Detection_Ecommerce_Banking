"""
Enhanced Data Processing Module for Fraud Detection

This module provides comprehensive data preprocessing capabilities
specifically designed for fraud detection in e-commerce and banking
transactions. It includes methods for handling missing values, feature
engineering, and data validation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from typing import List, Optional, Dict, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.data = data.copy()
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        logger.info(
            f"Initialized DataPreprocessing with dataset shape: "
            f"{self.data.shape}"
        )

    def handle_missing_values(
        self, method: str = "drop", fill_value: Optional[Any] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            method (str): Method to handle missing values ('drop', 'fill',
                         'interpolate')
            fill_value: Value to fill missing data with (if method='fill')

        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        logger.info(f"Handling missing values using method: {method}")

        if method == "drop":
            initial_rows = len(self.data)
            self.data = self.data.dropna()
            dropped_rows = initial_rows - len(self.data)
            logger.info(f"Dropped {dropped_rows} rows with missing values")

        elif method == "fill":
            if fill_value is not None:
                self.data = self.data.fillna(fill_value)
            else:
                # Use mean for numeric, mode for categorical
                numeric_cols = self.data.select_dtypes(
                    include=[np.number]
                ).columns
                categorical_cols = self.data.select_dtypes(
                    include=["object"]
                ).columns

                if len(numeric_cols) > 0:
                    self.data[numeric_cols] = self.data[numeric_cols].fillna(
                        self.data[numeric_cols].mean()
                    )
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        mode_value = (
                            self.data[col].mode()[0]
                            if len(self.data[col].mode()) > 0
                            else "Unknown"
                        )
                        self.data[col] = self.data[col].fillna(mode_value)

        elif method == "interpolate":
            self.data = self.data.interpolate(method="linear")

        return self.data

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.

        Returns:
            pd.DataFrame: Dataset with duplicates removed
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed_rows = initial_rows - len(self.data)
        logger.info(f"Removed {removed_rows} duplicate rows")
        return self.data

    def correct_data_types(self) -> pd.DataFrame:
        """
        Convert data types to appropriate formats for fraud detection.

        Returns:
            pd.DataFrame: Dataset with corrected data types
        """
        # Convert datetime columns
        datetime_columns = ["signup_time", "purchase_time", "transaction_time"]
        for col in datetime_columns:
            if col in self.data.columns:
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")

        # Convert IP addresses to numeric if they're strings
        if "ip_address" in self.data.columns:
            if self.data["ip_address"].dtype == "object":
                try:
                    self.data["ip_address"] = pd.to_numeric(
                        self.data["ip_address"], errors="coerce"
                    )
                    logger.info("Converted ip_address to numeric")
                except Exception as e:
                    logger.warning(
                        f"Could not convert ip_address to numeric: {e}"
                    )

        return self.data

    def normalize_and_scale(self, columns: List[str]) -> pd.DataFrame:
        """
        Normalize and scale specified columns using StandardScaler.

        Args:
            columns (List[str]): List of column names to scale

        Returns:
            pd.DataFrame: Dataset with scaled columns
        """
        if not columns:
            logger.warning("No columns specified for scaling")
            return self.data

        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        self.scalers["standard"] = scaler
        logger.info(f"Scaled {len(columns)} columns using StandardScaler")
        return self.data

    def encode_categorical(
        self, columns: List[str], method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical variables using specified method.

        Args:
            columns (List[str]): List of categorical column names
            method (str): Encoding method ('onehot', 'label', 'frequency')

        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        if not columns:
            logger.warning("No categorical columns specified for encoding")
            return self.data

        if method == "onehot":
            self.data = pd.get_dummies(
                self.data, columns=columns, drop_first=True
            )
            logger.info(f"One-hot encoded {len(columns)} categorical columns")

        elif method == "label":
            for col in columns:
                if col in self.data.columns:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(
                        self.data[col].astype(str)
                    )
                    self.label_encoders[col] = le
            logger.info(f"Label encoded {len(columns)} categorical columns")

        elif method == "frequency":
            for col in columns:
                if col in self.data.columns:
                    freq_map = self.data[col].value_counts().to_dict()
                    self.data[col] = self.data[col].map(freq_map)
            logger.info(
                f"Frequency encoded {len(columns)} categorical columns"
            )

        return self.data

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
            self.data["source_count"] = self.data.groupby("source")[
                "source"
            ].transform("count")

        # Age-based features
        if "age" in self.data.columns:
            self.data["age_group"] = pd.cut(
                self.data["age"],
                bins=[0, 25, 35, 50, 100],
                labels=["18-25", "26-35", "36-50", "50+"],
            )

        logger.info("Fraud-specific features created successfully")
        return self.data

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
        }

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
        }

        return summary
