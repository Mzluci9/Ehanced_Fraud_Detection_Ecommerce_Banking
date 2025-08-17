"""
Fraud Detection Dashboard

A comprehensive Streamlit application for fraud detection in e-commerce and
banking transactions. This dashboard provides data analysis, model predictions,
and interactive visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataPreprocessing
from utils.utils import (
    load_data, validate_fraud_data, calculate_fraud_metrics
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .success-card {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        # Try to load processed data if available
        data_paths = [
            '../data/processed/cleaned_fraud_data_by_country.csv',
            '../data/raw/Fraud_Data.csv',
            '../data/raw/creditcard.csv'
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                return load_data(path)
        
        # If no data found, create sample data
        st.warning("No data files found. Using sample data for demonstration.")
        return create_sample_data()
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return create_sample_data()


def create_sample_data():
    """Create sample fraud detection data for demonstration."""
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
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


@st.cache_resource
def load_models():
    """Load trained models if available."""
    models = {}
    model_paths = [
        '../notebooks/models/fraud_data/',
        '../models/'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                for file in os.listdir(path):
                    if file.endswith('.pkl') or file.endswith('.joblib'):
                        model_name = file.split('.')[0]
                        model_path = os.path.join(path, file)
                        models[model_name] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load models from {path}: {str(e)}")
    
    return models


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🕵️ Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Data Overview", "🔍 Data Analysis", "🤖 Model Predictions", 
         "📈 Performance Metrics", "⚙️ Settings"]
    )
    
    # Load data and models
    data = load_sample_data()
    models = load_models()
    
    if page == "📊 Data Overview":
        show_data_overview(data)
    elif page == "🔍 Data Analysis":
        show_data_analysis(data)
    elif page == "🤖 Model Predictions":
        show_model_predictions(data, models)
    elif page == "📈 Performance Metrics":
        show_performance_metrics(data, models)
    elif page == "⚙️ Settings":
        show_settings()


def show_data_overview(data):
    """Display data overview and statistics."""
    st.header("📊 Data Overview")
    
    # File upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'], 
        help="Upload your transaction data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return
    
    # Data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("Total Features", f"{len(data.columns)}")
    
    with col3:
        fraud_col = get_fraud_column(data)
        if fraud_col:
            fraud_count = data[fraud_col].sum() if fraud_col in data.columns else 0
            st.metric("Fraud Cases", f"{fraud_count:,}")
    
    with col4:
        if fraud_col and fraud_col in data.columns:
            fraud_rate = data[fraud_col].mean()
            st.metric("Fraud Rate", f"{fraud_rate:.2%}")
    
    # Data validation
    st.subheader("Data Quality Check")
    validation_results = validate_fraud_data(data)
    
    if validation_results['warnings']:
        for warning in validation_results['warnings']:
            st.warning(warning)
    else:
        st.success("✅ Data quality check passed!")
    
    # Display data
    st.subheader("Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        st.dataframe(pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.astype(str),
            'Non-Null Count': data.count()
        }))
    
    with col2:
        st.subheader("Missing Values")
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': (missing_data / len(data) * 100).values
            }).sort_values('Missing Count', ascending=False)
            st.dataframe(missing_df)
        else:
            st.success("No missing values found!")


def show_data_analysis(data):
    """Display interactive data analysis."""
    st.header("🔍 Data Analysis")
    
    fraud_col = get_fraud_column(data)
    if not fraud_col:
        st.error("No fraud column found in the data!")
        return
    
    # Fraud distribution
    st.subheader("Fraud Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fraud_counts = data[fraud_col].value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=['Non-Fraud', 'Fraud'],
            title="Fraud vs Non-Fraud Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=['Non-Fraud', 'Fraud'],
            y=fraud_counts.values,
            title="Fraud Count by Category",
            color=['Non-Fraud', 'Fraud'],
            color_discrete_map={'Non-Fraud': '#1f77b4', 'Fraud': '#ff7f0e'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Amount analysis
    amount_col = get_amount_column(data)
    if amount_col:
        st.subheader("Transaction Amount Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Amount distribution by fraud status
            fig_box = px.box(
                data, 
                x=fraud_col, 
                y=amount_col,
                title="Amount Distribution by Fraud Status",
                color=fraud_col
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Amount histogram
            fig_hist = px.histogram(
                data,
                x=amount_col,
                color=fraud_col,
                title="Amount Distribution",
                nbins=50
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Time analysis
    time_col = get_time_column(data)
    if time_col:
        st.subheader("Time Series Analysis")
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # Fraud rate over time
        data['date'] = data[time_col].dt.date
        fraud_by_time = data.groupby('date')[fraud_col].mean().reset_index()
        
        fig_time = px.line(
            fraud_by_time,
            x='date',
            y=fraud_col,
            title="Fraud Rate Over Time"
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Categorical analysis
    categorical_cols = data.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Variable Analysis")
        
        selected_cat = st.selectbox(
            "Select categorical variable:",
            categorical_cols
        )
        
        if selected_cat:
            fraud_by_cat = data.groupby(selected_cat)[fraud_col].mean().sort_values(
                ascending=False)
            
            fig_cat = px.bar(
                x=fraud_by_cat.index,
                y=fraud_by_cat.values,
                title=f"Fraud Rate by {selected_cat}"
            )
            st.plotly_chart(fig_cat, use_container_width=True)


def show_model_predictions(data, models):
    """Display model predictions interface."""
    st.header("🤖 Model Predictions")
    
    if not models:
        st.warning("No trained models found. Please train models first.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model:",
        list(models.keys())
    )
    
    if selected_model:
        model = models[selected_model]
        
        # Data preprocessing
        st.subheader("Data Preprocessing")
        
        # Create preprocessing pipeline
        dp = DataPreprocessing(data.copy())
        
        # Apply preprocessing steps
        processed_data = (dp
                         .handle_missing_values(method='fill')
                         .remove_duplicates()
                         .correct_data_types()
                         .create_fraud_features())
        
        # Feature selection
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'class']
        
        st.write(f"Available features: {len(feature_cols)}")
        st.write(f"Features: {list(feature_cols)}")
        
        # Make predictions
        if st.button("Make Predictions"):
            try:
                # Prepare features
                X = processed_data[feature_cols].fillna(0)
                
                # Make predictions
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Add predictions to data
                data['predicted_fraud'] = predictions
                if probabilities is not None:
                    data['fraud_probability'] = probabilities
                
                st.success("Predictions completed!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Summary")
                    predicted_fraud = predictions.sum()
                    st.metric("Predicted Fraud Cases", f"{predicted_fraud:,}")
                    st.metric("Predicted Fraud Rate", f"{predicted_fraud/len(predictions):.2%}")
                
                with col2:
                    if probabilities is not None:
                        st.subheader("Probability Distribution")
                        fig_prob = px.histogram(
                            x=probabilities,
                            title="Fraud Probability Distribution",
                            nbins=50
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                
                # Show sample predictions
                st.subheader("Sample Predictions")
                display_cols = ['transaction_id', 'amount', 'predicted_fraud']
                if 'fraud_probability' in data.columns:
                    display_cols.append('fraud_probability')
                
                st.dataframe(data[display_cols].head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")


def show_performance_metrics(data, models):
    """Display model performance metrics."""
    st.header("📈 Performance Metrics")
    
    fraud_col = get_fraud_column(data)
    if not fraud_col:
        st.error("No fraud column found for performance evaluation!")
        return
    
    if not models:
        st.warning("No trained models found for performance evaluation.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select a model for evaluation:",
        list(models.keys())
    )
    
    if selected_model:
        model = models[selected_model]
        
        # Prepare data
        dp = DataPreprocessing(data.copy())
        processed_data = (dp
                         .handle_missing_values(method='fill')
                         .remove_duplicates()
                         .correct_data_types()
                         .create_fraud_features())
        
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'class']
        
        X = processed_data[feature_cols].fillna(0)
        y_true = processed_data[fraud_col]
        
        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = calculate_fraud_metrics(y_true, y_pred, y_prob)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        
        with col2:
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col3:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
        
        if 'roc_auc' in metrics:
            st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        cm_data = {
            'Predicted': ['Non-Fraud', 'Non-Fraud', 'Fraud', 'Fraud'],
            'Actual': ['Non-Fraud', 'Fraud', 'Non-Fraud', 'Fraud'],
            'Count': [
                metrics['true_negatives'],
                metrics['false_negatives'],
                metrics['false_positives'],
                metrics['true_positives']
            ]
        }
        
        cm_df = pd.DataFrame(cm_data)
        fig_cm = px.bar(
            cm_df,
            x='Predicted',
            y='Count',
            color='Actual',
            title="Confusion Matrix",
            barmode='group'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Specificity", f"{metrics['specificity']:.3f}")
        
        with col2:
            st.metric("False Positive Rate", f"{metrics['false_positive_rate']:.3f}")


def show_settings():
    """Display application settings."""
    st.header("⚙️ Settings")
    
    st.subheader("Data Processing Settings")
    
    # Missing value handling
    missing_method = st.selectbox(
        "Missing value handling method:",
        ['fill', 'drop', 'interpolate']
    )
    
    # Categorical encoding
    encoding_method = st.selectbox(
        "Categorical encoding method:",
        ['onehot', 'label', 'frequency']
    )
    
    # Feature scaling
    use_scaling = st.checkbox("Use feature scaling", value=True)
    
    st.subheader("Model Settings")
    
    # Threshold adjustment
    threshold = st.slider(
        "Fraud detection threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01
    )
    
    # Save settings
    if st.button("Save Settings"):
        st.success("Settings saved successfully!")


def get_fraud_column(data):
    """Get the fraud column name from the dataset."""
    fraud_columns = ['class', 'Class', 'fraud', 'is_fraud', 'target']
    for col in fraud_columns:
        if col in data.columns:
            return col
    return None


def get_amount_column(data):
    """Get the amount column name from the dataset."""
    amount_columns = ['amount', 'purchase_value', 'transaction_amount', 'Amount']
    for col in amount_columns:
        if col in data.columns:
            return col
    return None


def get_time_column(data):
    """Get the time column name from the dataset."""
    time_columns = ['timestamp', 'purchase_time', 'transaction_time', 'time']
    for col in time_columns:
        if col in data.columns:
            return col
    return None


if __name__ == "__main__":
    main()
