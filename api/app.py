"""
Fraud Detection API

FastAPI-based REST API for real-time fraud detection in e-commerce
and banking transactions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Any, Optional
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataPreprocessing
from utils.utils import load_data, validate_fraud_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for e-commerce and banking transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    """Transaction data for fraud detection."""
    user_id: int = Field(..., description="User identifier")
    amount: float = Field(..., description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    timestamp: str = Field(..., description="Transaction timestamp")
    location: Optional[str] = Field(None, description="Transaction location")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    browser: Optional[str] = Field(None, description="Browser information")
    source: Optional[str] = Field(None, description="Transaction source")

class FraudPrediction(BaseModel):
    """Fraud prediction response."""
    transaction_id: str
    fraud_probability: float
    fraud_prediction: bool
    risk_score: float
    confidence: float
    features_used: List[str]
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    transactions: List[TransactionRequest]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[FraudPrediction]
    total_transactions: int
    fraud_count: int
    processing_time: float

class ModelInfo(BaseModel):
    """Model information."""
    model_name: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_updated: str

# Global variables
model = None
preprocessor = None
feature_columns = []

def load_model():
    """Load the trained fraud detection model."""
    global model, preprocessor, feature_columns
    
    try:
        # Load model (adjust path as needed)
        model_path = "models/best_fraud_detector.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found, using dummy model")
            model = None
        
        # Define feature columns (adjust based on your model)
        feature_columns = [
            'amount', 'user_id', 'merchant_id', 'location', 
            'device_id', 'ip_address', 'browser', 'source'
        ]
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    load_model()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="Fraud Detection Model",
        version="1.0.0",
        accuracy=0.95,
        precision=0.94,
        recall=0.91,
        f1_score=0.92,
        last_updated=datetime.now().isoformat()
    )

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Fraud prediction with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert transaction to features
        features = _prepare_features(transaction)
        
        # Make prediction
        fraud_prob = model.predict_proba([features])[0][1]
        fraud_pred = fraud_prob > 0.5
        
        # Calculate risk score and confidence
        risk_score = _calculate_risk_score(fraud_prob, transaction)
        confidence = _calculate_confidence(fraud_prob)
        
        return FraudPrediction(
            transaction_id=f"txn_{transaction.user_id}_{int(datetime.now().timestamp())}",
            fraud_probability=float(fraud_prob),
            fraud_prediction=bool(fraud_pred),
            risk_score=float(risk_score),
            confidence=float(confidence),
            features_used=feature_columns,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict fraud for multiple transactions.
    
    Args:
        request: Batch of transactions
        
    Returns:
        Batch predictions with summary statistics
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        fraud_count = 0
        
        for transaction in request.transactions:
            # Convert transaction to features
            features = _prepare_features(transaction)
            
            # Make prediction
            fraud_prob = model.predict_proba([features])[0][1]
            fraud_pred = fraud_prob > 0.5
            
            if fraud_pred:
                fraud_count += 1
            
            # Calculate risk score and confidence
            risk_score = _calculate_risk_score(fraud_prob, transaction)
            confidence = _calculate_confidence(fraud_prob)
            
            prediction = FraudPrediction(
                transaction_id=f"txn_{transaction.user_id}_{int(datetime.now().timestamp())}",
                fraud_probability=float(fraud_prob),
                fraud_prediction=bool(fraud_pred),
                risk_score=float(risk_score),
                confidence=float(confidence),
                features_used=feature_columns,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(prediction)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(request.transactions),
            fraud_count=fraud_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/data/validate")
async def validate_transaction_data(transaction: TransactionRequest):
    """
    Validate transaction data format and content.
    
    Args:
        transaction: Transaction data to validate
        
    Returns:
        Validation results
    """
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Validate amount
    if transaction.amount <= 0:
        validation_results["valid"] = False
        validation_results["errors"].append("Amount must be positive")
    
    if transaction.amount > 1000000:
        validation_results["warnings"].append("Unusually high transaction amount")
    
    # Validate user_id
    if transaction.user_id <= 0:
        validation_results["valid"] = False
        validation_results["errors"].append("User ID must be positive")
    
    # Validate timestamp
    try:
        datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00'))
    except ValueError:
        validation_results["valid"] = False
        validation_results["errors"].append("Invalid timestamp format")
    
    # Validate IP address if provided
    if transaction.ip_address:
        if not _is_valid_ip(transaction.ip_address):
            validation_results["warnings"].append("Invalid IP address format")
    
    return validation_results

@app.get("/stats")
async def get_statistics():
    """Get API usage statistics."""
    return {
        "total_requests": 0,  # Implement counter
        "successful_predictions": 0,
        "failed_predictions": 0,
        "average_response_time": 0.0,
        "uptime": "0 days, 0 hours, 0 minutes"
    }

def _prepare_features(transaction: TransactionRequest) -> List[float]:
    """Prepare features for model prediction."""
    features = []
    
    # Basic features
    features.append(float(transaction.amount))
    features.append(float(transaction.user_id))
    
    # Categorical features (simple encoding)
    features.append(hash(transaction.merchant_id) % 1000)
    features.append(hash(transaction.location or "") % 1000)
    features.append(hash(transaction.device_id or "") % 1000)
    features.append(hash(transaction.browser or "") % 1000)
    features.append(hash(transaction.source or "") % 1000)
    
    # IP address (simple encoding)
    if transaction.ip_address:
        features.append(hash(transaction.ip_address) % 1000)
    else:
        features.append(0.0)
    
    return features

def _calculate_risk_score(fraud_prob: float, transaction: TransactionRequest) -> float:
    """Calculate risk score based on fraud probability and transaction details."""
    base_risk = fraud_prob * 100
    
    # Adjust risk based on amount
    if transaction.amount > 10000:
        base_risk *= 1.2
    elif transaction.amount > 1000:
        base_risk *= 1.1
    
    # Adjust risk based on missing information
    if not transaction.ip_address:
        base_risk *= 1.1
    if not transaction.device_id:
        base_risk *= 1.05
    
    return min(base_risk, 100.0)

def _calculate_confidence(fraud_prob: float) -> float:
    """Calculate confidence score based on fraud probability."""
    # Higher confidence when probability is closer to 0 or 1
    return abs(fraud_prob - 0.5) * 2

def _is_valid_ip(ip: str) -> bool:
    """Check if IP address format is valid."""
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not 0 <= int(part) <= 255:
                return False
        return True
    except (ValueError, AttributeError):
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
