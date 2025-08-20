#!/usr/bin/env python3
"""
Simple script to run the complete Fraud Detection System

This script provides an easy way to start all components of the fraud detection
system with a single command.
"""

import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def print_banner():
    """Print a welcome banner."""
    print("=" * 80)
    print("🕵️  Enhanced Fraud Detection System")
    print("=" * 80)
    print("🚀 Starting all system components...")
    print("=" * 80)

def check_requirements():
    """Check if all required packages are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'streamlit', 'fastapi', 
        'uvicorn', 'plotly', 'matplotlib', 'xgboost', 'lightgbm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def start_api():
    """Start the FastAPI server."""
    print("🔌 Starting API server...")
    try:
        subprocess.run([
            'uvicorn', 'api.app:app', 
            '--host', '0.0.0.0', 
            '--port', '8000',
            '--reload'
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("🛑 API server stopped")

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("📊 Starting dashboard...")
    try:
        subprocess.run([
            'streamlit', 'run', 'dashboard/app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("🛑 Dashboard stopped")

def main():
    """Main function to run the complete system."""
    print_banner()
    
    # Check dependencies
    if not check_requirements():
        sys.exit(1)
    
    print("\n🎯 Starting system components...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("🔌 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("\n⏹️  Press Ctrl+C to stop all services")
    print("-" * 80)
    
    # Start API and dashboard in separate threads
    api_thread = threading.Thread(target=start_api, daemon=True)
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    
    try:
        # Start API first
        api_thread.start()
        time.sleep(3)  # Give API time to start
        
        # Start dashboard
        dashboard_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        print("✅ System stopped successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
