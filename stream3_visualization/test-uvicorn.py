#!/usr/bin/env python3
"""
Test script for Uvicorn deployment
Run this to test if the dashboard works with Uvicorn locally
"""

import uvicorn
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ASGI app from dashboard
from Budget.code.dashboard import asgi_app

if __name__ == "__main__":
    print("🚀 Starting WSL Dashboard with Uvicorn...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("🔍 API endpoints:")
    print("   - Health: http://localhost:8000/api/health")
    print("   - Scenarios: http://localhost:8000/api/data/scenarios")
    print("   - Countries: http://localhost:8000/api/data/countries")
    print("\nPress Ctrl+C to stop the server")
    
    # Run with Uvicorn
    uvicorn.run(
        asgi_app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker for local testing
        log_level="info"
    ) 