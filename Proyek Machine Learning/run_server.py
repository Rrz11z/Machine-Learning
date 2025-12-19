#!/usr/bin/env python3
"""
Script to run the FastAPI server
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == '__main__':
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    reload = os.getenv('API_RELOAD', 'True').lower() == 'true'
    
    print(f"🚀 Starting server on {host}:{port}")
    print(f"📱 Open your browser to http://localhost:{port}")
    
    uvicorn.run(
        'main:app',
        host=host,
        port=port,
        reload=reload,
        log_level='info'
    )
