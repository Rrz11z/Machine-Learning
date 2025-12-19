#!/usr/bin/env python3
"""
API Server Launcher with Complete Health Checks
Automatically loads model and starts server with best practices
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_requirements():
    """Verify all required files exist"""
    logger.info("🔍 Checking requirements...")
    
    required_files = [
        'main.py',
        'index.html',
        'admin.html',
        'real_drug_dataset.csv',
        'models/model.joblib',
        'models/encoders.joblib'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logger.error(f"❌ Missing required file: {file_path}")
            return False
        logger.info(f"✓ Found: {file_path}")
    
    logger.info("✅ All requirements satisfied\n")
    return True


def check_dependencies():
    """Verify Python packages are installed"""
    logger.info("📦 Checking dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'sklearn',
        'joblib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}")
        except ImportError:
            logger.error(f"❌ Missing package: {package}")
            logger.error("Run: pip install -r requirements.txt")
            return False
    
    logger.info("✅ All dependencies installed\n")
    return True


def print_startup_banner():
    """Print startup information"""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        💊 DRUG EFFECTIVENESS PREDICTOR v1.0.0            ║
║                                                           ║
║                   🚀 STARTING SERVER                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

📊 Server Configuration:
   • Host:     0.0.0.0
   • Port:     8000
   • Reload:   Enabled (development)
   • Debug:    True

🌐 Access URLs:
   • Main App:    http://localhost:8000
   • Admin Panel: http://localhost:8000/admin
   • API Docs:    http://localhost:8000/docs
   • ReDoc:       http://localhost:8000/redoc
   • Health:      http://localhost:8000/health

📚 Documentation:
   • README.md           - Full documentation
   • QUICKSTART.md       - Quick start guide
   • DEPLOYMENT.md       - Deployment guide
   • PRODUCTION_CONFIG.md - Production settings

⚠️  Press CTRL+C to stop the server
"""
    print(banner)


def main():
    """Main entry point"""
    try:
        # Verify everything is ready
        if not check_requirements():
            sys.exit(1)
        
        if not check_dependencies():
            sys.exit(1)
        
        # Print startup info
        print_startup_banner()
        
        # Import and run Uvicorn
        import uvicorn
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        host = os.getenv('API_HOST', '0.0.0.0')
        port = int(os.getenv('API_PORT', 8000))
        reload = os.getenv('API_RELOAD', 'True').lower() == 'true'
        
        logger.info(f"🔧 Starting Uvicorn server...\n")
        
        # Run server
        uvicorn.run(
            'main:app',
            host=host,
            port=port,
            reload=reload,
            log_level='info'
        )
        
    except KeyboardInterrupt:
        logger.info("\n\n❌ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
