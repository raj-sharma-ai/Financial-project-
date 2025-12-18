"""
Application Configuration
Centralized configuration management
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# API Configuration
API_TITLE = "Financial Recommendation API"
API_VERSION = "1.0.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# LLM Configuration (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Scheduler Configuration
class SchedulerConfig:
    """Scheduler configuration"""
    BASE_DIR = BASE_DIR
    
    # Script paths
    STOCK_DATA_SCRIPT = str(SCRIPTS_DIR / "data_injection" / "stock_data_gathering.py")
    STOCK_VECTOR_SCRIPT = str(SCRIPTS_DIR / "data_injection" / "stock_vector.py")
    FUND_DATA_SCRIPT = str(SCRIPTS_DIR / "data_injection" / "mutual_fund_data_gathering.py")
    FUND_VECTOR_SCRIPT = str(SCRIPTS_DIR / "data_injection" / "mutualfund_vector.py")
    
    # Scheduling times
    STOCK_SCHEDULE_TIME = "02:00"  # Daily at 2 AM
    FUND_SCHEDULE_DAY = "sunday"   # Weekly on Sunday
    FUND_SCHEDULE_TIME = "03:00"   # Sunday at 3 AM
    
    ENABLED = True
    
    # Timeouts
    GATHERING_TIMEOUT = 7200  # 2 hours
    VECTOR_TIMEOUT = 1800     # 30 minutes

# File paths
USERS_CSV_PATH = DATA_DIR / "test_users_BANK.csv"
RISK_MODEL_PATH = MODELS_DIR / "risk_model.pkl"
STOCKS_JSON_PATH = DATA_DIR / "engineered_stocks.json"
FUNDS_JSON_PATH = DATA_DIR / "engineered_funds.json"
INSURANCE_JSON_PATH = DATA_DIR / "engineered_insurance.json"

# Insurance model paths
INSURANCE_MODEL_PATH = MODELS_DIR / "xgb_model_synthetic.pkl"
INSURANCE_SCALER_PATH = MODELS_DIR / "xgb_scaler_synthetic.pkl"
INSURANCE_ENCODER_PATH = MODELS_DIR / "xgb_label_encoder_synthetic.pkl"
INSURANCE_FEATURES_PATH = MODELS_DIR / "xgb_feature_names_synthetic.pkl"
HEALTH_POLICIES_PATH = DATA_DIR / "health_policies.json"

