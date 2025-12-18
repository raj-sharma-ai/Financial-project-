"""
FastAPI Application Initialization
Enterprise-level application setup
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import json
import pickle
from app.core.config import (
    API_TITLE, API_VERSION, DATA_DIR, MODELS_DIR,
    USERS_CSV_PATH, RISK_MODEL_PATH, STOCKS_JSON_PATH,
    FUNDS_JSON_PATH, INSURANCE_JSON_PATH
)
from app.utils.helpers import add_log

# Initialize FastAPI app
app = FastAPI(title=API_TITLE, version=API_VERSION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data storage
users_df = None
model_data = None
stocks_data = None
funds_data = None
insurance_data = None
admin_logs = []

# Insurance model globals
insurance_model = None
insurance_scaler = None
insurance_label_encoder = None
insurance_feature_names = None
health_policies = None
insurance_users_df = None

# Scheduler status
scheduler_status = {
    "enabled": True,
    "last_run": None,
    "next_run": None,
    "is_running": False,
    "last_result": None
}


def load_initial_data():
    """Load all initial data on startup"""
    global users_df, model_data, stocks_data, funds_data, insurance_data
    
    # Load users CSV
    if USERS_CSV_PATH.exists():
        users_df = pd.read_csv(USERS_CSV_PATH)
        add_log(admin_logs, "STARTUP", "SUCCESS", f"Loaded {len(users_df)} users from CSV")
    else:
        users_df = pd.DataFrame()
        add_log(admin_logs, "STARTUP", "WARNING", "Users CSV not found")
    
    # Load risk model
    try:
        with open(RISK_MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        add_log(admin_logs, "STARTUP", "SUCCESS", "Risk model loaded")
    except Exception as e:
        model_data = None
        add_log(admin_logs, "STARTUP", "WARNING", f"Risk model failed to load: {str(e)}")
    
    # Load stocks JSON
    if STOCKS_JSON_PATH.exists():
        with open(STOCKS_JSON_PATH, 'r') as f:
            stocks_data = json.load(f)
        add_log(admin_logs, "STARTUP", "SUCCESS", f"Loaded {len(stocks_data)} stocks")
    else:
        stocks_data = []
        add_log(admin_logs, "STARTUP", "WARNING", "Stocks JSON not found")
    
    # Load funds JSON
    if FUNDS_JSON_PATH.exists():
        with open(FUNDS_JSON_PATH, 'r') as f:
            funds_data = json.load(f)
        add_log(admin_logs, "STARTUP", "SUCCESS", f"Loaded {len(funds_data)} funds")
    else:
        funds_data = []
        add_log(admin_logs, "STARTUP", "WARNING", "Funds JSON not found")
    
    # Load insurance JSON
    if INSURANCE_JSON_PATH.exists():
        with open(INSURANCE_JSON_PATH, 'r') as f:
            insurance_data = json.load(f)
        add_log(admin_logs, "STARTUP", "SUCCESS", f"Loaded {len(insurance_data)} insurance products")
    else:
        insurance_data = []
        add_log(admin_logs, "STARTUP", "WARNING", "Insurance JSON not found")


# Load data on module import
load_initial_data()

# Register all routes from separate route files
from app.api.routes import root, users, recommendations, insurance, admin, scheduler, llm

app.include_router(root.router)
app.include_router(users.router)
app.include_router(recommendations.router)
app.include_router(insurance.router)
app.include_router(admin.router)
app.include_router(scheduler.router)
app.include_router(llm.router)

# Import main.py to load helper functions (LLM, insurance, scheduler functions)
# These are kept in main.py as they're shared across routes
# import app.api.main

