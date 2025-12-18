"""
Admin Routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
import pandas as pd
import json
from app.api.dependencies import users_df, model_data, stocks_data, funds_data, admin_logs
from app.core.config import DATA_DIR
from app.utils.helpers import add_log
import os

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/logs")
async def get_admin_logs():
    """Get admin logs"""
    return {
        "total_logs": len(admin_logs),
        "logs": admin_logs
    }


@router.post("/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """Refresh data from files"""
    try:
        global users_df, model_data, stocks_data, funds_data
        
        if os.path.exists(str(DATA_DIR / 'test_users_BANK.csv')):
            users_df = pd.read_csv(DATA_DIR / 'test_users_BANK.csv')
            add_log(admin_logs, "REFRESH", "SUCCESS", f"Reloaded {len(users_df)} users")
        
        if os.path.exists(str(DATA_DIR / 'engineered_stocks.json')):
            with open(DATA_DIR / 'engineered_stocks.json', 'r') as f:
                stocks_data = json.load(f)
            add_log(admin_logs, "REFRESH", "SUCCESS", f"Reloaded {len(stocks_data)} stocks")
        
        if os.path.exists(str(DATA_DIR / 'engineered_funds.json')):
            with open(DATA_DIR / 'engineered_funds.json', 'r') as f:
                funds_data = json.load(f)
            add_log(admin_logs, "REFRESH", "SUCCESS", f"Reloaded {len(funds_data)} funds")
        
        return {"status": "success", "message": "Data refreshed successfully"}
    except Exception as e:
        add_log(admin_logs, "REFRESH", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "users_loaded": len(users_df) if users_df is not None else 0,
        "stocks_loaded": len(stocks_data) if stocks_data is not None else 0,
        "funds_loaded": len(funds_data) if funds_data is not None else 0,
        "model_loaded": model_data is not None,
        "total_logs": len(admin_logs)
    }

