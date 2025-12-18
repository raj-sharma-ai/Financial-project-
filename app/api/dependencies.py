"""
FastAPI Dependencies
Shared global state and dependencies
"""
from app.api.app import (
    users_df, model_data, stocks_data, funds_data, insurance_data,
    admin_logs, insurance_model, insurance_scaler, insurance_label_encoder,
    insurance_feature_names, health_policies, insurance_users_df, scheduler_status
)

__all__ = [
    "users_df", "model_data", "stocks_data", "funds_data", "insurance_data",
    "admin_logs", "insurance_model", "insurance_scaler", "insurance_label_encoder",
    "insurance_feature_names", "health_policies", "insurance_users_df", "scheduler_status"
]
