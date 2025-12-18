"""
User Routes
"""
from fastapi import APIRouter, HTTPException
from typing import Dict
import pandas as pd
import numpy as np
from app.api.dependencies import users_df, model_data, insurance_users_df, admin_logs
from app.utils.feature_engineering import calculate_derived_features, engineer_single_user_vector
from app.services.risk_service import predict_user_risk
from app.utils.helpers import add_log

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/users")
async def get_all_users():
    """Get list of all users"""
    if users_df is None:
        raise HTTPException(status_code=404, detail="Users data not loaded")
    
    users_list = users_df.replace({np.nan: None}).to_dict('records')
    add_log(admin_logs, "GET_USERS", "SUCCESS", f"Retrieved {len(users_list)} users")
    
    return {
        "total_users": len(users_list),
        "users": users_list
    }


@router.get("/user/{user_id}")
async def get_user_profile(user_id: int):
    """Get complete user profile with risk analysis"""
    if users_df is None:
        raise HTTPException(status_code=404, detail="Users data not loaded")
    
    user_row = users_df[users_df['customer_id'] == user_id]
    
    if user_row.empty:
        add_log(admin_logs, "GET_USER", "ERROR", f"User {user_id} not found")
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    
    user_base = user_row.iloc[0].to_dict()
    
    # Calculate derived features
    derived_features = calculate_derived_features(user_base)
    user_complete = {**user_base, **derived_features}
    
    # Predict risk
    risk_pred = predict_user_risk(user_complete, model_data)
    user_complete['risk_label'] = risk_pred['risk_label']
    user_complete['risk_score'] = risk_pred['risk_score']
    
    # Engineer vector
    vector = engineer_single_user_vector(user_complete)
    
    # Get family information for insurance predictions
    family_size = 4
    num_children = 1
    num_elders = 1
    num_adults = 2
    
    # Try to get family data from insurance users CSV if available
    if insurance_users_df is not None:
        insurance_user_row = insurance_users_df[insurance_users_df['customer_id'] == user_id]
        if not insurance_user_row.empty:
            insurance_user = insurance_user_row.iloc[0]
            family_size = int(insurance_user.get('familysize', 4))
            num_children = int(insurance_user.get('numchildren', 1))
            num_elders = int(insurance_user.get('numelders', 1))
            num_adults = int(insurance_user.get('numadults', 2))
    
    # Build result with all required fields
    result = {
        "user_id": str(user_complete['customer_id']),
        "engineered_vector": [round(float(v), 6) for v in vector],
        "metadata": {
            "age": int(user_complete['age']),
            "gender": str(user_complete['gender']),
            "occupation": str(user_complete['occupation']),
            "risk_label": str(user_complete['risk_label']),
            "risk_score": round(float(user_complete['risk_score']), 2),
            "income": round(float(user_complete['annualincome']), 2),
            "credit_score": int(user_complete['creditscore']),
            "savings_rate": round(float(user_complete['savingsrate']), 4),
            "debt_to_income": round(float(user_complete['debttoincomeratio']), 4),
            "digital_activity": round(float(user_complete['digitalactivityscore']), 2),
            "portfolio_diversity": round(float(user_complete['portfoliodiversityscore']), 2),
            "city_tier": int(user_complete.get('citytier', 1)),
            "family_size": family_size,
            "num_children": num_children,
            "num_elders": num_elders,
            "num_adults": num_adults
        },
        "derived_features": {
            "transaction_volatility": round(float(user_complete['transactionvolatility']), 4),
            "spending_stability": round(float(user_complete['spendingstabilityindex']), 4),
            "credit_utilization": round(float(user_complete['creditutilizationratio']), 4),
            "missed_payments": int(user_complete['missedpaymentcount'])
        },
        "notes": "User vector aligned with stock, mutual fund, and insurance embeddings"
    }
    
    add_log(admin_logs, "GET_USER", "SUCCESS", f"Retrieved profile for {user_id}")
    return result

