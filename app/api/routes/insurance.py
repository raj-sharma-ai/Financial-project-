"""
Insurance Routes
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
from app.api.dependencies import (
    insurance_model, insurance_scaler, insurance_label_encoder,
    insurance_feature_names, health_policies, insurance_users_df,
    users_df, admin_logs
)
from app.api.main import predict_insurance_policies, load_insurance_prediction_models
from app.core.config import DATA_DIR, MODELS_DIR
from app.utils.helpers import add_log

router = APIRouter(prefix="/api/insurance", tags=["insurance"])


@router.get("/predict/{user_id}")
async def predict_user_insurance(user_id: int):
    """Predict which insurance policies a user is likely to purchase"""
    try:
        add_log(admin_logs, "INSURANCE_PREDICT", "PROCESSING", f"Predicting insurance for user {user_id}")
        
        result = predict_insurance_policies(user_id)
        
        add_log(admin_logs, "INSURANCE_PREDICT", "SUCCESS", 
                f"Generated predictions for user {user_id} - Top policy: {result['top_policy']['policy_name']} ({result['top_policy']['probability']}%)")
        
        return result
    except Exception as e:
        add_log(admin_logs, "INSURANCE_PREDICT", "ERROR", f"Error predicting insurance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/top-policies/{user_id}")
async def get_top_insurance_policies(user_id: int, top_k: int = 5):
    """Get top K insurance policies predicted for a user"""
    try:
        result = predict_insurance_policies(user_id)
        result['predicted_policies'] = result['predicted_policies'][:top_k]
        add_log(admin_logs, "INSURANCE_TOP", "SUCCESS", f"Retrieved top {top_k} policies for user {user_id}")
        return result
    except Exception as e:
        add_log(admin_logs, "INSURANCE_TOP", "ERROR", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def check_insurance_models_status():
    """Check if insurance prediction models are loaded"""
    return {
        "model_loaded": insurance_model is not None,
        "scaler_loaded": insurance_scaler is not None,
        "label_encoder_loaded": insurance_label_encoder is not None,
        "feature_names_loaded": insurance_feature_names is not None,
        "health_policies_loaded": health_policies is not None,
        "insurance_users_csv_loaded": insurance_users_df is not None,
        "num_insurance_users": len(insurance_users_df) if insurance_users_df is not None else 0,
        "num_policies": len(health_policies) if health_policies else 0,
        "num_features": len(insurance_feature_names) if insurance_feature_names else 0,
        "feature_names": insurance_feature_names if insurance_feature_names else []
    }


@router.get("/debug/files")
async def debug_check_files():
    """Debug endpoint to check which files exist"""
    files_to_check = {
        'test_users_BANK.csv': DATA_DIR / 'test_users_BANK.csv',
        'xgb_model_synthetic.pkl': MODELS_DIR / 'xgb_model_synthetic.pkl',
        'xgb_scaler_synthetic.pkl': MODELS_DIR / 'xgb_scaler_synthetic.pkl',
        'xgb_label_encoder_synthetic.pkl': MODELS_DIR / 'xgb_label_encoder_synthetic.pkl',
        'xgb_feature_names_synthetic.pkl': MODELS_DIR / 'xgb_feature_names_synthetic.pkl',
        'health_policies.json': DATA_DIR / 'health_policies.json'
    }
    
    file_status = {}
    for file_name, file_path in files_to_check.items():
        exists = os.path.exists(file_path)
        file_status[file_name] = {
            'exists': exists,
            'path': str(file_path),
            'size': os.path.getsize(file_path) if exists else 0
        }
    
    required_files = [
        'xgb_model_synthetic.pkl',
        'xgb_scaler_synthetic.pkl',
        'xgb_label_encoder_synthetic.pkl',
        'xgb_feature_names_synthetic.pkl',
        'health_policies.json'
    ]
    
    missing_required = [f for f in required_files if not file_status.get(f, {}).get('exists', False)]
    
    return {
        'base_directory': str(Path(__file__).resolve().parent.parent.parent.parent),
        'data_directory': str(DATA_DIR),
        'models_directory': str(MODELS_DIR),
        'files': file_status,
        'required_files_missing': missing_required,
        'status': 'OK' if not missing_required else 'MISSING_FILES',
        'message': 'All required files found!' if not missing_required else f'Missing: {", ".join(missing_required)}'
    }


@router.post("/reload")
async def reload_insurance_models():
    """Manually reload insurance prediction models"""
    try:
        success = load_insurance_prediction_models()
        
        if success:
            return {
                'status': 'success',
                'message': '✅ Insurance models reloaded successfully',
                'models_loaded': {
                    'model': insurance_model is not None,
                    'scaler': insurance_scaler is not None,
                    'label_encoder': insurance_label_encoder is not None,
                    'feature_names': insurance_feature_names is not None,
                    'health_policies': health_policies is not None,
                    'insurance_users': insurance_users_df is not None,
                    'num_users': len(insurance_users_df) if insurance_users_df is not None else 0
                }
            }
        else:
            return {
                'status': 'failed',
                'message': '❌ Failed to load insurance models - check admin logs',
                'suggestion': 'Use /api/insurance/debug/files to check file locations'
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

