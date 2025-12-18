"""
Root Route
"""
from fastapi import APIRouter
from app.api.dependencies import scheduler_status

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Financial Recommendation API",
        "version": "1.0.0",
        "scheduler": {
            "enabled": scheduler_status["enabled"],
            "next_run": scheduler_status["next_run"],
            "last_run": scheduler_status["last_run"]
        },
        "endpoints": {
            "users": "/api/users",
            "user_profile": "/api/user/{user_id}",
            "recommendations": "/api/recommendations/{user_id}",
            "explain": "/api/explain",
            "explain_individual": "/api/explain-individual",
            "llm_health": "/api/health",
            "admin_logs": "/api/admin/logs",
            "insurance_predict": "/api/insurance/predict/{user_id}",
            "insurance_top_policies": "/api/insurance/top-policies/{user_id}",
            "insurance_models_status": "/api/insurance/models/status",
            "insurance_debug_files": "/api/insurance/debug/files",
            "insurance_reload": "/api/insurance/reload"
        }
    }

