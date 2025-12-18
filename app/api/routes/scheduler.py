"""
Scheduler Routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.dependencies import scheduler_status, admin_logs
from app.api.main import scheduled_data_refresh
from app.utils.helpers import add_log

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/status")
async def get_scheduler_status():
    """Get scheduler status"""
    return scheduler_status


@router.post("/trigger")
async def trigger_scheduler(background_tasks: BackgroundTasks):
    """Manually trigger data refresh"""
    if scheduler_status["is_running"]:
        raise HTTPException(status_code=409, detail="Pipeline already running")
    
    background_tasks.add_task(scheduled_data_refresh)
    return {"status": "triggered", "message": "Data refresh started in background"}

