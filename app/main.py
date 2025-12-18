"""
FastAPI Application Entry Point
Enterprise-level entry point
"""
import uvicorn
from app.api.app import app
from app.core.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)

