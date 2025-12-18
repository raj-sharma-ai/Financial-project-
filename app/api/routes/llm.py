"""
LLM Routes
"""
from fastapi import APIRouter
import aiohttp
from app.core.config import GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL
from app.api.dependencies import admin_logs
from app.utils.helpers import add_log

router = APIRouter(prefix="", tags=["llm"])


@router.get("/health")
async def check_llm_health():
    """Check if Groq API is accessible"""
    try:
        if not GROQ_API_KEY:
            return {
                "status": "error",
                "message": "Groq API key not configured",
                "configured_model": GROQ_MODEL
            }
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "temperature": 0.5
            }
            
            async with session.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    return {
                        "status": "online",
                        "configured_model": GROQ_MODEL,
                        "api_url": GROQ_API_URL
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"API returned status {response.status}: {response_text}",
                        "configured_model": GROQ_MODEL
                    }
    except Exception as e:
        return {
            "status": "offline",
            "message": str(e),
            "configured_model": GROQ_MODEL
        }


@router.get("/debug/groq-test")
async def debug_groq_test():
    """Test Groq API with full error details"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10
            }
            
            async with session.post(
                GROQ_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_text = await response.text()
                
                return {
                    "status_code": response.status,
                    "response": response_text,
                    "payload_sent": payload,
                    "headers_sent": {"Authorization": "Bearer ***", "Content-Type": "application/json"}
                }
    except Exception as e:
        return {"error": str(e)}

