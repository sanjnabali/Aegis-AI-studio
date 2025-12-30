"""
Aegis AI Studio - Main Application Entry Point (Optimized)
===========================================================
High-performance FastAPI backend with:
- Groq (800 tok/s ultra-fast)
- HuggingFace (specialized models - optional)
- Zero-latency routing
- Aggressive caching
- Production-grade error handling
"""

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Import application modules
from app import openai_adapter
from app.core import models
from app.core.config import get_settings
from app.core.cache import init_cache, close_cache

# Load settings
settings = get_settings()

# Global startup time tracker
_startup_time = None

# ============================================================================
# LOGGING CONFIGURATION (Optimized)
# ============================================================================

def setup_logging():
    """Configure Loguru logger with minimal overhead"""
    
    # Remove default handler
    logger.remove()
    
    # Console handler (simple format for performance)
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        enqueue=False,  # Disable queue for lower latency
    )
    
    # File handler (only if not in production)
    if settings.log_level == "DEBUG":
        logger.add(
            "/app/logs/aegis_{time:YYYY-MM-DD}.log",
            rotation="00:00",
            retention="7 days",  # Reduced retention for space
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            enqueue=True,  # Use queue for file I/O
        )
    
    logger.info("✓ Logging initialized")


# ============================================================================
# APPLICATION LIFECYCLE (Optimized)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.
    Optimized for fast startup (<5 seconds).
    """
    global _startup_time
    
    logger.info("=" * 60)
    logger.info("🛡️  AEGIS AI STUDIO - OPTIMIZED")
    logger.info("=" * 60)
    
    start = time.time()
    _startup_time = start
    
    try:
        # Setup logging
        setup_logging()
        
        # Initialize APIs (Groq only, HF lazy-loaded)
        logger.info("📡 Initializing APIs...")
        models.initialize_apis()
        logger.success("✓ Groq initialized (HF lazy-loaded)")
        
        # Initialize cache (async, non-blocking)
        if settings.enable_caching:
            logger.info("💾 Connecting to Redis...")
            await init_cache()
            logger.success("✓ Cache ready")
        else:
            logger.info("⚠️  Cache disabled")
        
        # Verify config (fast checks only)
        _verify_config_fast()
        
        # Calculate startup time
        duration = time.time() - start
        
        logger.success("=" * 60)
        logger.success(f"✅ Ready in {duration:.2f}s")
        logger.success("=" * 60)
        logger.info(f"📍 API: http://0.0.0.0:8000")
        logger.info(f"📖 Docs: http://0.0.0.0:8000/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise
    
    # Yield control (app is running)
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")
    
    if settings.enable_caching:
        await close_cache()
    
    logger.success("✅ Shutdown complete")


def _verify_config_fast():
    """Fast configuration verification (no network calls)"""
    issues = []
    
    # Check API key format only (don't validate with network)
    if not settings.groq_api_key or not settings.groq_api_key.startswith("gsk_"):
        issues.append("GROQ_API_KEY invalid format")
    
    if issues:
        logger.warning(f"⚠️  Config issues: {', '.join(issues)}")
    else:
        logger.success("✓ Config OK")


# ============================================================================
# FASTAPI APPLICATION (Optimized)
# ============================================================================

app = FastAPI(
    title="Aegis AI Studio",
    description="Ultra-fast AI gateway powered by Groq + HuggingFace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,  # Disable ReDoc to save memory
    lifespan=lifespan,
    # Hide OpenAPI in production
    openapi_url="/openapi.json" if settings.log_level == "DEBUG" else None,
)

# ============================================================================
# MIDDLEWARE (Minimal overhead)
# ============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip (only for large responses)
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
    compresslevel=5,  # Lower = faster
)

# ============================================================================
# REQUEST LOGGING (Lightweight)
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Minimal request logging"""
    
    # Skip logging for health checks (reduce noise)
    if request.url.path in ["/health", "/ping"]:
        return await call_next(request)
    
    start = time.time()
    
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000  # ms
        
        # Log only slow requests (>1s) or errors
        if duration > 1000 or response.status_code >= 400:
            logger.warning(
                f"{request.method} {request.url.path} | "
                f"{response.status_code} | {duration:.0f}ms"
            )
        
        return response
        
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.error(f"{request.method} {request.url.path} | Error after {duration:.0f}ms")
        raise


# ============================================================================
# EXCEPTION HANDLER
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    
    logger.error(f"Unhandled error: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
            }
        },
    )


# ============================================================================
# API ROUTERS
# ============================================================================

app.include_router(
    openai_adapter.router,
    tags=["OpenAI API"],
)

# ============================================================================
# CORE ENDPOINTS (Optimized)
# ============================================================================

@app.get("/", tags=["Core"])
async def root():
    """Root endpoint"""
    return {
        "name": "Aegis AI Studio",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/v1/models",
            "chat": "/v1/chat/completions",
        },
    }


@app.get("/health", tags=["Core"])
async def health_check():
    """
    Fast health check endpoint.
    
    Returns:
        - status: healthy/degraded/unhealthy
        - components: service status
        - uptime: seconds since startup
    """
    
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": int(time.time() - _startup_time) if _startup_time else 0,
        "components": {},
    }
    
    # Check Groq (fast - no network call)
    try:
        models.get_groq_client()
        health["components"]["groq"] = "operational"
    except:
        health["components"]["groq"] = "unavailable"
        health["status"] = "degraded"
    
    # Check cache
    if settings.enable_caching:
        from app.core.cache import redis_client
        health["components"]["cache"] = "operational" if redis_client else "unavailable"
    else:
        health["components"]["cache"] = "disabled"
    
    # Check HF models
    health["components"]["hf_models"] = "enabled" if settings.enable_hf_models else "disabled"
    
    # Return appropriate status code
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(status_code=status_code, content=health)


@app.get("/ping", tags=["Core"])
async def ping():
    """Ultra-fast ping endpoint"""
    return {"status": "ok", "timestamp": int(time.time())}


@app.get("/stats", tags=["Core"])
async def stats():
    """
    Get system statistics.
    
    Returns:
        - Performance metrics
        - Feature flags
        - Model info
    """
    
    uptime = int(time.time() - _startup_time) if _startup_time else 0
    
    # Get metrics from backends
    backend_metrics = models.get_metrics()
    
    # Get cache stats
    from app.core.cache import get_cache_stats
    cache_stats = get_cache_stats()
    
    return {
        "uptime_seconds": uptime,
        "backends": backend_metrics,
        "cache": cache_stats,
        "features": {
            "caching": settings.enable_caching,
            "hf_models": settings.enable_hf_models,
            "streaming": True,
        },
        "configuration": {
            "groq_model": settings.groq_model,
            "max_tokens": settings.groq_max_tokens,
        },
    }


# ============================================================================
# DEBUG ENDPOINTS (Only in DEBUG mode)
# ============================================================================

if settings.log_level == "DEBUG":
    
    @app.get("/debug/config", tags=["Debug"])
    async def debug_config():
        """View configuration (secrets hidden)"""
        return {
            "groq_model": settings.groq_model,
            "groq_api_key_set": bool(settings.groq_api_key and len(settings.groq_api_key) > 10),
            "caching": settings.enable_caching,
            "hf_models": settings.enable_hf_models,
            "log_level": settings.log_level,
        }
    
    @app.post("/debug/test-groq", tags=["Debug"])
    async def test_groq():
        """Test Groq API"""
        try:
            result = []
            async for chunk in models.groq_stream([
                {"role": "user", "content": "Say 'Hello!'"}
            ]):
                result.append(chunk)
            return {"success": True, "response": "".join(result)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @app.get("/debug/available-models", tags=["Debug"])
    async def debug_models():
        """Get detailed model info"""
        return models.get_available_models()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Direct execution for development.
    Production: docker-compose or uvicorn app.main:app
    """
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=False,  # Disable access log (we have our own)
        workers=1,
    )