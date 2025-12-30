"""
Aegis AI Studio - Main Application
===================================
Ultra-Lightweight Version
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import sys

from app.core.config import get_settings
from app.openai_adapter import router as openai_router

settings = get_settings()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=settings.log_level,
    colorize=True,
)

logger.add(
    "/app/logs/aegis.log",
    rotation="100 MB",
    retention="7 days",
    compression="zip",
    level="INFO",
)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    logger.info("=" * 60)
    logger.info("🛡️  AEGIS AI STUDIO - ULTRA-LIGHTWEIGHT")
    logger.info("=" * 60)
    
    try:
        logger.info("✓ Logging initialized")
        
        # Initialize Groq client (0 disk/RAM)
        logger.info("📡 Initializing Groq API...")
        from app.core import models
        try:
            models.get_groq_client()
            logger.success("✓ Groq API ready (0 disk/RAM)")
        except Exception as e:
            logger.warning(f"⚠️  Groq API not configured: {e}")
            logger.info("💡 Set GROQ_API_KEY in .env to enable")
        
        # Check HF models status
        if settings.enable_hf_models:
            logger.info("🤗 HuggingFace models: ENABLED")
            logger.info("   Models will load on-demand:")
            logger.info("   • DeepSeek Coder (1.3GB) - for code")
            logger.info("   • MiniLM (80MB) - for embeddings")
            logger.info("   • Whisper Tiny (150MB) - for voice")
            logger.info("   Total: ~1.5GB (minimal disk usage)")
        else:
            logger.warning("⚠️  HuggingFace models: DISABLED")
            logger.info("💡 Set ENABLE_HF_MODELS=true in .env to enable")
        
        # Cache status
        if settings.enable_caching:
            logger.info("💾 Caching: ENABLED")
        else:
            logger.info("💾 Caching: DISABLED")
        
        # Web search status
        if settings.enable_web_search:
            logger.info("🔍 Web search: ENABLED")
        else:
            logger.info("🔍 Web search: DISABLED")
        
        logger.info("=" * 60)
        logger.success("✅ AEGIS STUDIO READY - ULTRA-LIGHTWEIGHT MODE")
        logger.info("=" * 60)
        logger.info("📊 System Info:")
        logger.info("   • Mode: Ultra-Lightweight (~5GB total)")
        logger.info("   • Primary: Groq (0 disk, 800 tok/s)")
        logger.info("   • Local: Only essentials (1.5GB)")
        logger.info("   • Strategy: Cloud-first, minimal local storage")
        logger.info("=" * 60)
        logger.info("🌐 API Endpoints:")
        logger.info("   • Health: /health")
        logger.info("   • Models: /v1/models")
        logger.info("   • Chat: /v1/chat/completions")
        logger.info("   • Features: /v1/features")
        logger.info("   • Docs: /docs")
        logger.info("=" * 60)
        
        yield
        
    except Exception as e:
        logger.critical(f"❌ Startup failed: {e}")
        raise
    
    finally:
        logger.info("👋 Shutting down Aegis Studio...")


# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Aegis AI Studio (Ultra-Lightweight)",
    description="OpenAI-compatible API with minimal disk usage (~5GB)",
    version="2.0.0-ultralight",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(openai_router, tags=["OpenAI Compatible"])


# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "name": "Aegis AI Studio",
        "version": "2.0.0-ultralight",
        "mode": "ultra-lightweight",
        "disk_usage": "~5GB total",
        "description": "OpenAI-compatible API with minimal disk usage",
        "docs": "/docs",
        "health": "/health",
        "models": "/v1/models",
        "features": "/v1/features",
        "strategy": {
            "primary": "Groq (0 disk, 800 tok/s)",
            "local": "Only essentials (1.5GB)",
            "removed": "Vision/Image gen (use external APIs)",
        },
    }


@app.get("/health")
async def health_check():
    """Quick health check"""
    return {
        "status": "healthy",
        "mode": "ultra-lightweight",
        "disk_usage": "~5GB",
        "endpoints": {
            "health": "/health",
            "models": "/v1/models", 
            "chat": "/v1/chat/completions",
            "features": "/v1/features",
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "Endpoint not found",
            "docs": "/docs",
            "available_endpoints": {
                "root": "/",
                "health": "/health",
                "models": "/v1/models",
                "chat": "/v1/chat/completions",
                "features": "/v1/features",
            }
        }
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Custom 500 handler"""
    logger.error(f"Server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "support": "Check logs or /health endpoint",
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )