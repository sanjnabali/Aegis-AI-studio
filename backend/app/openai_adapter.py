"""
OpenAI-Compatible API Adapter - Optimized
=========================================
Supports:
- Groq (primary): 800 tok/s, ultra-fast
- HuggingFace (specialized): Code, images, voice
- Smart routing with zero-latency overhead
- Aggressive caching
"""

import uuid
import time
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger

from app.core import models
from app.core.cache import (
    generate_cache_key,
    get_cached_response,
    set_cached_response,
    get_cache_stats,
)
from app.core.schemas import (
    ChatCompletionRequest,
    StreamChoice,
    StreamDelta,
    ChatCompletionStreamResponse,
    ModelData,
    ModelList,
)
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()

# ============================================================================
# MODEL REGISTRY (Updated for llama-3.3-70b-versatile)
# ============================================================================

MODELS = {
    "aegis-groq-turbo": {
        "id": "aegis-groq-turbo",
        "name": "Aegis Groq Llama 3.3 70B (Ultra-Fast)",
        "backend": "groq",
        "description": "Llama 3.3 70B via Groq - Maximum speed and capability",
        "speed": "800 tok/s",
        "ttft": "~200ms",
        "context": "8k tokens",
        "best_for": "Chat, general tasks, real-time responses",
        "rate_limit": "30 req/min",
        "provider": "Groq Cloud",
        "model_string": "llama-3.3-70b-versatile",
    },
    "aegis-auto": {
        "id": "aegis-auto",
        "name": "Aegis Auto (Smart Routing)",
        "backend": "auto",
        "description": "Intelligent routing - uses best model for each task",
        "speed": "Variable (up to 800 tok/s)",
        "context": "8k tokens",
        "best_for": "Mixed workloads, code + chat",
        "rate_limit": "30 req/min",
        "provider": "Auto-select",
        "model_string": "auto",
    },
    "aegis-groq-mixtral": {
        "id": "aegis-groq-mixtral",
        "name": "Aegis Mixtral 8x7B (Long Context)",
        "backend": "groq",
        "description": "Mixtral 8x7B via Groq - Extended context window",
        "speed": "500 tok/s",
        "ttft": "~300ms",
        "context": "32k tokens",
        "best_for": "Long documents, complex analysis",
        "rate_limit": "30 req/min",
        "provider": "Groq Cloud",
        "model_string": "mixtral-8x7b-32768",
    },
}

# Conditionally add HF models if enabled
if settings.enable_hf_models:
    MODELS.update({
        "aegis-code": {
            "id": "aegis-code",
            "name": "Aegis Code (DeepSeek 1.3B)",
            "backend": "hf_code",
            "description": "DeepSeek Coder 1.3B - Specialized code generation",
            "speed": "~100 tok/s (local)",
            "context": "4k tokens",
            "best_for": "Code generation, debugging, refactoring",
            "provider": "HuggingFace (Local)",
            "model_string": "deepseek-coder-1.3b",
        },
        "aegis-image": {
            "id": "aegis-image",
            "name": "Aegis Image (SDXL Turbo)",
            "backend": "hf_image",
            "description": "SDXL Turbo - Fast 1-step image generation",
            "speed": "~1s per image",
            "best_for": "Fast image generation, prototyping",
            "provider": "HuggingFace (Local)",
            "model_string": "sdxl-turbo",
        },
    })

# ============================================================================
# MODEL LIST ENDPOINT
# ============================================================================

@router.get("/v1/models")
async def get_models():
    """
    List all available models with performance characteristics.
    OpenAI-compatible endpoint.
    """
    return ModelList(
        data=[
            ModelData(
                id=model_id,
                object="model",
                owned_by="aegis",
                permission=[],
            )
            for model_id in MODELS.keys()
        ]
    )


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get detailed information about a specific model"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "aegis",
        **MODELS[model_id],
    }


# ============================================================================
# MESSAGE PROCESSING (Optimized)
# ============================================================================

def _process_messages(request: ChatCompletionRequest) -> list:
    """
    Fast message processing with minimal allocations.
    Extracts text content from potentially multimodal messages.
    """
    messages = []
    
    for msg in request.messages:
        if isinstance(msg.content, str):
            # Fast path: simple string content
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })
        elif isinstance(msg.content, list):
            # Handle multimodal content (extract text only)
            text_parts = [
                part.text
                for part in msg.content
                if hasattr(part, 'type') and part.type == "text" and hasattr(part, 'text') and part.text
            ]
            
            if text_parts:
                messages.append({
                    "role": msg.role,
                    "content": " ".join(text_parts),
                })
    
    return messages


# ============================================================================
# STREAMING GENERATOR (Optimized)
# ============================================================================

async def stream_generator(request: ChatCompletionRequest):
    """
    Highly optimized streaming generator with:
    - Zero-copy message processing
    - Smart backend selection
    - Aggressive caching
    - Minimal error overhead
    """
    
    start_time = time.time()
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"  # Shorter UUID
    
    # Process messages (fast path)
    messages = _process_messages(request)
    
    if not messages:
        yield 'data: {"error": "No valid messages"}\n\n'
        yield "data: [DONE]\n\n"
        return
    
    # Determine backend (zero overhead lookup)
    model_config = MODELS.get(request.model)
    if not model_config:
        logger.warning(f"Unknown model '{request.model}', using aegis-groq-turbo")
        request.model = "aegis-groq-turbo"
        model_config = MODELS["aegis-groq-turbo"]
    
    backend = model_config["backend"]
    
    # Check cache (only if enabled)
    cache_key = None
    full_response = []
    
    if settings.enable_caching:
        cache_key = generate_cache_key(messages, request.model)
        cached = await get_cached_response(cache_key)
        
        if cached:
            logger.info(f"⚡ Cache HIT (saved API call)")
            
            # Send role
            yield f'data: {{"id":"{chat_id}","object":"chat.completion.chunk","created":{int(time.time())},"model":"{request.model}","choices":[{{"delta":{{"role":"assistant"}},"index":0}}]}}\n\n'
            
            # Send cached content in chunks (simulate streaming for UX)
            chunk_size = 50
            for i in range(0, len(cached), chunk_size):
                chunk = cached[i:i+chunk_size]
                # Escape quotes in chunk
                chunk_escaped = chunk.replace('"', '\\"').replace('\n', '\\n')
                yield f'data: {{"id":"{chat_id}","object":"chat.completion.chunk","model":"{request.model}","choices":[{{"delta":{{"content":"{chunk_escaped}"}},"index":0}}]}}\n\n'
            
            # Send finish
            yield f'data: {{"id":"{chat_id}","object":"chat.completion.chunk","model":"{request.model}","choices":[{{"delta":{{}},"index":0,"finish_reason":"stop"}}]}}\n\n'
            yield "data: [DONE]\n\n"
            
            logger.info(f"✓ Served from cache in {(time.time()-start_time)*1000:.0f}ms")
            return
    
    # Cache miss - stream from backend
    logger.info(f"🚀 {request.model} ({backend}) | {len(messages)} messages")
    
    # Send initial role chunk
    first_chunk = ChatCompletionStreamResponse(
        id=chat_id,
        model=request.model,
        choices=[StreamChoice(
            delta=StreamDelta(role="assistant"),
            index=0
        )]
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"
    
    # Stream from backend
    try:
        # Route to appropriate backend
        if backend == "groq":
            # Use the model string from config
            groq_model = model_config.get("model_string", "llama-3.3-70b-versatile")
            stream = models.groq_stream(
                messages,
                model=groq_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        elif backend == "auto":
            stream = models.unified_stream(
                messages,
                model="auto",
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        elif backend == "hf_code":
            # HF code generation (non-streaming, but we'll chunk it)
            last_msg = messages[-1]["content"]
            code_result = await models.hf_code_generate(last_msg)
            
            # Send in chunks
            chunk_size = 50
            for i in range(0, len(code_result), chunk_size):
                chunk = code_result[i:i+chunk_size]
                full_response.append(chunk)
                
                chunk_response = ChatCompletionStreamResponse(
                    id=chat_id,
                    model=request.model,
                    choices=[StreamChoice(
                        delta=StreamDelta(content=chunk),
                        index=0
                    )]
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"
            
            # Done with code gen
            stream = None  # Skip streaming loop
        else:
            # Default to unified
            stream = models.unified_stream(messages)
        
        # Stream chunks (if streaming backend)
        if stream:
            async for chunk_text in stream:
                full_response.append(chunk_text)
                
                chunk_response = ChatCompletionStreamResponse(
                    id=chat_id,
                    model=request.model,
                    choices=[StreamChoice(
                        delta=StreamDelta(content=chunk_text),
                        index=0
                    )]
                )
                yield f"data: {chunk_response.model_dump_json()}\n\n"
        
        # Cache complete response
        if settings.enable_caching and cache_key and full_response:
            complete = "".join(full_response)
            await set_cached_response(cache_key, complete)
        
        # Send finish reason
        finish_chunk = ChatCompletionStreamResponse(
            id=chat_id,
            model=request.model,
            choices=[StreamChoice(
                delta=StreamDelta(),
                index=0,
                finish_reason="stop"
            )]
        )
        yield f"data: {finish_chunk.model_dump_json()}\n\n"
        
        # Log performance
        duration = time.time() - start_time
        tokens_approx = len("".join(full_response).split())
        speed = tokens_approx / duration if duration > 0 else 0
        
        logger.success(
            f"✓ Completed | {duration:.2f}s | "
            f"~{tokens_approx} tokens | {speed:.0f} tok/s"
        )
        
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
        
        # Send user-friendly error
        error_chunk = ChatCompletionStreamResponse(
            id=chat_id,
            model=request.model,
            choices=[StreamChoice(
                delta=StreamDelta(
                    content=f"\n\n[Error: {str(e)}. Please try again or select a different model.]"
                ),
                index=0
            )]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"


# ============================================================================
# CHAT COMPLETION ENDPOINT
# ============================================================================

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """
    OpenAI-compatible chat completion endpoint (optimized).
    
    Features:
    - Streaming (SSE) with minimal latency
    - Smart model routing
    - Aggressive caching
    - Zero-overhead for cached requests
    
    Performance:
    - Cached response: ~50ms
    - Groq new response: ~200ms TTFT, 800 tok/s
    - HF code gen: 1-3s
    """
    
    # Validate model (fast path)
    if request.model not in MODELS:
        logger.warning(f"Unknown model '{request.model}', defaulting to aegis-groq-turbo")
        request.model = "aegis-groq-turbo"
    
    # Log request (minimal overhead)
    logger.info(
        f"📨 {req.client.host} | {request.model} | "
        f"{len(request.messages)}msg"
    )
    
    # Return streaming response
    return StreamingResponse(
        stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


# ============================================================================
# PERFORMANCE METRICS ENDPOINT
# ============================================================================

@router.get("/v1/metrics")
async def get_performance_metrics():
    """
    Get real-time performance metrics.
    
    Returns:
    - Backend statistics (requests, speed, latency)
    - Cache statistics (hit rate, savings)
    - Theoretical benchmarks
    - Available models
    """
    
    model_metrics = models.get_metrics()
    cache_metrics = get_cache_stats()
    
    return {
        "backends": model_metrics,
        "cache": cache_metrics,
        "models": {
            "available": list(MODELS.keys()),
            "default": "aegis-groq-turbo",
            "total": len(MODELS),
        },
        "benchmark": {
            "groq_llama_3_3_70b": {
                "speed": "800 tok/s",
                "ttft": "~200ms",
                "rate_limit": "30 req/min",
                "context": "8k tokens",
                "recommended_for": "General chat, fast responses, real-time",
            },
            "groq_mixtral_8x7b": {
                "speed": "500 tok/s",
                "ttft": "~300ms",
                "rate_limit": "30 req/min",
                "context": "32k tokens",
                "recommended_for": "Long documents, complex analysis",
            },
            "hf_code": {
                "speed": "~100 tok/s (local)",
                "latency": "1-3s",
                "context": "4k tokens",
                "recommended_for": "Code generation",
                "enabled": settings.enable_hf_models,
            },
        },
        "system": {
            "uptime_seconds": time.time() - _startup_time,
            "caching_enabled": settings.enable_caching,
            "hf_models_enabled": settings.enable_hf_models,
        }
    }


# ============================================================================
# AVAILABLE MODELS INFO ENDPOINT
# ============================================================================

@router.get("/v1/models/info")
async def get_models_info():
    """
    Get detailed info about all available models.
    Includes performance characteristics and use cases.
    """
    return {
        "models": MODELS,
        "recommendations": {
            "fastest": "aegis-groq-turbo",
            "balanced": "aegis-auto",
            "code": "aegis-code" if settings.enable_hf_models else "aegis-groq-turbo",
            "long_context": "aegis-groq-mixtral",
        }
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/v1/health")
async def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
    - Component status
    - Model availability
    - Performance metrics summary
    """
    
    try:
        # Check Groq
        groq_status = "operational"
        try:
            models.get_groq_client()
        except:
            groq_status = "unavailable"
        
        # Check cache
        cache_status = "operational" if settings.enable_caching else "disabled"
        
        # Check HF
        hf_status = "enabled" if settings.enable_hf_models else "disabled"
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "groq": groq_status,
                "cache": cache_status,
                "hf_models": hf_status,
            },
            "models": {
                "available": list(MODELS.keys()),
                "count": len(MODELS),
                "primary": "llama-3.3-70b-versatile",
            },
            "performance": {
                "cache_hit_rate": get_cache_stats().get("hit_rate", "0%"),
                "total_requests": model_metrics.get("groq", {}).get("requests", 0),
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# ============================================================================
# STARTUP TRACKING
# ============================================================================

_startup_time = time.time()


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'router',
    'MODELS',
    'get_models',
    'chat_completions',
    'get_performance_metrics',
]