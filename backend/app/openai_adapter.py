"""
Ultra-Lightweight OpenAI Adapter - MINIMAL DISK USAGE
=====================================================
Total: ~5GB models (DeepSeek + Embeddings + Whisper)
Primary: Groq for 95% of tasks (0 disk)
"""

import uuid
import time
from typing import Optional
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response
from loguru import logger

from app.core import models
from app.core.cache import generate_cache_key, get_cached_response, set_cached_response, get_cache_stats
from app.core.schemas import ChatCompletionRequest, StreamChoice, StreamDelta, ChatCompletionStreamResponse, ModelData, ModelList
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()

# ============================================================================
# MODEL REGISTRY - ULTRA-LIGHTWEIGHT (5GB total)
# ============================================================================

MODELS = {
    # === PRIMARY: GROQ (0 DISK, RECOMMENDED FOR EVERYTHING) ===
    
    "aegis-groq-turbo": {
        "id": "aegis-groq-turbo",
        "name": "Aegis Groq Turbo (Llama 3.3 70B) ⚡ DEFAULT",
        "backend": "groq",
        "type": "chat",
        "description": "Ultra-fast, 0 disk/RAM - Use for 95% of tasks",
        "speed": "800 tok/s",
        "disk": "0GB (cloud)",
        "ram": "0GB (cloud)",
        "best_for": "Chat, reasoning, general tasks, vision (coming soon)",
        "recommended": True,
    },
    
    # === LOCAL: ONLY ESSENTIALS (5GB total) ===
    
    "aegis-code": {
        "id": "aegis-code",
        "name": "Aegis Code (DeepSeek 1.3B)",
        "backend": "hf_code",
        "type": "chat",
        "description": "Specialized code generation (only local model)",
        "speed": "~100 tok/s",
        "disk": "~1.3GB",
        "ram": "~2GB",
        "best_for": "Code generation, debugging (specialized)",
    },
    
    "aegis-vision": {
        "id": "aegis-vision",
        "name": "Aegis Vision (BLIP)",
        "backend": "hf_vision",
        "type": "vision",
        "description": "Image analysis and description",
        "speed": "~500ms per image",
        "disk": "~800MB",
        "ram": "~1GB",
        "best_for": "Image description, visual Q&A, scene understanding",
    },
    
    "aegis-embeddings": {
        "id": "aegis-embeddings",
        "name": "Aegis Embeddings (MiniLM)",
        "backend": "hf_embeddings",
        "type": "embeddings",
        "description": "Text embeddings for RAG",
        "disk": "~80MB",
        "ram": "~0.5GB",
        "best_for": "Semantic search, RAG, document similarity",
    },
    
    "aegis-whisper": {
        "id": "aegis-whisper",
        "name": "Aegis STT (Whisper Tiny)",
        "backend": "hf_whisper",
        "type": "audio",
        "description": "Speech-to-text (local for privacy)",
        "disk": "~150MB",
        "ram": "~0.5GB",
        "best_for": "Voice transcription",
    },
    
    # === SMART ROUTING ===
    
    "aegis-auto": {
        "id": "aegis-auto",
        "name": "Aegis Auto (Smart) 🤖",
        "backend": "auto",
        "type": "chat",
        "description": "Smart routing - Code→Local, Everything else→Groq",
        "disk": "~2.5GB",
        "best_for": "General use - Let AI decide",
    },
}

# ============================================================================
# INFO MESSAGES FOR REMOVED FEATURES
# ============================================================================

REMOVED_FEATURES_INFO = {
    "reasoning": "Use aegis-groq-turbo instead (same quality, 0 disk)",
    "image_gen": "Use external API: Stability AI, Replicate, or ComfyUI",
}

# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/v1/models")
async def get_models():
    """List available models"""
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
    """Get model details"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "aegis",
        **MODELS[model_id],
    }


def _process_messages(request: ChatCompletionRequest) -> list:
    """Process messages"""
    messages = []
    
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            text_parts = []
            image_urls = []
            
            for part in msg.content:
                if hasattr(part, 'type'):
                    if part.type == "text" and hasattr(part, 'text'):
                        text_parts.append(part.text)
                    elif part.type == "image_url" and hasattr(part, 'image_url'):
                        image_urls.append(part.image_url.url)
            
            if image_urls:
                # Vision request - will use Groq
                messages.append({
                    "role": msg.role,
                    "content": " ".join(text_parts) if text_parts else "What's in this image?",
                    "images": image_urls,
                })
            elif text_parts:
                messages.append({"role": msg.role, "content": " ".join(text_parts)})
    
    return messages


async def stream_generator(request: ChatCompletionRequest):
    """Ultra-lightweight streaming"""
    
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    messages = _process_messages(request)
    
    if not messages:
        yield 'data: {"error": "No messages"}\n\n'
        yield "data: [DONE]\n\n"
        return
    
    model_config = MODELS.get(request.model, MODELS["aegis-auto"])
    backend = model_config["backend"]
    
    # Check for images (use local BLIP for vision)
    has_images = any("images" in msg for msg in messages)
    if has_images:
        logger.info("🖼️ Vision request → Using local BLIP")
        logger.debug(f"Image count: {sum(len(msg.get('images', [])) for msg in messages)}")
        backend = "hf_vision"
    
    # Send role
    yield f'data: {{"id":"{chat_id}","object":"chat.completion.chunk","model":"{request.model}","choices":[{{"delta":{{"role":"assistant"}},"index":0}}]}}\n\n'
    
    full_response = []
    
    try:
        if backend == "hf_code":
            # Local code generation
            result = await models.hf_code_generate(messages[-1]["content"])
            
            # Stream in chunks
            chunk_size = 50
            for i in range(0, len(result), chunk_size):
                chunk = result[i:i+chunk_size]
                full_response.append(chunk)
                
                chunk_escaped = chunk.replace('"', '\\"').replace('\n', '\\n')
                yield f'data: {{"id":"{chat_id}","model":"{request.model}","choices":[{{"delta":{{"content":"{chunk_escaped}"}},"index":0}}]}}\n\n'
            
            stream = None
        
        elif backend == "hf_vision":
            # Local vision with BLIP
            last_msg = messages[-1]
            if "images" in last_msg and last_msg["images"]:
                try:
                    from PIL import Image
                    import base64
                    
                    image_data = last_msg["images"][0]
                    
                    # Handle base64 data URLs (from Open WebUI)
                    if image_data.startswith('data:image'):
                        # Extract base64 data
                        base64_data = image_data.split(',', 1)[1]
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(BytesIO(image_bytes))
                    elif image_data.startswith('http'):
                        # Handle regular URLs
                        import httpx
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(image_data)
                            image = Image.open(BytesIO(response.content))
                    else:
                        # Assume it's raw base64
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes))
                    
                    # Analyze with BLIP
                    caption = await models.hf_vision_analyze(image, last_msg["content"])
                    
                    full_response.append(caption)
                    
                    chunk_escaped = caption.replace('"', '\\"').replace('\n', '\\n')
                    yield f'data: {{"id":"{chat_id}","model":"{request.model}","choices":[{{"delta":{{"content":"{chunk_escaped}"}},"index":0}}]}}\n\n'
                    
                    stream = None
                except Exception as e:
                    logger.error(f"Vision error: {e}")
                    error_msg = f"[Error analyzing image: {str(e)}]"
                    error_escaped = error_msg.replace('"', '\\"')
                    yield f'data: {{"id":"{chat_id}","choices":[{{"delta":{{"content":"{error_escaped}"}},"index":0}}]}}\n\n'
                    stream = None
            else:
                error_msg = "No image provided for vision request"
                error_escaped = error_msg.replace('"', '\\"')
                yield f'data: {{"id":"{chat_id}","choices":[{{"delta":{{"content":"{error_escaped}"}},"index":0}}]}}\n\n'
                yield f'data: {{"id":"{chat_id}","model":"{request.model}","choices":[{{"delta":{{}},"index":0,"finish_reason":"stop"}}]}}\n\n'
                yield "data: [DONE]\n\n"
                return
        
        elif backend == "groq":
            # Groq streaming
            stream = models.groq_stream(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        
        elif backend == "auto":
            # Smart routing
            stream = models.unified_stream(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        
        else:
            # Default to Groq
            stream = models.groq_stream(messages)
        
        # Process stream
        if stream:
            async for chunk_text in stream:
                full_response.append(chunk_text)
                
                chunk_escaped = chunk_text.replace('"', '\\"').replace('\n', '\\n')
                yield f'data: {{"id":"{chat_id}","model":"{request.model}","choices":[{{"delta":{{"content":"{chunk_escaped}"}},"index":0}}]}}\n\n'
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        
        error_msg = f"[Error: {str(e)}]"
        error_escaped = error_msg.replace('"', '\\"')
        yield f'data: {{"id":"{chat_id}","choices":[{{"delta":{{"content":"{error_escaped}"}},"index":0}}]}}\n\n'
    
    # Send finish
    yield f'data: {{"id":"{chat_id}","model":"{request.model}","choices":[{{"delta":{{}},"index":0,"finish_reason":"stop"}}]}}\n\n'
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """Chat completion endpoint"""
    
    if request.model not in MODELS:
        logger.warning(f"Unknown model '{request.model}', using aegis-auto")
        request.model = "aegis-auto"
    
    logger.info(f"📨 {req.client.host} | {request.model} | {len(request.messages)}msg")
    
    return StreamingResponse(
        stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================================
# AUDIO TRANSCRIPTION
# ============================================================================

@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
):
    """Speech-to-text with Whisper Tiny"""
    
    if not settings.enable_hf_models:
        raise HTTPException(status_code=501, detail="Requires HF models")
    
    logger.info(f"🎤 STT: {file.filename}")
    
    try:
        audio_data = await file.read()
        
        import librosa
        audio_array, _ = librosa.load(BytesIO(audio_data), sr=16000)
        
        transcription = await models.hf_speech_to_text(audio_array)
        
        return {
            "text": transcription,
            "language": "en",
        }
        
    except Exception as e:
        logger.error(f"❌ STT error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EMBEDDINGS
# ============================================================================

@router.post("/v1/embeddings")
async def create_embeddings(request: dict):
    """Embeddings endpoint"""
    
    if not settings.enable_hf_models:
        raise HTTPException(status_code=501, detail="Requires HF models")
    
    texts = request.get("input", [])
    if isinstance(texts, str):
        texts = [texts]
    
    logger.info(f"📊 Embeddings: {len(texts)} texts")
    
    try:
        embeddings = await models.hf_get_embeddings(texts)
        
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb,
                    "index": i,
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": "aegis-embeddings",
        }
    
    except Exception as e:
        logger.error(f"❌ Embeddings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# METRICS & HEALTH
# ============================================================================

@router.get("/v1/metrics")
async def get_metrics():
    """Performance metrics"""
    
    return {
        "models": MODELS,
        "backends": models.get_metrics(),
        "cache": get_cache_stats(),
        "system": {
            "mode": "ultra-lightweight",
            "total_disk_usage": "~6GB",
            "local_models": ["DeepSeek Coder", "BLIP Vision", "MiniLM", "Whisper Tiny"],
            "primary_engine": "Groq (0 disk)",
            "removed_features": REMOVED_FEATURES_INFO,
        }
    }


@router.get("/health")
@router.get("/v1/health")
async def health_check():
    """Health check with disk usage info"""
    
    import psutil
    
    try:
        ram_free_gb = psutil.virtual_memory().available / (1024**3)
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "models": {
                "total": len(MODELS),
                "groq": 1,
                "local": 4,
            },
            "system": {
                "mode": "ultra-lightweight",
                "disk_usage": "~6GB total",
                "free_ram_gb": round(ram_free_gb, 1),
                "recommendation": "Use aegis-groq-turbo for best performance (0 disk/RAM)",
            },
            "components": {
                "groq": "operational",
                "hf_models": "enabled" if settings.enable_hf_models else "disabled",
                "cache": "enabled" if settings.enable_caching else "disabled",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@router.get("/v1/features")
async def get_features():
    """Show available features and alternatives for removed ones"""
    
    return {
        "available": {
            "chat": "✅ Groq Llama 3.3 70B (800 tok/s, 0 disk)",
            "code": "✅ DeepSeek Coder 1.3B (local, 1.3GB)",
            "vision": "✅ BLIP (local, 800MB)",
            "embeddings": "✅ MiniLM (local, 80MB)",
            "speech_to_text": "✅ Whisper Tiny (local, 150MB)",
        },
        "removed_to_save_space": {
            "reasoning_model": {
                "removed": "Phi-2 (4GB)",
                "alternative": "Use aegis-groq-turbo (same quality, 0 disk)",
            },
            "image_generation": {
                "removed": "SDXL Turbo (7GB)",
                "alternatives": [
                    "Stability AI API (https://platform.stability.ai/)",
                    "Replicate (https://replicate.com/)",
                    "ComfyUI (if you have GPU)",
                    "DALL-E API",
                ],
            },
        },
        "total_disk_saved": "~6GB (from 15GB → 6GB, included vision!)",
    }


_startup_time = time.time()

__all__ = ['router', 'MODELS']