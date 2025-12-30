"""
Ultra-Lightweight Model Manager - MINIMAL DISK USAGE
===================================================
Total: ~5GB models (vs 15GB optimized, 45GB full)
Strategy: Groq for most tasks + only essential local models
"""

import asyncio
import gc
import time
from typing import Optional, AsyncGenerator
from groq import AsyncGroq
from loguru import logger

from app.core.config import get_settings

settings = get_settings()

# ============================================================================
# GROQ CLIENT (0 DISK, 0 RAM - PRIMARY ENGINE)
# ============================================================================

_groq_client = None

def get_groq_client() -> AsyncGroq:
    """Get Groq client (cloud-based, 0 local storage)"""
    global _groq_client
    
    if _groq_client is None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        _groq_client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=settings.request_timeout,
        )
        logger.info("✓ Groq client initialized (0 disk/RAM)")
    
    return _groq_client


# ============================================================================
# MINIMAL LOCAL MODELS - ONLY ESSENTIALS
# ============================================================================

class UltraLightModels:
    """Minimal local models - total ~5GB disk"""
    
    def __init__(self):
        # Only keep these 3 essential models
        self.code_model = None          # ~1.3GB (essential for code)
        self.code_tokenizer = None
        self.embeddings_model = None    # ~80MB (essential for RAG)
        self.stt_model = None           # ~150MB (essential for voice)
        
        # Removed to save space:
        # - Reasoning model (use Groq instead)
        # - Vision model (use Groq Llama 3.2 Vision if available)
        # - Image generation (too slow on CPU, use external API)
        
        self._load_lock = asyncio.Lock()
        
        logger.info("🪶 Ultra-lightweight models (5GB total)")
    
    async def load_code(self):
        """Load DeepSeek Coder 1.3B (essential for code, 1.3GB)"""
        if self.code_model is not None:
            return
        
        async with self._load_lock:
            if self.code_model is not None:
                return
            
            logger.info("⏳ Loading DeepSeek Coder (1.3GB)...")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            loop = asyncio.get_event_loop()
            
            def _load():
                tokenizer = AutoTokenizer.from_pretrained(
                    "deepseek-ai/deepseek-coder-1.3b-instruct",
                    cache_dir="/app/models",
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/deepseek-coder-1.3b-instruct",
                    cache_dir="/app/models",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                
                return tokenizer, model
            
            self.code_tokenizer, self.code_model = await loop.run_in_executor(None, _load)
            
            logger.success("✓ DeepSeek Coder loaded (1.3GB)")
    
    async def load_embeddings(self):
        """Load MiniLM embeddings (essential for RAG, 80MB)"""
        if self.embeddings_model is not None:
            return
        
        async with self._load_lock:
            if self.embeddings_model is not None:
                return
            
            logger.info("⏳ Loading embeddings (80MB)...")
            
            from sentence_transformers import SentenceTransformer
            
            loop = asyncio.get_event_loop()
            
            def _load():
                return SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="/app/models",
                )
            
            self.embeddings_model = await loop.run_in_executor(None, _load)
            
            logger.success("✓ Embeddings loaded (80MB)")
    
    async def load_stt(self):
        """Load Whisper Tiny (essential for voice, 150MB)"""
        if self.stt_model is not None:
            return
        
        async with self._load_lock:
            if self.stt_model is not None:
                return
            
            logger.info("⏳ Loading Whisper Tiny (150MB)...")
            
            from transformers import pipeline
            
            loop = asyncio.get_event_loop()
            
            def _load():
                return pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",
                    cache_dir="/app/models",
                )
            
            self.stt_model = await loop.run_in_executor(None, _load)
            
            logger.success("✓ Whisper Tiny loaded (150MB)")


_hf_models = None

def _get_hf_models() -> Optional[UltraLightModels]:
    """Get ultra-lightweight models manager"""
    global _hf_models
    
    if not settings.enable_hf_models:
        return None
    
    if _hf_models is None:
        _hf_models = UltraLightModels()
    
    return _hf_models


# ============================================================================
# STREAMING FUNCTIONS
# ============================================================================

async def groq_stream(
    messages: list,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 8000,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Stream from Groq (primary engine, 0 disk/RAM)"""
    
    client = get_groq_client()
    
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Groq error: {e}")
        yield f"[Error: {str(e)}]"


async def hf_code_generate(prompt: str) -> str:
    """Generate code with DeepSeek (only local model for specialized task)"""
    
    hf = _get_hf_models()
    if not hf:
        # Fallback to Groq if HF disabled
        logger.info("→ HF disabled, using Groq for code")
        result = []
        async for chunk in groq_stream([{"role": "user", "content": prompt}]):
            result.append(chunk)
        return "".join(result)
    
    await hf.load_code()
    
    logger.info("→ Using DeepSeek Coder (local)")
    
    import torch
    
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    inputs = hf.code_tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(hf.code_model.device)
    
    loop = asyncio.get_event_loop()
    
    def _generate():
        with torch.no_grad():
            outputs = hf.code_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                pad_token_id=hf.code_tokenizer.eos_token_id,
            )
        
        return hf.code_tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
    
    return await loop.run_in_executor(None, _generate)


async def hf_speech_to_text(audio_array) -> str:
    """STT with Whisper Tiny (local for privacy)"""
    
    hf = _get_hf_models()
    if not hf:
        raise Exception("HuggingFace models not available")
    
    await hf.load_stt()
    
    logger.info("→ Using Whisper Tiny (local)")
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, hf.stt_model, audio_array)
    
    return result["text"]


async def hf_get_embeddings(texts: list) -> list:
    """Generate embeddings (local for RAG)"""
    
    hf = _get_hf_models()
    if not hf:
        raise Exception("HuggingFace models not available")
    
    await hf.load_embeddings()
    
    logger.info(f"→ Embeddings for {len(texts)} texts")
    
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, hf.embeddings_model.encode, texts)
    
    return embeddings.tolist()


async def unified_stream(
    messages: list,
    model: str = "auto",
    max_tokens: int = 8000,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """
    Ultra-lightweight routing
    Strategy: Use Groq for everything except code
    """
    
    last_message = messages[-1]["content"].lower() if messages else ""
    
    # Only use local model for code (DeepSeek is specialized)
    code_keywords = ["code", "function", "script", "debug", "implement", "program"]
    if any(kw in last_message for kw in code_keywords):
        logger.info("🎯 Routing: DeepSeek Coder (specialized for code)")
        result = await hf_code_generate(messages[-1]["content"])
        
        # Stream the result
        chunk_size = 50
        for i in range(0, len(result), chunk_size):
            yield result[i:i+chunk_size]
            await asyncio.sleep(0.01)
        return
    
    # Everything else → Groq (chat, reasoning, vision, etc.)
    logger.info("🎯 Routing: Groq Llama 3.3 70B (fast, 0 disk)")
    async for chunk in groq_stream(messages, max_tokens=max_tokens, temperature=temperature):
        yield chunk


# ============================================================================
# REMOVED FUNCTIONS (Use Groq or external APIs instead)
# ============================================================================

# Reasoning → Use Groq (same quality, 0 disk)
# Vision → Use Groq Llama 3.2 Vision API (when available) or external API
# Image Generation → Use external API (Stability AI, Replicate, etc.)


# ============================================================================
# METRICS
# ============================================================================

_metrics = {
    "groq": {"requests": 0, "errors": 0},
    "hf_code": {"requests": 0, "errors": 0},
}

def get_metrics() -> dict:
    return _metrics


__all__ = [
    'groq_stream',
    'hf_code_generate',
    'hf_speech_to_text',
    'hf_get_embeddings',
    'unified_stream',
    'get_groq_client',
    'get_metrics',
]