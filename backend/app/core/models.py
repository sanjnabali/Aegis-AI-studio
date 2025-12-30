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
from PIL import Image

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
    """Minimal local models - total ~6GB disk"""
    
    def __init__(self):
        # Essential models
        self.code_model = None          # ~1.3GB (essential for code)
        self.code_tokenizer = None
        self.vision_model = None        # ~800MB (for image analysis)
        self.vision_processor = None
        self.embeddings_model = None    # ~80MB (essential for RAG)
        self.stt_model = None           # ~150MB (essential for voice)
        
        # Removed to save space:
        # - Reasoning model (use Groq instead)
        # - Image generation (too slow on CPU, use external API)
        
        self._load_lock = asyncio.Lock()
        
        logger.info("🪶 Ultra-lightweight models (6GB total with vision)")
    
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
    
    async def load_vision(self):
        """Load BLIP for vision (essential for image analysis, 800MB)"""
        if self.vision_model is not None:
            return
        
        async with self._load_lock:
            if self.vision_model is not None:
                return
            
            logger.info("⏳ Loading BLIP vision (800MB)...")
            
            # Import with proper error handling
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                import torch
            except Exception as e:
                logger.error(f"Failed to import vision libraries: {e}")
                raise
            
            loop = asyncio.get_event_loop()
            
            def _load():
                # Load with CPU device explicitly
                processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    cache_dir="/app/models",
                )
                
                model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    cache_dir="/app/models",
                    torch_dtype=torch.float32,  # Use float32 for CPU stability
                )
                
                # Move to CPU explicitly
                model = model.to('cpu')
                model.eval()  # Set to evaluation mode
                
                return processor, model
            
            self.vision_processor, self.vision_model = await loop.run_in_executor(None, _load)
            
            logger.success("✓ BLIP vision loaded (800MB)")
    
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


async def hf_vision_analyze(image, prompt: str = "Describe this image in detail") -> str:
    """Analyze image with BLIP (local for privacy)"""
    
    hf = _get_hf_models()
    if not hf:
        return "[HuggingFace models not available]"
    
    await hf.load_vision()
    
    logger.info("→ Using BLIP for vision (local)")
    
    loop = asyncio.get_event_loop()
    
    def _analyze():
        import torch
        
        try:
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image_rgb = image.convert('RGB')
            else:
                image_rgb = image
            
            # Resize if too large (BLIP works best with reasonable sizes)
            max_size = 512
            if max(image_rgb.size) > max_size:
                ratio = max_size / max(image_rgb.size)
                new_size = tuple(int(dim * ratio) for dim in image_rgb.size)
                image_rgb = image_rgb.resize(new_size, Image.LANCZOS)
            
            # Process image for BLIP - unconditional captioning (most reliable)
            with torch.no_grad():  # Disable gradient computation
                # Process single image
                pixel_values = hf.vision_processor(
                    images=image_rgb,
                    return_tensors="pt"
                ).pixel_values
                
                # Move to same device as model
                pixel_values = pixel_values.to(hf.vision_model.device)
                
                # Generate caption
                generated_ids = hf.vision_model.generate(
                    pixel_values=pixel_values,
                    max_length=100,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                )
                
                # Decode output
                caption = hf.vision_processor.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
            
            # Clean up caption
            caption = caption.strip()
            
            # Add context if specific question was asked
            if prompt and prompt.lower() not in [
                "describe this image",
                "describe this image in detail",
                "what's in this image?",
                "what is this?",
                "analyze the image",
                "what do you see?"
            ]:
                caption = f"Image shows: {caption}\n\nRegarding your question '{prompt}': Based on the image, {caption.lower()}"
            
            return caption
            
        except Exception as e:
            logger.error(f"Error in vision analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    result = await loop.run_in_executor(None, _analyze)
    
    return result


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
    'hf_vision_analyze',
    'hf_speech_to_text',
    'hf_get_embeddings',
    'unified_stream',
    'get_groq_client',
    'get_metrics',
]