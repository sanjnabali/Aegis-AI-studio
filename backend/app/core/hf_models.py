"""
HuggingFace Model Manager
=========================
Loads and manages lightweight HF models for specialized tasks.
"""

import os
import torch
from typing import Optional, List, Dict, Any
from pathlib import Path
from loguru import logger

# Lazy imports to avoid loading all models at startup
_code_model = None
_image_model = None
_image_caption_model = None
_tts_model = None
_stt_model = None
_embeddings_model = None

# Model paths (will download automatically on first use)
MODELS = {
    "code": {
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "size": "1.3B",
        "task": "Code generation",
        "memory": "~2GB"
    },
    "image_gen": {
        "name": "stabilityai/sdxl-turbo",
        "size": "6.9B",
        "task": "Image generation (1 step)",
        "memory": "~7GB"
    },
    "image_caption": {
        "name": "Salesforce/blip-image-captioning-base",
        "size": "400M",
        "task": "Image analysis/captioning",
        "memory": "~800MB"
    },
    "tts": {
        "name": "microsoft/speecht5_tts",
        "size": "200M",
        "task": "Text-to-speech",
        "memory": "~500MB"
    },
    "stt": {
        "name": "openai/whisper-tiny",
        "size": "39M",
        "task": "Speech-to-text",
        "memory": "~150MB"
    },
    "embeddings": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "size": "80M",
        "task": "Text embeddings/semantic search",
        "memory": "~200MB"
    }
}

# Check if GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"HuggingFace models will use: {DEVICE}")

# Cache directory
CACHE_DIR = Path("/app/models")
CACHE_DIR.mkdir(exist_ok=True)


class CodeModel:
    """Lightweight code generation model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
    
    def load(self):
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info("Loading DeepSeek Coder 1.3B...")
            model_name = MODELS["code"]["name"]
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=CACHE_DIR
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                cache_dir=CACHE_DIR
            )
            
            if DEVICE == "cpu":
                self.model = self.model.to(DEVICE)
            
            logger.success("✓ Code model loaded")
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        self.load()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()


class ImageGenModel:
    """Fast image generation (1-step SDXL Turbo)"""
    
    def __init__(self):
        self.pipe = None
    
    def load(self):
        if self.pipe is None:
            from diffusers import AutoPipelineForText2Image
            
            logger.info("Loading SDXL Turbo...")
            
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                MODELS["image_gen"]["name"],
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                cache_dir=CACHE_DIR
            )
            
            if DEVICE == "cuda":
                self.pipe = self.pipe.to(DEVICE)
            
            logger.success("✓ Image generation model loaded")
    
    def generate(self, prompt: str, num_steps: int = 1) -> Any:
        self.load()
        
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0
        ).images[0]
        
        return image


class ImageCaptionModel:
    """Image analysis and captioning"""
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    def load(self):
        if self.model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            logger.info("Loading BLIP captioning model...")
            
            self.processor = BlipProcessor.from_pretrained(
                MODELS["image_caption"]["name"],
                cache_dir=CACHE_DIR
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                MODELS["image_caption"]["name"],
                cache_dir=CACHE_DIR
            ).to(DEVICE)
            
            logger.success("✓ Image caption model loaded")
    
    def caption(self, image) -> str:
        self.load()
        
        inputs = self.processor(image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=100)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


class TTSModel:
    """Text-to-speech (lightweight)"""
    
    def __init__(self):
        self.model = None
        self.vocoder = None
        self.processor = None
    
    def load(self):
        if self.model is None:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
            
            logger.info("Loading TTS model...")
            
            self.processor = SpeechT5Processor.from_pretrained(
                MODELS["tts"]["name"],
                cache_dir=CACHE_DIR
            )
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                MODELS["tts"]["name"],
                cache_dir=CACHE_DIR
            ).to(DEVICE)
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                cache_dir=CACHE_DIR
            ).to(DEVICE)
            
            logger.success("✓ TTS model loaded")
    
    def synthesize(self, text: str) -> Any:
        self.load()
        
        inputs = self.processor(text=text, return_tensors="pt").to(DEVICE)
        
        # Load speaker embeddings (you need to provide these)
        # For now, using default
        import numpy as np
        speaker_embeddings = torch.tensor(np.random.randn(512)).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )
        
        return speech.cpu().numpy()


class STTModel:
    """Speech-to-text (Whisper tiny)"""
    
    def __init__(self):
        self.model = None
        self.processor = None
    
    def load(self):
        if self.model is None:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info("Loading Whisper Tiny...")
            
            self.processor = WhisperProcessor.from_pretrained(
                MODELS["stt"]["name"],
                cache_dir=CACHE_DIR
            )
            self.model = WhisperForConditionalGeneration.from_pretrained(
                MODELS["stt"]["name"],
                cache_dir=CACHE_DIR
            ).to(DEVICE)
            
            logger.success("✓ STT model loaded")
    
    def transcribe(self, audio_data) -> str:
        self.load()
        
        inputs = self.processor(
            audio_data,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs["input_features"])
        
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription


class EmbeddingsModel:
    """Text embeddings for semantic search"""
    
    def __init__(self):
        self.model = None
    
    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading embeddings model...")
            
            self.model = SentenceTransformer(
                MODELS["embeddings"]["name"],
                cache_folder=CACHE_DIR
            )
            
            logger.success("✓ Embeddings model loaded")
    
    def encode(self, texts: List[str]) -> Any:
        self.load()
        return self.model.encode(texts)


# Global instances (lazy loaded)
code_model = CodeModel()
image_gen_model = ImageGenModel()
image_caption_model = ImageCaptionModel()
tts_model = TTSModel()
stt_model = STTModel()
embeddings_model = EmbeddingsModel()


def get_available_models() -> Dict[str, Any]:
    """Get info about all available models"""
    return {
        "groq": {
            "status": "always_loaded",
            "models": ["llama-3.1-8b-instant"],
            "use_case": "General chat, fast responses"
        },
        "huggingface": {
            "status": "lazy_loaded",
            "models": MODELS
        }
    }