"""
Intelligent Model Router
========================
Routes requests to the best model based on task type.
"""

import re
from typing import Optional, Dict, Any
from loguru import logger

from app.core import models as groq_models
from app.core import hf_models


class ModelRouter:
    """Routes requests to appropriate model"""
    
    @staticmethod
    def detect_intent(prompt: str) -> str:
        """Detect what kind of task this is"""
        
        prompt_lower = prompt.lower()
        
        # Code-related keywords
        code_keywords = [
            "write code", "function", "class", "debug", "implement",
            "python", "javascript", "java", "c++", "rust", "go",
            "algorithm", "program", "script", "syntax"
        ]
        
        # Image generation keywords
        image_gen_keywords = [
            "generate image", "create image", "draw", "picture of",
            "illustration", "artwork", "visualize", "paint"
        ]
        
        # Image analysis keywords
        image_analysis_keywords = [
            "describe image", "what's in", "analyze image",
            "caption", "identify"
        ]
        
        # Check patterns
        if any(kw in prompt_lower for kw in code_keywords):
            return "code"
        
        if any(kw in prompt_lower for kw in image_gen_keywords):
            return "image_generation"
        
        if any(kw in prompt_lower for kw in image_analysis_keywords):
            return "image_analysis"
        
        # Default to chat
        return "chat"
    
    @staticmethod
    async def route(prompt: str, context: Optional[Dict] = None):
        """Route to appropriate model and return response"""
        
        intent = ModelRouter.detect_intent(prompt)
        logger.info(f"Detected intent: {intent}")
        
        if intent == "code":
            logger.info("→ Using DeepSeek Coder")
            response = hf_models.code_model.generate(prompt)
            return {
                "content": response,
                "model_used": "deepseek-coder-1.3b",
                "provider": "huggingface"
            }
        
        elif intent == "image_generation":
            logger.info("→ Using SDXL Turbo")
            image = hf_models.image_gen_model.generate(prompt)
            return {
                "content": "[Image generated]",
                "image": image,
                "model_used": "sdxl-turbo",
                "provider": "huggingface"
            }
        
        elif intent == "image_analysis":
            logger.info("→ Using BLIP")
            # Assuming image is provided in context
            if context and "image" in context:
                caption = hf_models.image_caption_model.caption(context["image"])
                return {
                    "content": caption,
                    "model_used": "blip",
                    "provider": "huggingface"
                }
            else:
                return {
                    "content": "Please provide an image to analyze",
                    "model_used": "none",
                    "provider": "none"
                }
        
        else:
            # Default to Groq for chat
            logger.info("→ Using Groq")
            messages = [{"role": "user", "content": prompt}]
            
            response_chunks = []
            async for chunk in groq_models.groq_stream(messages):
                response_chunks.append(chunk)
            
            return {
                "content": "".join(response_chunks),
                "model_used": "llama-3.3-70b",
                "provider": "groq"
            }


router = ModelRouter()