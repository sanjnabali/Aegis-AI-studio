from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str = ""
    gemini_api_key: str = ""  # Optional now
    openai_api_key: str = ""
    
    # Models
    groq_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 8000
    
    # Features
    enable_caching: bool = True
    enable_hf_models: bool = False  # NEW
    enable_web_search: bool = True
    enable_streaming: bool = True
    enable_voice_optimization: bool = True
    
    # Performance
    request_timeout: int = 120
    cache_ttl: int = 3600
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

_settings = None

def get_settings():
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings