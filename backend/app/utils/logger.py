"""
Advanced Logging Configuration
===============================
Structured logging with context, performance tracking, and log aggregation.
"""

import sys
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

from loguru import logger


class StructuredLogger:
    """
    Enhanced logger with structured output and context tracking.
    """
    
    def __init__(self):
        self.context = {}
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure Loguru with custom formatting"""
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colors (for development)
        logger.add(
            sys.stdout,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        
        # JSON file handler (for production/parsing)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "aegis_{time:YYYY-MM-DD}.json",
            format="{message}",
            level="INFO",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            serialize=True,  # JSON serialization
        )
        
        # Error file handler
        logger.add(
            log_dir / "aegis_errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="00:00",
            retention="90 days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
    
    def set_context(self, **kwargs):
        """Set context variables for all subsequent logs"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()
    
    def log_with_context(self, level: str, message: str, **extra):
        """Log with context and extra data"""
        log_data = {
            **self.context,
            **extra,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        getattr(logger, level.lower())(json.dumps(log_data))
    
    def log_api_request(
        self,
        method: str,
        path: str,
        client_ip: str,
        request_id: str,
    ):
        """Log API request"""
        self.log_with_context(
            "info",
            f"API Request: {method} {path}",
            method=method,
            path=path,
            client_ip=client_ip,
            request_id=request_id,
            type="api_request",
        )
    
    def log_api_response(
        self,
        request_id: str,
        status_code: int,
        duration_ms: float,
    ):
        """Log API response"""
        self.log_with_context(
            "info",
            f"API Response: {status_code} in {duration_ms:.2f}ms",
            request_id=request_id,
            status_code=status_code,
            duration_ms=duration_ms,
            type="api_response",
        )
    
    def log_llm_call(
        self,
        backend: str,
        model: str,
        tokens: int,
        latency_ms: float,
    ):
        """Log LLM API call"""
        self.log_with_context(
            "info",
            f"LLM Call: {backend}/{model} - {tokens} tokens in {latency_ms:.0f}ms",
            backend=backend,
            model=model,
            tokens=tokens,
            latency_ms=latency_ms,
            type="llm_call",
        )
    
    def log_cache_hit(self, cache_key: str):
        """Log cache hit"""
        self.log_with_context(
            "debug",
            f"Cache HIT: {cache_key[:16]}...",
            cache_key=cache_key,
            type="cache_hit",
        )
    
    def log_cache_miss(self, cache_key: str):
        """Log cache miss"""
        self.log_with_context(
            "debug",
            f"Cache MISS: {cache_key[:16]}...",
            cache_key=cache_key,
            type="cache_miss",
        )


# Global logger instance
structured_logger = StructuredLogger()