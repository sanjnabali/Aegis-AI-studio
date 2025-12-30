"""
Input Validation Utilities
===========================
Validates and sanitizes user inputs to prevent errors and security issues.
"""

import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, validator, Field


class MessageValidator:
    """Validates chat messages"""
    
    @staticmethod
    def validate_role(role: str) -> bool:
        """Validate message role"""
        return role in ["system", "user", "assistant", "function"]
    
    @staticmethod
    def validate_content_length(content: str, max_length: int = 32000) -> bool:
        """Validate content length"""
        return len(content) <= max_length
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content to prevent injection attacks"""
        # Remove null bytes
        content = content.replace('\x00', '')
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    @staticmethod
    def validate_messages(messages: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
        """
        Validate a list of messages.
        
        Returns:
            (is_valid, error_message)
        """
        
        if not messages:
            return False, "Messages list cannot be empty"
        
        if len(messages) > 100:
            return False, "Too many messages (max: 100)"
        
        for idx, msg in enumerate(messages):
            # Check required fields
            if "role" not in msg:
                return False, f"Message {idx}: missing 'role' field"
            
            if "content" not in msg:
                return False, f"Message {idx}: missing 'content' field"
            
            # Validate role
            if not MessageValidator.validate_role(msg["role"]):
                return False, f"Message {idx}: invalid role '{msg['role']}'"
            
            # Validate content
            content = msg["content"]
            if isinstance(content, str):
                if not MessageValidator.validate_content_length(content):
                    return False, f"Message {idx}: content too long (max: 32000 chars)"
            elif isinstance(content, list):
                # Multimodal content
                for part_idx, part in enumerate(content):
                    if not isinstance(part, dict):
                        return False, f"Message {idx}, part {part_idx}: must be a dict"
                    
                    if "type" not in part:
                        return False, f"Message {idx}, part {part_idx}: missing 'type'"
        
        return True, None


class URLValidator:
    """Validates URLs for web scraping"""
    
    ALLOWED_SCHEMES = ["http", "https"]
    BLOCKED_DOMAINS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "192.168.",
        "10.",
        "172.16.",
    ]
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid and safe"""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in URLValidator.ALLOWED_SCHEMES:
                return False
            
            # Check if netloc exists
            if not parsed.netloc:
                return False
            
            # Check for blocked domains (prevent SSRF)
            for blocked in URLValidator.BLOCKED_DOMAINS:
                if blocked in parsed.netloc:
                    return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL"""
        # Remove whitespace
        url = url.strip()
        
        # Add https if no scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        return url


class CodeValidator:
    """Validates code for safe execution"""
    
    DANGEROUS_IMPORTS = [
        "os",
        "sys",
        "subprocess",
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "file",
        "input",
        "raw_input",
    ]
    
    @staticmethod
    def is_safe_code(code: str) -> tuple[bool, Optional[str]]:
        """
        Check if code is safe to execute.
        
        Returns:
            (is_safe, warning_message)
        """
        
        code_lower = code.lower()
        
        # Check for dangerous imports
        for dangerous in CodeValidator.DANGEROUS_IMPORTS:
            if dangerous in code_lower:
                return False, f"Code contains potentially dangerous operation: {dangerous}"
        
        # Check for infinite loops
        if "while True:" in code or "while 1:" in code:
            return False, "Code may contain infinite loop"
        
        # Check code length
        if len(code) > 10000:
            return False, "Code too long (max: 10000 characters)"
        
        return True, None


class QueryValidator:
    """Validates search queries"""
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Sanitize search query"""
        # Remove excess whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove special characters that could break search
        query = re.sub(r'[^\w\s\-\'".,?!]', '', query)
        
        return query
    
    @staticmethod
    def is_valid_query(query: str) -> tuple[bool, Optional[str]]:
        """
        Validate search query.
        
        Returns:
            (is_valid, error_message)
        """
        
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 500:
            return False, "Query too long (max: 500 characters)"
        
        # Check for suspicious patterns
        if re.search(r'<script|javascript:|data:', query, re.IGNORECASE):
            return False, "Query contains suspicious content"
        
        return True, None


# Pydantic models for request validation
class ChatRequest(BaseModel):
    """Validated chat request"""
    model: str = Field(..., min_length=1, max_length=100)
    messages: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8000, ge=1, le=32000)
    stream: bool = Field(default=True)
    
    @validator('messages')
    def validate_messages_list(cls, v):
        is_valid, error = MessageValidator.validate_messages(v)
        if not is_valid:
            raise ValueError(error)
        return v


class WebSearchRequest(BaseModel):
    """Validated web search request"""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    
    @validator('query')
    def validate_query_content(cls, v):
        is_valid, error = QueryValidator.is_valid_query(v)
        if not is_valid:
            raise ValueError(error)
        return QueryValidator.sanitize_query(v)


class WebScrapeRequest(BaseModel):
    """Validated web scrape request"""
    url: str = Field(..., min_length=1, max_length=2000)
    
    @validator('url')
    def validate_url_safety(cls, v):
        v = URLValidator.sanitize_url(v)
        if not URLValidator.is_valid_url(v):
            raise ValueError("Invalid or unsafe URL")
        return v


class CodeExecutionRequest(BaseModel):
    """Validated code execution request"""
    code: str = Field(..., min_length=1, max_length=10000)
    timeout: int = Field(default=5, ge=1, le=30)
    
    @validator('code')
    def validate_code_safety(cls, v):
        is_safe, warning = CodeValidator.is_safe_code(v)
        if not is_safe:
            raise ValueError(warning)
        return v