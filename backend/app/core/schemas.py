"""
Pydantic Schemas - Complete Data Models (All Features Enabled)
==============================================================
Supports: Chat, Web Search, Code Execution, Image Generation,
Image Analysis, Voice Optimization, and Agent Routing
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
import time


# ============================================================================
# ENUMS
# ============================================================================

class MessageRole(str, Enum):
    """Valid message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class ContentType(str, Enum):
    """Content types for multimodal messages"""
    TEXT = "text"
    IMAGE_URL = "image_url"


class TaskType(str, Enum):
    """Agent task types"""
    CHAT = "chat"
    CODE = "code"
    SEARCH = "search"
    RESEARCH = "research"
    IMAGE_GEN = "image_generation"
    IMAGE_ANALYZE = "image_analysis"


# ============================================================================
# CHAT COMPLETION MODELS (Core)
# ============================================================================

class ImageURL(BaseModel):
    """Image URL container"""
    url: str = Field(..., description="URL of the image")
    detail: Literal["auto", "low", "high"] = "auto"


class ContentPart(BaseModel):
    """Content part for multimodal messages"""
    type: ContentType
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v, info):
        if info.data.get('type') == ContentType.TEXT and not v:
            raise ValueError("Text required for text type")
        return v
    
    @field_validator('image_url')
    @classmethod
    def validate_image(cls, v, info):
        if info.data.get('type') == ContentType.IMAGE_URL and not v:
            raise ValueError("Image URL required for image_url type")
        return v


class ChatMessage(BaseModel):
    """Single chat message"""
    role: MessageRole
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)"""
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=100)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=32000)
    stream: bool = Field(default=True)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    
    # Extended parameters
    response_format: Optional[Dict[str, str]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"


class StreamDelta(BaseModel):
    """Streaming response delta"""
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class StreamChoice(BaseModel):
    """Streaming choice"""
    delta: StreamDelta
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response chunk"""
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


# ============================================================================
# MODEL LIST MODELS
# ============================================================================

class ModelPermission(BaseModel):
    """Model permissions"""
    id: str = "modelperm-aegis"
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = False
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    is_blocking: bool = False


class ModelData(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "aegis"
    permission: List[ModelPermission] = Field(default_factory=list)
    
    # Extended model info
    description: Optional[str] = None
    speed: Optional[str] = None
    context_length: Optional[str] = None
    best_for: Optional[str] = None


class ModelList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelData]


# ============================================================================
# WEB SEARCH MODELS
# ============================================================================

class WebSearchRequest(BaseModel):
    """Web search request"""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    search_type: Literal["web", "news", "instant"] = "web"
    region: str = Field(default="wt-wt", description="Search region")
    time_range: Optional[Literal["d", "w", "m", "y"]] = None


class WebSearchResult(BaseModel):
    """Single search result"""
    position: int
    title: str
    snippet: str
    url: str
    source: Optional[str] = None
    date: Optional[str] = None
    relevance_score: Optional[float] = None


class WebSearchResponse(BaseModel):
    """Search results response"""
    query: str
    results: List[WebSearchResult]
    total_results: int
    search_time_ms: float
    cached: bool = False


# ============================================================================
# WEB SCRAPING MODELS
# ============================================================================

class WebScrapeRequest(BaseModel):
    """Web scraping request"""
    url: str = Field(..., description="URL to scrape")
    max_length: int = Field(default=10000, ge=100, le=50000)
    extract_links: bool = Field(default=False)
    extract_images: bool = Field(default=False)
    clean_html: bool = Field(default=True)


class WebScrapeResponse(BaseModel):
    """Scraping result"""
    success: bool
    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    html: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    links: Optional[List[str]] = None
    images: Optional[List[str]] = None
    length: int = 0
    scrape_time: float = 0.0
    error: Optional[str] = None


# ============================================================================
# RESEARCH MODELS
# ============================================================================

class ResearchRequest(BaseModel):
    """Research request"""
    query: str = Field(..., min_length=1, max_length=500)
    num_sources: int = Field(default=3, ge=1, le=10)
    include_news: bool = Field(default=False)
    deep_analysis: bool = Field(default=True)
    max_content_per_source: int = Field(default=5000, ge=1000, le=20000)


class ResearchSource(BaseModel):
    """Single research source"""
    url: str
    title: str
    content: str
    relevance: float
    word_count: int


class ResearchResponse(BaseModel):
    """Research results"""
    query: str
    status: Literal["completed", "partial", "failed"]
    sources: List[ResearchSource]
    summary: Optional[str] = None
    key_findings: Optional[List[str]] = None
    sources_analyzed: int
    total_words: int
    research_time: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# CODE EXECUTION MODELS
# ============================================================================

class CodeExecutionRequest(BaseModel):
    """Code execution request"""
    code: str = Field(..., min_length=1, max_length=10000)
    language: Literal["python", "javascript", "bash"] = "python"
    timeout: int = Field(default=5, ge=1, le=30)
    capture_output: bool = Field(default=True)
    input_data: Optional[str] = None
    packages: Optional[List[str]] = None


class CodeExecutionResponse(BaseModel):
    """Code execution result"""
    success: bool
    output: str
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: float
    exit_code: int = 0
    language: str
    memory_used: Optional[int] = None


# ============================================================================
# IMAGE GENERATION MODELS (HuggingFace)
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """Image generation request (SDXL Turbo)"""
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: Optional[str] = None
    num_steps: int = Field(default=1, ge=1, le=50)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=20.0)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    seed: Optional[int] = None


class ImageGenerationResponse(BaseModel):
    """Image generation result"""
    success: bool
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    prompt: str
    negative_prompt: Optional[str] = None
    width: int
    height: int
    num_steps: int
    generation_time: float
    seed: Optional[int] = None
    error: Optional[str] = None


# ============================================================================
# IMAGE ANALYSIS MODELS (HuggingFace BLIP)
# ============================================================================

class ImageAnalysisRequest(BaseModel):
    """Image analysis request"""
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    analysis_type: Literal["caption", "detailed", "objects"] = "caption"
    max_length: int = Field(default=100, ge=20, le=200)
    
    @field_validator('image_url', 'image_base64')
    @classmethod
    def validate_image_source(cls, v, info):
        values = info.data
        if not values.get('image_url') and not values.get('image_base64'):
            raise ValueError("Either image_url or image_base64 required")
        return v


class ImageAnalysisResponse(BaseModel):
    """Image analysis result"""
    success: bool
    caption: Optional[str] = None
    detailed_description: Optional[str] = None
    objects: Optional[List[str]] = None
    confidence: Optional[float] = None
    analysis_time: float
    analysis_type: str
    error: Optional[str] = None


# ============================================================================
# AUDIO MODELS (TTS/STT)
# ============================================================================

class TextToSpeechRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(..., min_length=1, max_length=4000)
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "nova"
    model: Literal["tts-1", "tts-1-hd"] = "tts-1"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: Literal["mp3", "opus", "aac", "flac"] = "mp3"


class TextToSpeechResponse(BaseModel):
    """Text-to-speech result"""
    success: bool
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    duration: Optional[float] = None
    format: str
    generation_time: float
    error: Optional[str] = None


class SpeechToTextRequest(BaseModel):
    """Speech-to-text request"""
    audio_url: Optional[str] = None
    audio_base64: Optional[str] = None
    language: str = Field(default="en-US")
    model: Literal["whisper-tiny", "whisper-base", "whisper-small"] = "whisper-tiny"


class SpeechToTextResponse(BaseModel):
    """Speech-to-text result"""
    success: bool
    text: str
    language: str
    confidence: Optional[float] = None
    duration: Optional[float] = None
    transcription_time: float
    error: Optional[str] = None


# ============================================================================
# VOICE OPTIMIZATION MODELS
# ============================================================================

class VoiceDetectionRequest(BaseModel):
    """Voice query detection"""
    text: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VoiceOptimizationResponse(BaseModel):
    """Voice-optimized response"""
    original_text: str
    optimized_text: str
    is_voice_query: bool
    confidence: float
    optimizations_applied: List[str]
    tokens_saved: int


# ============================================================================
# AGENT MODELS
# ============================================================================

class AgentTask(BaseModel):
    """Agent task definition"""
    type: TaskType
    query: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None
    max_iterations: int = Field(default=3, ge=1, le=10)


class AgentStep(BaseModel):
    """Single agent execution step"""
    step_number: int
    action: str
    observation: str
    thought: Optional[str] = None
    tool_used: Optional[str] = None


class AgentResponse(BaseModel):
    """Agent execution result"""
    task_type: str
    success: bool
    result: Any
    steps: List[AgentStep]
    total_steps: int
    execution_time: float
    agent_used: str
    tokens_used: Optional[int] = None


class AgentStats(BaseModel):
    """Agent routing statistics"""
    total_requests: int
    tasks_by_type: Dict[str, int]
    average_execution_time: float
    success_rate: float
    most_used_tools: List[str]


# ============================================================================
# SYSTEM MODELS
# ============================================================================

class HealthStatus(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    uptime_seconds: int
    components: Dict[str, str]
    models_available: Optional[List[str]] = None


class SystemStats(BaseModel):
    """System statistics"""
    uptime_seconds: int
    total_requests: int
    backends: Dict[str, Any]
    cache: Dict[str, Any]
    features: Dict[str, bool]
    configuration: Dict[str, Any]
    performance: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    groq: Dict[str, Any]
    huggingface: Dict[str, Any]
    cache: Dict[str, Any]
    average_latency_ms: float
    requests_per_second: float


class ErrorResponse(BaseModel):
    """Error response"""
    error: Dict[str, str]
    request_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# CACHE MODELS
# ============================================================================

class CacheStats(BaseModel):
    """Cache statistics"""
    enabled: bool
    hits: int
    misses: int
    errors: int
    hit_rate: str
    total_requests: int
    memory_used: Optional[str] = None
    keys_count: Optional[int] = None


class CacheInvalidationRequest(BaseModel):
    """Cache invalidation request"""
    pattern: str = Field(..., description="Key pattern to invalidate")
    confirm: bool = Field(default=False)


# ============================================================================
# RATE LIMIT MODELS
# ============================================================================

class RateLimitInfo(BaseModel):
    """Rate limit information"""
    limit: int
    remaining: int
    reset_at: str
    window_seconds: int = 60


class RateLimitStatus(BaseModel):
    """Current rate limit status"""
    ip: RateLimitInfo
    groq: Optional[RateLimitInfo] = None


# ============================================================================
# EMBEDDINGS MODELS (For RAG/Semantic Search)
# ============================================================================

class EmbeddingsRequest(BaseModel):
    """Embeddings generation request"""
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    model: str = Field(default="all-MiniLM-L6-v2")
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    """Single embedding"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    """Embeddings response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


# ============================================================================
# DOCUMENT PROCESSING MODELS (For RAG)
# ============================================================================

class DocumentUploadRequest(BaseModel):
    """Document upload for RAG"""
    content: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    overlap: int = Field(default=200, ge=0, le=1000)


class DocumentUploadResponse(BaseModel):
    """Document upload result"""
    success: bool
    document_id: str
    chunks_created: int
    total_tokens: int
    processing_time: float


class DocumentSearchRequest(BaseModel):
    """Search documents"""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)


class DocumentSearchResult(BaseModel):
    """Single search result"""
    document_id: str
    chunk_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]


class DocumentSearchResponse(BaseModel):
    """Document search results"""
    query: str
    results: List[DocumentSearchResult]
    total_results: int
    search_time: float


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "MessageRole",
    "ContentType",
    "TaskType",
    
    # Core Chat Models
    "ImageURL",
    "ContentPart",
    "ChatMessage",
    "ChatCompletionRequest",
    "StreamDelta",
    "StreamChoice",
    "ChatCompletionStreamResponse",
    
    # Model Management
    "ModelPermission",
    "ModelData",
    "ModelList",
    
    # Web Features
    "WebSearchRequest",
    "WebSearchResult",
    "WebSearchResponse",
    "WebScrapeRequest",
    "WebScrapeResponse",
    "ResearchRequest",
    "ResearchSource",
    "ResearchResponse",
    
    # Code Execution
    "CodeExecutionRequest",
    "CodeExecutionResponse",
    
    # Image Features
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageAnalysisRequest",
    "ImageAnalysisResponse",
    
    # Audio Features
    "TextToSpeechRequest",
    "TextToSpeechResponse",
    "SpeechToTextRequest",
    "SpeechToTextResponse",
    
    # Voice Optimization
    "VoiceDetectionRequest",
    "VoiceOptimizationResponse",
    
    # Agent System
    "AgentTask",
    "AgentStep",
    "AgentResponse",
    "AgentStats",
    
    # System
    "HealthStatus",
    "SystemStats",
    "PerformanceMetrics",
    "ErrorResponse",
    
    # Cache
    "CacheStats",
    "CacheInvalidationRequest",
    
    # Rate Limiting
    "RateLimitInfo",
    "RateLimitStatus",
    
    # Embeddings & RAG
    "EmbeddingsRequest",
    "EmbeddingData",
    "EmbeddingsResponse",
    "DocumentUploadRequest",
    "DocumentUploadResponse",
    "DocumentSearchRequest",
    "DocumentSearchResult",
    "DocumentSearchResponse",
]