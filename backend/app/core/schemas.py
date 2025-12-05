# File: backend/app/core/schemas.py
# This file defines the "shapes" of our API data.

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# --- THIS IS THE KEY CHANGE ---
# We need to define what an "image part" looks like
class ImageURL(BaseModel):
    url: str

class ContentPart(BaseModel):
    type: str  # Will be "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

# 'content' can now be a simple string OR a list of ContentParts
ContentType = Union[str, List[ContentPart]]

# This is the shape of a single message
class ChatMessage(BaseModel):
    role: str
    content: ContentType  # <-- This is now upgraded

# This is the shape of the *entire request* from Open WebUI
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    
    class Config:
        extra = "allow"

# --- (The rest of the file is identical to before) ---

# --- These models define the "streaming" response ---
class StreamDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None

class StreamChoice(BaseModel):
    delta: StreamDelta
    index: int = 0

class ChatCompletionStreamResponse(BaseModel):
    id: str = "chatcmpl-aegis"
    model: str
    object: str = "chat.completion.chunk"
    choices: List[StreamChoice]

# --- These models define the "/v1/models" response ---
class ModelData(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "aegis"
    permission: list = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelData]