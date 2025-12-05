# File: backend/app/openai_adapter.py

import uuid
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.core import models
from app.core.schemas import (
    ChatCompletionRequest,
    StreamChoice,
    StreamDelta,
    ChatCompletionStreamResponse,
    ModelData,
    ModelList,
    ContentPart
)

router = APIRouter()

# --- 1. DEFINE OUR NEW MODEL NAMES ---
VISION_CHAT_MODEL_ID = "Aegis Vision (Chat & Vision)" 
CODE_MODEL_ID = "Aegis Code (Specialist)"
# We have removed the "Aegis Agent (Auto-Select)"

# --- 2. THE /v1/models ENDPOINT ---
@router.get("/v1/models")
async def get_models():
    """
    Returns a list of *only* our two specialist models.
    """
    return ModelList(
        data=[
            ModelData(id=VISION_CHAT_MODEL_ID),
            ModelData(id=CODE_MODEL_ID)
        ]
    )

# --- 3. THE SIMPLIFIED STREAM GENERATOR ---
async def real_stream_generator(request: ChatCompletionRequest):
    """
    This is now a *simple router*. It checks which model
    the user *manually* selected and calls it.
    This is much faster as it only runs the AI once.
    """
    
    # --- This is the new, simple "brain" ---
    if request.model == CODE_MODEL_ID:
        print("MANUAL: Routing request to CODE model")
        llm = models.get_code_llm()
        model_id = CODE_MODEL_ID
    
    else: # Default to the Vision/Chat model
        print("MANUAL: Routing request to VISION/CHAT model")
        llm = models.get_vision_llm()
        model_id = VISION_CHAT_MODEL_ID
    # --- End of "brain" ---

    # --- This code remains to fix the LLaVA bug ---
    # We must format the messages *differently* based on which model was chosen.
    
    messages_for_llm = []
    
    if model_id == VISION_CHAT_MODEL_ID:
        # LLaVA (Vision) model *always* expects content to be a list
        for msg in request.messages:
            new_content = []
            if isinstance(msg.content, str):
                new_content.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == "text":
                        new_content.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        new_content.append({"type": "image_url", "image_url": {"url": part.image_url.url}})
            messages_for_llm.append({"role": msg.role, "content": new_content})
    
    else:
        # CodeLlama model *only* wants simple strings
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages_for_llm.append({"role": msg.role, "content": msg.content})
            else:
                for part in msg.content:
                    if part.type == "text":
                        messages_for_llm.append({"role": msg.role, "content": part.text})
                        
    # --- END OF FIX ---

    # Call the *selected* model
    stream = llm.create_chat_completion(
        messages=messages_for_llm,
        stream=True,
    )

    # (The rest of this function is identical)
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    first_chunk_data = StreamChoice(delta=StreamDelta(role="assistant"), index=0)
    first_response = ChatCompletionStreamResponse(
        id=chat_id,
        model=model_id,
        choices=[first_chunk_data]
    )
    yield f"data: {first_response.model_dump_json()}\n\n"

    for chunk in stream:
        if chunk["choices"][0]["delta"].get("content"):
            chunk_data = StreamChoice(
                delta=StreamDelta(content=chunk["choices"][0]["delta"]["content"]),
                index=0
            )
            response = ChatCompletionStreamResponse(
                id=chat_id,
                model=model_id,
                choices=[chunk_data]
            )
            yield f"data: {response.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"


# --- 4. THE MAIN ENDPOINT ---
@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    This endpoint now calls the *real* agentic stream generator.
    """
    return StreamingResponse(
        real_stream_generator(request),
        media_type="text/event-stream"
    )