# File: backend/app/main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from app import openai_adapter
from app.core import models
# from app.core import rag_memory # We will add this in the next milestone

# --- This is the "lifespan" function ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs *before* the server starts.
    It's the perfect place to load our AI models.
    """
    print("--- Server is starting up ---")
    
    # Load both models into memory
    models.load_vision_model()
    models.load_code_model()
    
    # We will initialize this in the next milestone
    # rag_memory.initialize_memory()
    
    yield
    # Code here would run on shutdown
    print("--- Server is shutting down ---")

# --- Tell FastAPI to use our lifespan function ---
app = FastAPI(
    title="Aegis AI Backend",
    description="The 'brain' of the Aegis AI Studio. This is a private, FOSS-first agentic backend.",
    version="1.0.0",
    lifespan=lifespan
)

# "Mount" the router
app.include_router(openai_adapter.router)

@app.get("/")
def get_root():
    """
    A simple root endpoint to confirm the server is alive.
    """
    return {"message": "Aegis backend is alive!"}

@app.get("/ping")
def get_ping():
    """
    A simple health check endpoint.
    """
    return {"status": "ok", "message": "pong"}