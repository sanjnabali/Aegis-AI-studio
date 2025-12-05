# File: backend/app/core/models.py

import os
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

llm_vision = None
llm_code = None

# Get the correct path to models folder
# Current file: backend/app/core/models.py
# Models folder: backend/app/models
# So we go up one level from 'core' to 'app', then into 'models'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets backend/app/core
APP_DIR = os.path.dirname(CURRENT_DIR)  # Gets backend/app
MODELS_DIR = os.path.join(APP_DIR, "models")  # Gets backend/app/models

def load_vision_model():
    """Loads the Chat/Vision model (LLaVA) into memory."""
    global llm_vision
    
    # Debug: Print the path we're looking for
    print(f"Looking for models in: {MODELS_DIR}")
    
    # Check if directory exists
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found at: {MODELS_DIR}")
    
    # List all files to help debug
    print(f"Files in models directory: {os.listdir(MODELS_DIR)}")
    
    # Find the model files
    model_files = os.listdir(MODELS_DIR)
    
    # Find llava model (brain)
    llava_file = [f for f in model_files if f.startswith("llava") and f.endswith(".gguf")]
    if not llava_file:
        raise FileNotFoundError(f"LLaVA model not found in {MODELS_DIR}")
    model_path = os.path.join(MODELS_DIR, llava_file[0])
    
    # Find mmproj model (eyes)
    mmproj_file = [f for f in model_files if f.startswith("mmproj") and f.endswith(".gguf")]
    if not mmproj_file:
        raise FileNotFoundError(f"MMPROJ model not found in {MODELS_DIR}")
    clip_path = os.path.join(MODELS_DIR, mmproj_file[0])
    
    print(f"✓ Found VISION model: {llava_file[0]}")
    print(f"✓ Found CLIP model: {mmproj_file[0]}")
    print(f"Loading models into memory... (this may take a minute)")

    # Create the "chat_handler" and give it the "eyes" file
    chat_handler = Llava15ChatHandler(clip_model_path=clip_path, verbose=True)
    
    # Load the "brain" and pass it the "eyes" handler
    llm_vision = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=0,  # Set to -1 if you have NVIDIA GPU with CUDA
        verbose=True,
        n_threads=4 
    )
    
    print("✓ VISION model loaded successfully!")

def load_code_model():
    """Loads the Code Specialist model (CodeLlama) into memory."""
    global llm_code
    
    print(f"Looking for CODE model in: {MODELS_DIR}")
    
    # Check if directory exists
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found at: {MODELS_DIR}")
    
    # Find codellama model
    model_files = os.listdir(MODELS_DIR)
    codellama_file = [f for f in model_files if f.startswith("codellama") and f.endswith(".gguf")]
    if not codellama_file:
        raise FileNotFoundError(f"CodeLlama model not found in {MODELS_DIR}")
    model_path = os.path.join(MODELS_DIR, codellama_file[0])
    
    print(f"✓ Found CODE model: {codellama_file[0]}")
    print(f"Loading model into memory... (this may take a minute)")
    
    llm_code = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,  # Set to -1 if you have NVIDIA GPU with CUDA
        verbose=True,
        n_threads=4
    )
    
    print("✓ CODE model loaded successfully!")

def get_vision_llm():
    """A helper function to get the loaded VISION model."""
    if llm_vision is None:
        raise RuntimeError("Vision model has not been loaded!")
    return llm_vision

def get_code_llm():
    """A helper function to get the loaded CODE model."""
    if llm_code is None:
        raise RuntimeError("Code model has not been loaded!")
    return llm_code