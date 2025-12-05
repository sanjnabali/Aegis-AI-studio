# File: backend/Dockerfile

# Step 1: Base OS (Debian Bookworm)
FROM python:3.11-slim-bookworm

WORKDIR /app

# Step 2: Install system dependencies
RUN apt-get update && apt-get install -y build-essential cmake pkg-config && rm -rf /var/lib/apt/lists/*

# Step 3: Copy requirements file (path is from the root)
COPY backend/requirements.txt .

# Step 4: Install our Python libraries
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- This ensures all 3 files are copied with the correct names ---
# Step 5: Copy ALL THREE models
RUN mkdir /app/models
COPY models/llava-v1.6-mistral-7b.Q4_K_M.gguf /app/models/llava-v1.6-mistral-7b.Q4_K_M.gguf
COPY models/codellama-7b-instruct.Q4_K_M.gguf /app/models/codellama-7b-instruct.Q4_K_M.gguf
COPY models/mmproj-model-f16.gguf /app/models/mmproj-model-f16.gguf

# Step 6: Verify the files
RUN ls -lh /app/models

# Step 7: Copy our application code (path is from the root)
COPY ./backend/app /app/app

# Step 8: Define the run command
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]