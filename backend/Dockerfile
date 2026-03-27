
```dockerfile
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Final stage
# ============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/models /app/cache /app/logs

# Copy application code
COPY app /app/app

# Set
context: backend
file: backend/Dockerfile
