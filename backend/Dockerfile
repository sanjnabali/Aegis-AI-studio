# ============================================================================
# Aegis Backend - Optimized for HuggingFace Models
# ============================================================================

# Stage 1: Builder (dependencies only)
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================================
# Stage 2: Runtime (minimal)
# ============================================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create directories with proper permissions
RUN mkdir -p /app/cache /app/logs /app/models && \
    chmod 777 /app/cache /app/logs /app/models

# Copy application code
COPY backend/app /app/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # HuggingFace cache
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    # Optimize Python
    PYTHONOPTIMIZE=1 \
    # Disable HF telemetry
    HF_HUB_DISABLE_TELEMETRY=1

# Run application (optimized)
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop"]