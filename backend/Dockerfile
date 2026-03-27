FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app/backend

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create app directory
COPY . .

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
