# 🛡️ Aegis AI Studio - HuggingFace First

> Complete Local AI Platform with Advanced Models

## ✨ Models Available

### Primary Models (HuggingFace - Local)

| Model | Type | Size | Speed | Best For |
|-------|------|------|-------|----------|
| **GPT-OSS-20B** | Reasoning | 20GB | ~100 tok/s | Complex analysis, logic problems |
| **Llama 4 Scout 17B** | Vision | 17GB | ~2-3s | Image understanding, visual Q&A |
| **DeepSeek Coder 1.3B** | Code | 1.3GB | ~100 tok/s | Code generation, debugging |
| **SDXL Turbo** | Image Gen | 6.9GB | ~1-2s | Fast image creation |
| **MiniLM-L6-v2** | Embeddings | 80MB | 1000/s | Semantic search, RAG |
| **Whisper Tiny** | Speech | 150MB | ~150ms | Voice transcription |

### Secondary Models (Groq - Cloud)

| Model | Speed | Best For |
|-------|-------|----------|
| **Llama 3.3 70B** | 800 tok/s | Ultra-fast chat |

**Total Download: ~45GB** (first run only)

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
- Docker & Docker Compose
- 16GB RAM (32GB recommended)
- 50GB disk space
- Groq API Key (free): https://console.groq.com/keys
```

### 2. Install

```bash
# Clone
git clone <repo>
cd aegis-studio

# Configure
cp .env.example .env
# Edit .env, add GROQ_API_KEY

# Deploy (will download ~45GB models on first run)
docker-compose up -d

# Monitor model downloads (takes 10-20 min)
docker-compose logs -f backend
```

### 3. Access

- **Open WebUI**: http://localhost:3000
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs

---

## 🎯 Usage Guide

### For Reasoning Tasks

```
Model: aegis-reasoning (GPT-OSS-20B)
Prompt: "Explain the P vs NP problem"
→ Deep analysis with logical reasoning
```

### For Image Analysis

```
Model: aegis-vision (Llama 4 Scout 17B)
[Upload image]
Prompt: "Describe this scene in detail"
→ Advanced visual understanding
```

### For Code Generation

```
Model: aegis-code (DeepSeek Coder)
Prompt: "Write a Python function to parse JSON"
→ Optimized code with explanations
```

### For Image Generation

```
Model: aegis-image-gen (SDXL Turbo)
Prompt: "A cyberpunk city at night"
→ 512x512 image in ~1-2 seconds
```

### For Fast Chat

```
Model: aegis-groq-turbo (Llama 3.3 70B)
Prompt: "What's quantum computing?"
→ 800 tokens/second response
```

### Auto Routing (Smart)

```
Model: aegis-auto
Prompt: [any task]
→ Automatically selects best model
```

---

## ⚙️ Configuration

### `.env` File

```bash
# Required
GROQ_API_KEY=gsk_your_key_here

# Features (all enabled by default)
ENABLE_HF_MODELS=true
ENABLE_CACHING=true
ENABLE_WEB_SEARCH=true

# Model Selection
GROQ_MODEL=llama-3.3-70b-versatile
```

### Resource Requirements

**Minimum:**
- RAM: 16GB
- Disk: 50GB free
- CPU: 4 cores

**Recommended:**
- RAM: 32GB
- Disk: 100GB SSD
- GPU: NVIDIA with 24GB VRAM (optional, 3-5x faster)

---

## 🔧 Advanced Usage

### Enable GPU Acceleration (Optional)

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### API Usage

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Reasoning
response = client.chat.completions.create(
    model="aegis-reasoning",
    messages=[{"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}]
)

# Vision
response = client.chat.completions.create(
    model="aegis-vision",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://..."}}
        ]
    }]
)

# Embeddings
embeddings = client.embeddings.create(
    model="aegis-embeddings",
    input=["text to embed"]
)
```

---

## 📊 Performance Benchmarks

**With HF Models (CPU):**
- Reasoning: ~100 tok/s
- Vision: 2-3s per image
- Code: ~100 tok/s
- Image Gen: 1-2s per image

**With GPU (NVIDIA):**
- Reasoning: ~300 tok/s
- Vision: <1s per image
- Code: ~300 tok/s
- Image Gen: <1s per image

---

## 🔍 Troubleshooting

### Models Not Loading

```bash
# Check logs
docker-compose logs -f backend

# Verify disk space (need 50GB)
df -h

# Restart if needed
docker-compose restart backend
```

### Out of Memory

```bash
# Reduce resource limits in docker-compose.yml
# Or disable some models:
# Edit .env: ENABLE_HF_MODELS=false
```

### Slow Performance

```bash
# Add GPU support (see Advanced Usage)
# Or use Groq for fast chat:
# Model: aegis-groq-turbo
```

---

## 📦 Model Details

### GPT-OSS-20B (Reasoning)
- **Source**: OpenAI
- **Parameters**: 20 billion
- **Specialization**: Logical reasoning, analysis
- **Context**: 4k tokens

### Llama 4 Scout 17B (Vision)
- **Source**: Meta
- **Parameters**: 17 billion
- **Specialization**: Image understanding
- **Context**: 8k tokens

### DeepSeek Coder 1.3B
- **Source**: DeepSeek AI
- **Parameters**: 1.3 billion
- **Specialization**: Code generation
- **Context**: 4k tokens

### SDXL Turbo
- **Source**: Stability AI
- **Type**: Diffusion model
- **Specialization**: Fast image generation
- **Steps**: 1 (turbo mode)

---

## 🛠️ Management Commands

```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Update & rebuild
git pull
docker-compose up -d --build

# Clean cache
docker system prune -f
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🤝 Support

- Issues: GitHub Issues
- Docs: /docs endpoint
- API: OpenAI-compatible

**Total Setup Time**: 10-20 minutes (first run)
**Disk Space**: ~45GB (models) + 5GB (system)
**RAM Usage**: 8-16GB active