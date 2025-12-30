# 🛡️ Aegis AI Studio

> Complete AI Platform: Groq (Llama 3.3 70B) + HuggingFace Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## ✨ Features

### 🚀 Core Capabilities
- **Ultra-Fast Chat**: 800+ tok/s via Groq Llama 3.3 70B
- **Code Generation**: DeepSeek Coder 1.3B (local)
- **Image Generation**: SDXL Turbo - 1-step diffusion
- **Image Analysis**: BLIP captioning & visual Q&A
- **Embeddings**: Semantic search with MiniLM-L6

### 🎯 Additional Features
- **Web Search**: Free DuckDuckGo integration
- **Voice Mode**: Browser STT + OpenAI TTS (optional)
- **Multi-Language**: 100+ languages supported
- **RAG Ready**: Built-in embeddings for retrieval
- **No GPU Required**: Runs on CPU (GPU optional for speed)

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    Aegis AI Studio                      │
├─────────────────────────────────────────────────────────┤
│  Frontend: Open WebUI (Port 3000)                      │
├─────────────────────────────────────────────────────────┤
│  Backend: FastAPI Gateway (Port 8000)                  │
│  ├─ Groq API (Cloud)                                   │
│  │  └─ Llama 3.3 70B (800 tok/s)                      │
│  └─ HuggingFace Models (Local)                         │
│     ├─ DeepSeek Coder 1.3B (code)                     │
│     ├─ SDXL Turbo (images)                            │
│     ├─ BLIP (image analysis)                          │
│     └─ MiniLM-L6 (embeddings)                         │
├─────────────────────────────────────────────────────────┤
│  Cache: Redis (Port 6379)                              │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB RAM minimum (16GB recommended)
- 10GB disk space (for models)
- Groq API Key (free): https://console.groq.com/keys

### Installation
```bash
# 1. Clone repository
git clone https://github.com/yourusername/aegis-studio.git
cd aegis-studio

# 2. Setup environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Launch (first run downloads models ~6GB)
docker-compose up -d

# 4. Wait for models to download (5-10 min on first run)
docker-compose logs -f backend

# 5. Access when you see "✓ HuggingFace models available"
open http://localhost:3000
```

## 📊 Available Models

| Model | Type | Speed | Size | Use Case |
|-------|------|-------|------|----------|
| **Llama 3.3 70B** | Chat | 800 tok/s | API | General chat, fast responses |
| **Mixtral 8x7B** | Chat | 500 tok/s | API | Long context (32k) |
| **DeepSeek Coder** | Code | ~100 tok/s | 1.3GB | Code generation |
| **SDXL Turbo** | Image | ~1-2s | 6.9GB | Image generation |
| **BLIP** | Vision | ~500ms | 800MB | Image analysis |
| **MiniLM-L6** | Embeddings | 1000/s | 80MB | Semantic search |

## 🎯 Usage Examples

### 1. Fast Chat
```
Model: aegis-groq-turbo
Prompt: "Explain quantum computing"
→ 800 tokens/sec response
```

### 2. Code Generation
```
Model: aegis-code
Prompt: "Write a Python function to sort a list"
→ Specialized code model generates optimal code
```

### 3. Image Generation
```
Model: aegis-image-gen
Prompt: "A futuristic city at sunset"
→ Generates 512x512 image in ~2 seconds
```

### 4. Image Analysis
```
Model: aegis-image-analyze
[Upload image]
Prompt: "Describe this image"
→ Detailed caption and analysis
```

## ⚙️ Configuration

### Required API Keys
- **Groq API**: Required (free, 30 req/min)
  - Get at: https://console.groq.com/keys

### Optional API Keys
- **OpenAI API**: For text-to-speech
  - Get at: https://platform.openai.com/api-keys

### Resource Requirements

**Minimum:**
- 8GB RAM
- 10GB disk space
- 2 CPU cores

**Recommended:**
- 16GB RAM
- 20GB disk space
- 4 CPU cores
- GPU (optional, improves HF model speed)

### Environment Variables
```bash
# Core
GROQ_API_KEY=gsk_xxx          # Required
GROQ_MODEL=llama-3.3-70b-versatile

# Features (all enabled by default)
ENABLE_HF_MODELS=true         # HuggingFace models
ENABLE_CACHING=true           # Redis caching
ENABLE_WEB_SEARCH=true        # DuckDuckGo search

# Optional
OPENAI_API_KEY=sk-xxx         # For TTS only
```

## 📈 Performance

### Benchmarks

**Groq (Cloud):**
- Latency: ~200ms TTFT
- Speed: 800 tokens/sec
- Context: 8k tokens

**HuggingFace (Local - CPU):**
- Code Gen: 1-3s for typical function
- Image Gen: 1-2s per image (512x512)
- Image Analysis: ~500ms per image
- Embeddings: ~1000 docs/sec

**With GPU (Optional):**
- Code Gen: <1s
- Image Gen: <1s
- 3-5x faster overall

## 🔧 Advanced Usage

### Enable Web Search
Already enabled by default. Uses DuckDuckGo (no API key needed).

### Custom System Prompts
Edit in Open WebUI settings or via API.

### RAG (Retrieval Augmented Generation)
Use `aegis-embeddings` model to generate embeddings, store in vector DB, retrieve and augment prompts.

### Voice Mode
Enable in Open WebUI:
1. Settings → Audio
2. STT: Web Speech API (free)
3. TTS: OpenAI (requires API key)

## 🐛 Troubleshooting

### Models Not Loading
```bash
# Check logs
docker-compose logs backend

# Models download on first run (5-10 min)
# Look for: "✓ Code model loaded"
```

### Out of Memory
```bash
# Reduce Docker memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 6G  # Reduce from 8G
```

### Slow Performance
```bash
# Disable HF models if not needed
ENABLE_HF_MODELS=false

# Or use GPU for faster inference
# Install nvidia-docker and update docker-compose.yml
```

## 📦 Model Downloads

Models are downloaded automatically on first run:

1. **DeepSeek Coder** (~1.3GB)
2. **SDXL Turbo** (~6.9GB)
3. **BLIP** (~800MB)
4. **MiniLM-L6** (~80MB)

**Total:** ~9GB initial download

Models are cached in `./data/models/` and reused on restart.

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- Groq for ultra-fast inference
- HuggingFace for open models
- Open WebUI for the interface
- All open source contributors

---

**Built with ❤️ for the AI community**