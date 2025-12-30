#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🛡️  AEGIS AI STUDIO - COMPLETE SETUP                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check requirements
echo -e "${YELLOW}[1/8] Checking requirements...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed${NC}"

available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 12 ]; then
    echo -e "${RED}Need 12GB disk space, have ${available_space}GB${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Disk space: ${available_space}GB${NC}"

# Setup .env
echo -e "${YELLOW}[2/8] Setting up environment...${NC}"
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "Get Groq API key: https://console.groq.com/keys"
    read -p "Enter GROQ_API_KEY: " GROQ_KEY
    sed -i "s/GROQ_API_KEY=.*/GROQ_API_KEY=$GROQ_KEY/" .env
    echo -e "${GREEN}✓ Configuration saved${NC}"
fi

# Create directories
echo -e "${YELLOW}[3/8] Creating directories...${NC}"
mkdir -p data/{cache,logs,open-webui,redis,models}
echo -e "${GREEN}✓ Directories created${NC}"

# Build
echo -e "${YELLOW}[4/8] Building containers...${NC}"
docker-compose build --no-cache

# Start
echo -e "${YELLOW}[5/8] Starting services...${NC}"
docker-compose up -d

# Wait for models
echo -e "${YELLOW}[6/8] Downloading models (~10GB, 5-10 min)...${NC}"
for i in {1..120}; do
    if docker-compose logs backend 2>/dev/null | grep -q "HuggingFace models available"; then
        echo -e "${GREEN}✓ Models loaded${NC}"
        break
    fi
    sleep 5
    echo -n "."
done

# Health check
echo -e "${YELLOW}[7/8] Running health checks...${NC}"
for i in {1..60}; do
    if curl -sf http://localhost:8000/health &> /dev/null; then
        echo -e "${GREEN}✓ Backend healthy${NC}"
        break
    fi
    sleep 1
done

MODEL_COUNT=$(curl -s http://localhost:8000/v1/models | grep -o '"id"' | wc -l)
echo -e "${GREEN}✓ Found $MODEL_COUNT models${NC}"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     SETUP COMPLETED SUCCESSFULLY!             ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Access:${NC}"
echo "  Open WebUI:  http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs:    http://localhost:8000/docs"
echo ""
echo -e "${BLUE}Models:${NC}"
echo "  • aegis-groq-turbo (Llama 3.3 70B) - 800 tok/s"
echo "  • aegis-code (DeepSeek 1.3B) - Code generation"
echo "  • aegis-image-gen (SDXL) - Image generation"
echo "  • aegis-image-analyze (BLIP) - Image analysis"
echo ""