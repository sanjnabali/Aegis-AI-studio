#!/bin/bash
# scripts/setup.sh - Complete setup script for Aegis Studio

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🛡️  AEGIS MULTIMODAL AI STUDIO - SETUP              ║
║                                                           ║
║     Free-Tier | Zero-GPU | Production-Ready              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo -e "${YELLOW}[1/8] Pre-flight checks...${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker first.${NC}"
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✓ Docker installed: $(docker --version)${NC}"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Docker Compose not found. Please install Docker Compose first.${NC}"
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose installed${NC}"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker daemon is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker daemon running${NC}"

# Check available disk space (need at least 2GB)
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 2 ]; then
    echo -e "${RED}Insufficient disk space. Need at least 2GB, have ${available_space}GB${NC}"
    exit 1
fi
echo -e "${GREEN}Sufficient disk space: ${available_space}GB available${NC}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo ""
echo -e "${YELLOW}[2/8] Setting up environment...${NC}"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file from template...${NC}"
    cp .env.example .env
    
    echo ""
    echo -e "${YELLOW}IMPORTANT: You need to add your API keys!${NC}"
    echo ""
    echo "Please visit these URLs to get your FREE API keys:"
    echo ""
    echo "1. Groq API (Ultra-fast LLM):"
    echo "   → https://console.groq.com/keys"
    echo ""
    echo "2. Google Gemini API (Advanced reasoning):"
    echo "   → https://ai.google.dev/gemini-api/docs/api-key"
    echo ""
    echo "3. (Optional) OpenAI API for TTS:"
    echo "   → https://platform.openai.com/api-keys"
    echo ""
    
    read -p "Press ENTER after you've obtained your API keys..."
    
    # Interactive API key input
    read -p "Enter your GROQ API KEY (starts with gsk_): " GROQ_KEY
    read -p "Enter your GEMINI API KEY: " GEMINI_KEY
    read -p "Enter your OPENAI API KEY (optional, press ENTER to skip): " OPENAI_KEY
    
    # Update .env file
    sed -i.bak "s/GROQ_API_KEY=.*/GROQ_API_KEY=$GROQ_KEY/" .env
    sed -i.bak "s/GEMINI_API_KEY=.*/GEMINI_API_KEY=$GEMINI_KEY/" .env
    
    if [ ! -z "$OPENAI_KEY" ]; then
        sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
    fi
    
    rm .env.bak
    
    echo -e "${GREEN}API keys configured${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
    
    # Validate API keys
    source .env
    
    if [ -z "$GROQ_API_KEY" ] || [ "$GROQ_API_KEY" == "gsk_your_groq_api_key_here" ]; then
        echo -e "${RED}GROQ_API_KEY not set in .env file${NC}"
        echo "   Please edit .env and add your Groq API key"
        exit 1
    fi
    
    if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" == "your_gemini_api_key_here" ]; then
        echo -e "${RED}GEMINI_API_KEY not set in .env file${NC}"
        echo "   Please edit .env and add your Gemini API key"
        exit 1
    fi
    
    echo -e "${GREEN}API keys validated${NC}"
fi

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

echo ""
echo -e "${YELLOW}[3/8] Creating data directories...${NC}"

mkdir -p data/{cache,logs,open-webui,redis,prometheus,grafana}
mkdir -p logs
mkdir -p monitoring

echo -e "${GREEN}Directories created${NC}"

# ============================================================================
# DOCKER NETWORK
# ============================================================================

echo ""
echo -e "${YELLOW}[4/8] Setting up Docker network...${NC}"

# Check if network exists
if docker network inspect aegis-network &> /dev/null; then
    echo -e "${BLUE}ℹ  Network 'aegis-network' already exists${NC}"
else
    docker network create aegis-network
    echo -e "${GREEN}Network created${NC}"
fi

# ============================================================================
# BUILD CONTAINERS
# ============================================================================

echo ""
echo -e "${YELLOW}[5/8] Building Docker containers...${NC}"
echo -e "${BLUE}ℹ  This may take 3-5 minutes on first run...${NC}"

docker-compose build --no-cache

echo -e "${GREEN}Containers built successfully${NC}"

# ============================================================================
# START SERVICES
# ============================================================================

echo ""
echo -e "${YELLOW}[6/8] Starting services...${NC}"

docker-compose up -d

echo -e "${GREEN}Services started${NC}"

# ============================================================================
# WAIT FOR SERVICES
# ============================================================================

echo ""
echo -e "${YELLOW}[7/8] Waiting for services to be ready...${NC}"

# Wait for Redis
echo -n "   Waiting for Redis..."
for i in {1..30}; do
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

# Wait for Backend
echo -n "   Waiting for Backend..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

# Wait for Frontend
echo -n "   Waiting for Frontend..."
for i in {1..30}; do
    if curl -sf http://localhost:3000 &> /dev/null; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    sleep 1
    echo -n "."
done

# ============================================================================
# HEALTH CHECK
# ============================================================================

echo ""
echo -e "${YELLOW}[8/8] Running health checks...${NC}"

# Check backend health
HEALTH_STATUS=$(curl -s http://localhost:8000/health | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$HEALTH_STATUS" == "healthy" ]; then
    echo -e "${GREEN}Backend is healthy${NC}"
else
    echo -e "${RED}Backend health check failed${NC}"
    echo "   Checking logs..."
    docker-compose logs --tail=20 backend
    exit 1
fi

# Check available models
MODEL_COUNT=$(curl -s http://localhost:8000/v1/models | grep -o '"id"' | wc -l)

if [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}Found $MODEL_COUNT AI models${NC}"
else
    echo -e "${RED}No models found${NC}"
    exit 1
fi

# ============================================================================
# SUCCESS!
# ============================================================================

echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                  SETUP COMPLETED SUCCESSFULLY!            ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${BLUE}Access Points:${NC}"
echo ""
echo -e "   ${GREEN}Open WebUI:${NC}      http://localhost:3000"
echo -e "   ${GREEN}Backend API:${NC}     http://localhost:8000"
echo -e "   ${GREEN}API Docs:${NC}        http://localhost:8000/docs"
echo -e "   ${GREEN}Redis:${NC}           localhost:6379"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Create your admin account"
echo "   3. Start chatting with Aegis!"
echo ""
echo "   ${YELLOW}Optional:${NC}"
echo "   • Copy system prompt from docs/SYSTEM_PROMPT.md"
echo "   • Import web search function from frontend/functions/"
echo "   • Enable monitoring: docker-compose --profile monitoring up -d"
echo ""

echo -e "${BLUE}Useful Commands:${NC}"
echo ""
echo "   View logs:        docker-compose logs -f"
echo "   Stop services:    docker-compose down"
echo "   Restart:          docker-compose restart"
echo "   Update:           docker-compose pull && docker-compose up -d"
echo "   Backup data:      ./scripts/backup.sh"
echo ""

echo -e "${GREEN}🎉 Happy coding with Aegis Studio!${NC}"
echo ""