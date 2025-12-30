#!/bin/bash
# scripts/update.sh - Update Aegis Studio to latest version

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🛡️  Aegis Studio - Update${NC}"
echo ""

# Backup before update
echo -e "${YELLOW}[1/5] Creating backup...${NC}"
./scripts/backup.sh
echo ""

# Pull latest changes
echo -e "${YELLOW}[2/5] Pulling latest changes...${NC}"
git pull origin main
echo -e "${GREEN}Code updated${NC}"

# Pull latest images
echo -e "${YELLOW}[3/5] Pulling Docker images...${NC}"
docker-compose pull
echo -e "${GREEN}Images updated${NC}"

# Rebuild
echo -e "${YELLOW}[4/5] Rebuilding containers...${NC}"
docker-compose build
echo -e "${GREEN}Containers rebuilt${NC}"

# Restart
echo -e "${YELLOW}[5/5] Restarting services...${NC}"
docker-compose down
docker-compose up -d
echo -e "${GREEN}Services restarted${NC}"

echo ""
echo -e "${GREEN}Update completed!${NC}"
echo ""