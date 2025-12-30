#!/bin/bash
# scripts/backup.sh - Backup all Aegis data

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🛡️  Aegis Studio - Data Backup${NC}"
echo ""

# Configuration
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="aegis_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo -e "${YELLOW}[1/5] Preparing backup...${NC}"
mkdir -p "$BACKUP_PATH"

# Backup .env file (without exposing keys in filename)
echo -e "${YELLOW}[2/5] Backing up configuration...${NC}"
cp .env "${BACKUP_PATH}/.env"
echo -e "${GREEN}✓ Configuration backed up${NC}"

# Backup Open WebUI data
echo -e "${YELLOW}[3/5] Backing up Open WebUI data...${NC}"
if [ -d "data/open-webui" ]; then
    cp -r data/open-webui "${BACKUP_PATH}/"
    echo -e "${GREEN}Open WebUI data backed up${NC}"
else
    echo -e "${YELLOW}No Open WebUI data found${NC}"
fi

# Backup Redis data
echo -e "${YELLOW}[4/5] Backing up cache data...${NC}"
if [ -d "data/redis" ]; then
    # Stop Redis gracefully to ensure data consistency
    docker-compose exec redis redis-cli BGSAVE &> /dev/null || true
    sleep 2
    
    cp -r data/redis "${BACKUP_PATH}/"
    echo -e "${GREEN}Cache data backed up${NC}"
else
    echo -e "${YELLOW}No Redis data found${NC}"
fi

# Create compressed archive
echo -e "${YELLOW}[5/5] Compressing backup...${NC}"
tar -czf "${BACKUP_PATH}.tar.gz" -C "$BACKUP_DIR" "$BACKUP_NAME"
rm -rf "$BACKUP_PATH"

# Get backup size
BACKUP_SIZE=$(du -h "${BACKUP_PATH}.tar.gz" | cut -f1)

echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
echo ""
echo -e "   ${BLUE}Backup file:${NC} ${BACKUP_PATH}.tar.gz"
echo -e "   ${BLUE}Size:${NC}        $BACKUP_SIZE"
echo ""
echo -e "${BLUE}To restore this backup:${NC}"
echo "   ./scripts/restore.sh ${BACKUP_NAME}.tar.gz"
echo ""