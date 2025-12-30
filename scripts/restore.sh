#!/bin/bash
# scripts/restore.sh - Restore Aegis data from backup

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Usage: ./scripts/restore.sh <backup_file.tar.gz>${NC}"
    echo ""
    echo "Available backups:"
    ls -lh backups/*.tar.gz 2>/dev/null || echo "  No backups found"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}Aegis Studio - Data Restoration${NC}"
echo ""
echo -e "${YELLOW}WARNING: This will overwrite existing data!${NC}"
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Restoration cancelled."
    exit 0
fi

# Stop services
echo ""
echo -e "${YELLOW}[1/4] Stopping services...${NC}"
docker-compose down
echo -e "${GREEN}✓ Services stopped${NC}"

# Extract backup
echo -e "${YELLOW}[2/4] Extracting backup...${NC}"
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_NAME=$(ls "$TEMP_DIR")
echo -e "${GREEN}✓ Backup extracted${NC}"

# Restore data
echo -e "${YELLOW}[3/4] Restoring data...${NC}"

# Backup current data (just in case)
if [ -d "data" ]; then
    mv data "data.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${BLUE}ℹ  Current data backed up${NC}"
fi

# Restore files
cp "${TEMP_DIR}/${BACKUP_NAME}/.env" .env 2>/dev/null || true
cp -r "${TEMP_DIR}/${BACKUP_NAME}/open-webui" data/ 2>/dev/null || true
cp -r "${TEMP_DIR}/${BACKUP_NAME}/redis" data/ 2>/dev/null || true

# Cleanup
rm -rf "$TEMP_DIR"

echo -e "${GREEN}Data restored${NC}"

# Restart services
echo -e "${YELLOW}[4/4] Restarting services...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}Restoration completed!${NC}"
echo ""
echo "Please wait 30 seconds for services to start..."
echo ""