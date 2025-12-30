#!/bin/bash
# deploy/production-setup.sh - Complete production server setup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🛡️  AEGIS STUDIO - PRODUCTION SETUP                  ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

# =============================================================================
# SYSTEM UPDATES
# =============================================================================

echo -e "${YELLOW}[1/10] Updating system...${NC}"
apt-get update
apt-get upgrade -y
apt-get install -y curl wget git vim ufw fail2ban

echo -e "${GREEN}✓ System updated${NC}"

# =============================================================================
# DOCKER INSTALLATION
# =============================================================================

echo -e "${YELLOW}[2/10] Installing Docker...${NC}"

if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    
    # Install Docker Compose
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${GREEN}✓ Docker already installed${NC}"
fi

# =============================================================================
# USER SETUP
# =============================================================================

echo -e "${YELLOW}[3/10] Creating aegis user...${NC}"

if ! id -u aegis &>/dev/null; then
    useradd -m -s /bin/bash aegis
    usermod -aG docker aegis
    echo -e "${GREEN}✓ User created${NC}"
else
    echo -e "${GREEN}✓ User already exists${NC}"
fi

# =============================================================================
# FIREWALL CONFIGURATION
# =============================================================================

echo -e "${YELLOW}[4/10] Configuring firewall...${NC}"

ufw --force enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw reload

echo -e "${GREEN}✓ Firewall configured${NC}"

# =============================================================================
# FAIL2BAN SETUP
# =============================================================================

echo -e "${YELLOW}[5/10] Setting up Fail2Ban...${NC}"

cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-limit-req]
enabled = true
EOF

systemctl enable fail2ban
systemctl restart fail2ban

echo -e "${GREEN}✓ Fail2Ban configured${NC}"

# =============================================================================
# NGINX INSTALLATION
# =============================================================================

echo -e "${YELLOW}[6/10] Installing Nginx...${NC}"

apt-get install -y nginx
systemctl enable nginx

echo -e "${GREEN}✓ Nginx installed${NC}"

# =============================================================================
# PROJECT DEPLOYMENT
# =============================================================================

echo -e "${YELLOW}[7/10] Deploying project...${NC}"

cd /opt
if [ ! -d "aegis-studio" ]; then
    git clone https://github.com/yourusername/aegis-studio.git
    chown -R aegis:aegis aegis-studio
fi

cd aegis-studio

# Create .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please edit /opt/aegis-studio/.env with your API keys${NC}"
    read -p "Press ENTER after editing .env..."
fi

echo -e "${GREEN}✓ Project deployed${NC}"

# =============================================================================
# SSL CERTIFICATES
# =============================================================================

echo -e "${YELLOW}[8/10] Setting up SSL...${NC}"

read -p "Enter your domain name (e.g., aegis.studio): " DOMAIN
read -p "Enter your email for Let's Encrypt: " EMAIL

apt-get install -y certbot python3-certbot-nginx

systemctl stop nginx

certbot certonly --standalone \
    --non-interactive \
    --agree-tos \
    --email $EMAIL \
    -d $DOMAIN \
    -d www.$DOMAIN

echo -e "${GREEN}✓ SSL certificates obtained${NC}"

# =============================================================================
# NGINX CONFIGURATION
# =============================================================================

echo -e "${YELLOW}[9/10] Configuring Nginx...${NC}"

# Copy Nginx config
cp deploy/nginx.conf /etc/nginx/sites-available/aegis
sed -i "s/aegis.studio/$DOMAIN/g" /etc/nginx/sites-available/aegis
ln -sf /etc/nginx/sites-available/aegis /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test configuration
nginx -t

systemctl start nginx

echo -e "${GREEN}✓ Nginx configured${NC}"

# =============================================================================
# START SERVICES
# =============================================================================

echo -e "${YELLOW}[10/10] Starting Aegis services...${NC}"

# Build and start containers
su - aegis -c "cd /opt/aegis-studio && docker-compose up -d --build"

# Setup systemd service
cp deploy/systemd/aegis.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable aegis
systemctl start aegis

echo -e "${GREEN}✓ Services started${NC}"

# =============================================================================
# HEALTH CHECK
# =============================================================================

echo ""
echo -e "${YELLOW}Performing health check...${NC}"
sleep 30

if curl -f http://localhost:8000/health &> /dev/null; then
    echo -e "${GREEN}Backend is healthy${NC}"
else
    echo -e "${RED}Backend health check failed${NC}"
fi

if curl -f http://localhost:3000 &> /dev/null; then
    echo -e "${GREEN}Frontend is healthy${NC}"
else
    echo -e "${RED}Frontend health check failed${NC}"
fi

# =============================================================================
# SUCCESS
# =============================================================================

echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║             PRODUCTION SETUP COMPLETED!                   ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo ""
echo -e "${BLUE}🌐 Your Aegis Studio is live at:${NC}"
echo -e "   https://$DOMAIN"
echo ""
echo -e "${BLUE}Monitoring:${NC}"
echo "   Logs: journalctl -u aegis -f"
echo "   Docker: docker-compose logs -f"
echo "   Nginx: tail -f /var/log/nginx/aegis_access.log"
echo ""
echo -e "${BLUE}🔧 Management:${NC}"
echo "   Restart: systemctl restart aegis"
echo "   Stop: systemctl stop aegis"
echo "   Status: systemctl status aegis"
echo ""
echo -e "${YELLOW}⚠️  Important next steps:${NC}"
echo "   1. Update firewall rules if needed"
echo "   2. Configure monitoring (Prometheus/Grafana)"
echo "   3. Setup automated backups"
echo "   4. Review logs for any errors"
echo ""