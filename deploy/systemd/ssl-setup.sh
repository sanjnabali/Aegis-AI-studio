#!/bin/bash
# deploy/ssl-setup.sh - Setup SSL certificates with Let's Encrypt

set -e

DOMAIN="aegis.studio"
EMAIL="admin@aegis.studio"

echo "Setting up SSL certificates for $DOMAIN"

# Install Certbot
if ! command -v certbot &> /dev/null; then
    echo "Installing Certbot..."
    sudo apt-get update
    sudo apt-get install -y certbot python3-certbot-nginx
fi

# Stop Nginx if running
sudo systemctl stop nginx || true

# Obtain certificate
sudo certbot certonly --standalone \
    --non-interactive \
    --agree-tos \
    --email $EMAIL \
    -d $DOMAIN \
    -d www.$DOMAIN

# Setup auto-renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Start Nginx
sudo systemctl start nginx

echo "SSL certificates installed successfully!"
echo ""
echo "Certificate locations:"
echo "  Cert: /etc/letsencrypt/live/$DOMAIN/fullchain.pem"
echo "  Key:  /etc/letsencrypt/live/$DOMAIN/privkey.pem"
echo ""
echo "Auto-renewal is enabled via systemd timer"