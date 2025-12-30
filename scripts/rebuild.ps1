# Complete rebuild script
Write-Host "🔨 Rebuilding Aegis Studio..." -ForegroundColor Blue

# Stop everything
Write-Host "`n[1/5] Stopping containers..." -ForegroundColor Yellow
docker-compose down -v

# Clean Docker cache
Write-Host "`n[2/5] Cleaning Docker cache..." -ForegroundColor Yellow
docker system prune -f

# Rebuild
Write-Host "`n[3/5] Building images..." -ForegroundColor Yellow
docker-compose build --no-cache

# Start
Write-Host "`n[4/5] Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for startup
Write-Host "`n[5/5] Waiting for services..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Health check
Write-Host "`n✅ Running health check..." -ForegroundColor Green
.\scripts\health-check.ps1

Write-Host "`n🎉 Rebuild complete!" -ForegroundColor Green
Write-Host "Visit: http://localhost:3000" -ForegroundColor Cyan