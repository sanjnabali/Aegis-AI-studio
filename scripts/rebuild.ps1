Write-Host "🔨 Aegis Studio - Complete Rebuild" -ForegroundColor Blue
Write-Host ""

Write-Host "[1/5] Stopping containers..." -ForegroundColor Yellow
docker-compose down -v

Write-Host "[2/5] Cleaning Docker cache..." -ForegroundColor Yellow
docker system prune -f

Write-Host "[3/5] Building images..." -ForegroundColor Yellow
docker-compose build --no-cache

Write-Host "[4/5] Starting services..." -ForegroundColor Yellow
docker-compose up -d

Write-Host "[5/5] Waiting for services..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host ""
Write-Host "✅ Rebuild complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Running health check..." -ForegroundColor Cyan
.\scripts\health-check.ps1

Write-Host ""
Write-Host "🎉 Ready at: http://localhost:3000" -ForegroundColor Green