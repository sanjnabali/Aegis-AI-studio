# Health check script
Write-Host "🔍 Checking Aegis Studio Health..." -ForegroundColor Blue

# Check containers
Write-Host "`n📦 Container Status:" -ForegroundColor Yellow
docker-compose ps

# Check backend health
Write-Host "`n🔧 Backend Health:" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
    Write-Host "Status: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "Backend unreachable!" -ForegroundColor Red
}

# Check models
Write-Host "`n🤖 Available Models:" -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "http://localhost:8000/v1/models"
    Write-Host "Found $($models.data.Count) models" -ForegroundColor Green
} catch {
    Write-Host "Failed to fetch models" -ForegroundColor Red
}

# Check metrics
Write-Host "`n📊 Performance Metrics:" -ForegroundColor Yellow
try {
    $metrics = Invoke-RestMethod -Uri "http://localhost:8000/v1/metrics"
    Write-Host "Groq: $($metrics.groq.requests) requests" -ForegroundColor Green
} catch {
    Write-Host "Failed to fetch metrics" -ForegroundColor Red
}