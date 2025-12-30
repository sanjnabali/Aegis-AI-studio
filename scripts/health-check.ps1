Write-Host "🔍 Aegis Studio - Health Check" -ForegroundColor Blue
Write-Host ""

# Containers
Write-Host "📦 Containers:" -ForegroundColor Yellow
docker-compose ps

# Backend health
Write-Host "`n🔧 Backend Health:" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/health"
    Write-Host "  Status: $($health.status)" -ForegroundColor Green
    Write-Host "  Primary Model: $($health.models.primary)" -ForegroundColor Cyan
} catch {
    Write-Host "  ❌ Backend unreachable" -ForegroundColor Red
}

# Models
Write-Host "`n🤖 Available Models:" -ForegroundColor Yellow
try {
    $models = Invoke-RestMethod -Uri "http://localhost:8000/v1/models"
    Write-Host "  Found $($models.data.Count) models:" -ForegroundColor Green
    foreach ($model in $models.data) {
        Write-Host "    • $($model.id)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "  ❌ Failed to fetch models" -ForegroundColor Red
}

# Cache
Write-Host "`n💾 Cache Status:" -ForegroundColor Yellow
try {
    $metrics = Invoke-RestMethod -Uri "http://localhost:8000/v1/metrics"
    Write-Host "  Hit Rate: $($metrics.cache.hit_rate)" -ForegroundColor Green
    Write-Host "  Hits: $($metrics.cache.hits)" -ForegroundColor Cyan
    Write-Host "  Misses: $($metrics.cache.misses)" -ForegroundColor Cyan
} catch {
    Write-Host "  ⚠️  Cache metrics unavailable" -ForegroundColor Yellow
}

Write-Host ""