from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["mode"] == "ultra-lightweight"

def test_features_endpoint():
    response = client.get("/v1/features")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "chat" in data["available"]
