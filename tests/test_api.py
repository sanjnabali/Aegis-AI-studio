from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health endpoint returns correct status"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    # Note: The health endpoint returns 'mode' in the openai_adapter
    # but the root health endpoint doesn't, so we just check status

def test_features_endpoint():
    """Test the features endpoint is accessible"""
    response = client.get("/v1/features")
    assert response.status_code == 200
    data = response.json()
    assert "available" in data
    assert "chat" in data["available"]