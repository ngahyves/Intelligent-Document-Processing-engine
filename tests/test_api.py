from fastapi.testclient import TestClient
from api.main import app # importing the app

# Create a false client
client = TestClient(app)

def test_health():
    """verifying the health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_no_question():
    """API send an error if no question is asked"""
    # Empty request
    response = client.post("/ask", json={})
    # 422 means incorrect data
    assert response.status_code == 422