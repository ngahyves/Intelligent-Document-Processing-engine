from fastapi.testclient import TestClient
import unittest.mock as mock
import pytest

#Simulate the call of the model
with mock.patch("onnxruntime.InferenceSession"):
    with mock.patch("src.vision.classifier.DocumentClassifier"):
        from api.main import app

client = TestClient(app)

def test_health():
    """Vérify if API is healthy(running)"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_ask_no_question():
    """Vérify if the API rejects empty requests"""
    response = client.post("/ask", json={})
    assert response.status_code == 422