"""
Tests for Smart Governance AI API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Smart Governance AI" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_topic():
    response = client.post("/predict-topic", json={"text": "Water pipeline burst on main road"})
    assert response.status_code == 200
    assert "topic_label" in response.json()


def test_predict_urgency():
    response = client.post("/predict-urgency", json={"text": "Emergency! Building collapsed"})
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_analyze_text():
    response = client.post("/analyze-text", json={"text": "Garbage not collected for 2 weeks"})
    assert response.status_code == 200
    data = response.json()
    assert "topic" in data
    assert "urgency" in data


def test_similar_news():
    response = client.post("/similar-news", json={"text": "Government announces new policy"})
    assert response.status_code == 200


def test_multilingual_hindi():
    response = client.post("/predict-urgency", json={"text": "सड़क पर बड़ा गड्ढा है, बहुत खतरनाक है"})
    assert response.status_code == 200


def test_multilingual_telugu():
    response = client.post("/predict-topic", json={"text": "రోడ్డు మీద పెద్ద గుంత ఉంది"})
    assert response.status_code == 200


def test_empty_text_rejected():
    response = client.post("/predict-topic", json={"text": "ab"})
    assert response.status_code == 422  # Validation error (min_length=5)
