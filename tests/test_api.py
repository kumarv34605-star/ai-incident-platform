from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["model"] == "incident-classifier"


def test_predict_incident():
    telemetry = {
        "cpu_usage": 95,
        "memory_usage": 75,
        "request_rate": 900,
        "latency_ms": 700,
        "error_rate": 4,
        "db_connections": 90,
        "http_5xx": 35,
    }

    response = client.post("/predict", json=telemetry)

    assert response.status_code == 200
    assert response.json()["incident"] == 1
    assert response.json()["status"] == "incident_detected"


def test_invalid_telemetry():
    telemetry = {
        "cpu_usage": 150,
        "memory_usage": 75,
        "request_rate": 900,
        "latency_ms": 700,
        "error_rate": 4,
        "db_connections": 90,
        "http_5xx": 35,
    }

    response = client.post("/predict", json=telemetry)

    assert response.status_code == 422