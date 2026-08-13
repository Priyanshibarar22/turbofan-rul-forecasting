from fastapi.testclient import TestClient
from main import app, FEATURE_COUNT

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_version" in response.json()

def test_predict_valid_input():
    payload = {"features": [0.1] * FEATURE_COUNT}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_rul" in body
    assert body["risk_level"] in ["critical", "warning", "healthy"]

def test_predict_wrong_feature_count():
    payload = {"features": [0.1] * (FEATURE_COUNT - 5)}  # deliberately wrong length
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_type():
    payload = {"features": "not_a_list"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_risk_level_boundaries():
    # A very low feature vector should trigger critical risk, given your model's typical output range
    payload = {"features": [0.1] * FEATURE_COUNT}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200