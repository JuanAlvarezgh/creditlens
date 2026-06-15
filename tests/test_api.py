from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

VALID_APPLICATION = {
    "revolving_utilization": 0.45,
    "age": 35,
    "times_30_59_days_late": 0,
    "debt_ratio": 0.25,
    "monthly_income": 5000.0,
    "open_credit_lines": 4,
    "times_90_days_late": 0,
    "real_estate_loans": 1,
    "times_60_89_days_late": 0,
    "dependents": 2,
}

MOCK_SCORE_RESULT = {
    "probability_of_default": 0.15,
    "risk_level": "Bajo",
    "shap_explanation": [
        {"feature": "revolving_utilization", "impact": 0.08},
        {"feature": "debt_ratio", "impact": 0.04},
        {"feature": "age", "impact": -0.02},
    ],
}

MOCK_VERSION_INFO = {
    "model_name": "credit_risk_model",
    "version": "1",
    "stage": "Production",
    "auc_roc": 0.87,
    "registered_at": "2026-01-01",
}


@pytest.fixture
def client():
    with patch("api.model_loader.model_loader.load"):
        with patch("api.main.model_loader") as mock_loader:
            mock_loader.version_info = MOCK_VERSION_INFO
            from api.main import app

            with TestClient(app) as c:
                yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_score_valid_input(client):
    with patch("api.main.score", return_value=MOCK_SCORE_RESULT):
        resp = client.post("/api/v1/score", json=VALID_APPLICATION)
    assert resp.status_code == 200
    body = resp.json()
    assert "probability_of_default" in body
    assert body["risk_level"] in ("Bajo", "Medio", "Alto")
    assert len(body["shap_explanation"]) == 3


def test_score_invalid_age_below_minimum(client):
    bad = {**VALID_APPLICATION, "age": 10}
    resp = client.post("/api/v1/score", json=bad)
    assert resp.status_code == 422


def test_score_missing_required_field(client):
    bad = {k: v for k, v in VALID_APPLICATION.items() if k != "monthly_income"}
    resp = client.post("/api/v1/score", json=bad)
    assert resp.status_code == 422


def test_score_negative_utilization(client):
    bad = {**VALID_APPLICATION, "revolving_utilization": -0.1}
    resp = client.post("/api/v1/score", json=bad)
    assert resp.status_code == 422


def test_score_utilization_above_max(client):
    bad = {**VALID_APPLICATION, "revolving_utilization": 1.5}
    resp = client.post("/api/v1/score", json=bad)
    assert resp.status_code == 422


def test_model_info_returns_production_stage(client):
    resp = client.get("/api/v1/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "Production"
    assert body["auc_roc"] == 0.87
