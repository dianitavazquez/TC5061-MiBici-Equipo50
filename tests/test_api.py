
"""Pruebas de la API REST para el modelo MiBici P2."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from mibici_ml.api.main import app, get_model

client = TestClient(app)

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "models/serving/p2/input_example.json"
)

with EXAMPLE_PATH.open(encoding="utf-8") as file:
    example = json.load(file)

VALID_INPUT = dict(zip(example["columns"], example["data"][0]))


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    """Simula el modelo para evitar dependencias de MLflow."""
    model = Mock()
    model.predict.return_value = [[0, 0]]

    monkeypatch.setattr(
        "mibici_ml.api.main.get_model",
        lambda: model,
    )

    yield model
    get_model.cache_clear()


def test_health_returns_200():
    """Verifica que el endpoint de salud responde correctamente."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "P2",
    }


def test_predict_valid_input(mock_model):
    """Verifica una predicción válida."""
    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json() == {
        "risk_no_bikes": 0,
        "risk_no_docks": 0,
    }

    mock_model.predict.assert_called_once()


def test_predict_missing_field():
    """Rechaza peticiones con variables obligatorias faltantes."""
    payload = VALID_INPUT.copy()
    del payload["capacity"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_negative_capacity():
    """Rechaza capacidades negativas."""
    payload = {**VALID_INPUT, "capacity": -5}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_weekend():
    """Rechaza valores de fin de semana fuera del rango permitido."""
    payload = {**VALID_INPUT, "is_weekend": 2}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_extra_field():
    """Rechaza variables adicionales."""
    payload = {**VALID_INPUT, "unknown_feature": 123}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_model_unavailable(monkeypatch):
    """Devuelve 503 cuando no es posible cargar el modelo."""

    def unavailable_model():
        raise RuntimeError("Modelo no disponible")

    monkeypatch.setattr(
        "mibici_ml.api.main.get_model",
        unavailable_model,
    )

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 503


def test_predict_inference_error(mock_model):
    """Devuelve 500 cuando falla la inferencia."""
    mock_model.predict.side_effect = RuntimeError(
        "Error de inferencia"
    )

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 500
