"""Escenario Locust para el endpoint de predicción de MiBici."""

import json
from pathlib import Path
import random

from locust import HttpUser, between, task

PAYLOADS_PATH = Path(__file__).with_name("payloads.json")
with PAYLOADS_PATH.open(encoding="utf-8") as payloads_file:
    PREDICTION_PAYLOADS = json.load(payloads_file)


class PredictionUser(HttpUser):
    """Usuario que solicita predicciones con estaciones de ejemplo reales."""

    wait_time = between(0.2, 1.0)

    @task
    def predict_station_risk(self):
        payload = random.choice(PREDICTION_PAYLOADS)
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
                return

            try:
                prediction = response.json()
            except ValueError:
                response.failure("La respuesta no es JSON")
                return

            expected_keys = {"risk_no_bikes", "risk_no_docks"}
            if set(prediction) != expected_keys:
                response.failure("La respuesta no contiene los dos riesgos esperados")
            elif any(prediction[key] not in (0, 1) for key in expected_keys):
                response.failure("La predicción contiene valores fuera de 0 y 1")
