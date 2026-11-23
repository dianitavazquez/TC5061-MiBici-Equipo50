"""Prueba de saturación de /predict: mismas peticiones que el escenario de Anuar, sin espera."""

import json
from pathlib import Path
import random

from locust import HttpUser, LoadTestShape, constant, task

PAYLOADS = json.loads(Path(__file__).with_name("payloads.json").read_text(encoding="utf-8"))
ESCALONES = [1, 2, 4, 8, 16, 32, 64]  # usuarios concurrentes por escalón
SEGUNDOS_POR_ESCALON = 40


class UsuarioSinEspera(HttpUser):
    wait_time = constant(0)

    @task
    def predecir(self):
        with self.client.post(
            "/predict", json=random.choice(PAYLOADS), name="POST /predict", catch_response=True
        ) as respuesta:
            if respuesta.status_code != 200:
                respuesta.failure(f"HTTP {respuesta.status_code}")
            elif set(respuesta.json()) != {"risk_no_bikes", "risk_no_docks"}:
                respuesta.failure("Respuesta sin los dos riesgos")


class Escalones(LoadTestShape):
    def tick(self):
        paso = int(self.get_run_time() // SEGUNDOS_POR_ESCALON)
        if paso >= len(ESCALONES):
            return None
        return ESCALONES[paso], ESCALONES[paso]
