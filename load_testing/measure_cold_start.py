"""Mide cuánto tarda la API en responder /health desde que se crea su contenedor.

Uso: python load_testing/measure_cold_start.py reports/load-tests/<carpeta> [corridas]
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

IMAGEN = os.environ.get("IMAGEN", "mibici-equipo50-api:latest")
PAYLOAD = json.loads(Path(__file__).with_name("payloads.json").read_text(encoding="utf-8"))[0]
URL = "http://127.0.0.1:18000"


def esperar_health(limite=120):
    inicio = time.perf_counter()
    while time.perf_counter() - inicio < limite:
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2) as respuesta:
                if respuesta.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.05)
    raise TimeoutError("La API no respondió /health")


def medir(corridas):
    filas = []
    for corrida in range(1, corridas + 1):
        subprocess.run(["docker", "rm", "-f", "finops-frio"], capture_output=True)
        inicio = time.perf_counter()
        subprocess.run(
            ["docker", "run", "-d", "--name", "finops-frio", "--cpus=1", "--memory=512m"]
            + ["-p", "18000:8000", IMAGEN],
            check=True,
            capture_output=True,
        )
        creado = time.perf_counter() - inicio
        esperar_health()
        listo = time.perf_counter() - inicio
        peticion = urllib.request.Request(
            f"{URL}/predict",
            data=json.dumps(PAYLOAD).encode(),
            headers={"Content-Type": "application/json"},
        )
        antes = time.perf_counter()
        urllib.request.urlopen(peticion, timeout=10).read()
        filas.append(
            {
                "corrida": corrida,
                "docker_run_s": round(creado, 3),
                "hasta_health_s": round(listo, 3),
                "primera_prediccion_ms": round((time.perf_counter() - antes) * 1000, 1),
            }
        )
        print(filas[-1], flush=True)
    subprocess.run(["docker", "rm", "-f", "finops-frio"], capture_output=True)
    return filas


if __name__ == "__main__":
    salida = Path(sys.argv[1])
    corridas = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    salida.mkdir(parents=True, exist_ok=True)
    resultado = medir(corridas)
    (salida / "arranque_frio.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
