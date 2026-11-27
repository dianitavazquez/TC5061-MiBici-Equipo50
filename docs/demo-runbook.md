# Runbook del demo MLOps

## Preparación previa

```bash
git switch main
git pull --ff-only
docker compose down -v --remove-orphans
docker compose build
docker compose up -d --wait
docker compose ps -a
```

El borrado del volumen es intencional únicamente durante el ensayo: demuestra que el Registry
se reconstruye desde los artefactos versionados. No usar `down -v` sobre un entorno que
contenga experimentos que deban conservarse.

Validación rápida:

```bash
curl --fail http://localhost:8000/health
curl --fail http://localhost:8000/model-info
curl --fail http://localhost:5050/health
```

Abrir antes de grabar:

- API: http://localhost:8000/docs
- MLflow: http://localhost:5050
- Evidently: `reports/evidently/p2/drift_2026_10.html`
- Model Card: `MODEL_CARD.md`

## Guion sugerido

### 1. Repositorio y reproducibilidad

Mostrar `MLproject`, `python_env.yaml`, `requirements.txt`, `mibici_ml/features.py` y la
estructura Cookiecutter. Explicar que ingesta, features, entrenamiento, selección, registro y
monitoreo son entry points reproducibles.

### 2. Tracking y Model Registry

En MLflow mostrar:

1. experimento `mibici-equipo50-demo`;
2. modelo `mibici-equipo50-p2-riesgo`;
3. aliases `champion` y `challenger`;
4. tags `source_run_id`, `source_registry_version`, `lifecycle` y SHA-256.

Repetir el bootstrap para demostrar idempotencia:

```bash
docker compose run --rm registry-bootstrap
```

El comando debe reutilizar las versiones existentes, no crear duplicados.

### 3. API

En Swagger ejecutar `GET /health`, `GET /model-info` y `POST /predict`. Para usar el ejemplo
real desde terminal:

```bash
python3 - <<'PY'
import json
import urllib.request
from pathlib import Path

example = json.loads(
    Path("models/serving/p2/serving_input_example.json").read_text()
)["dataframe_split"]
payload = dict(zip(example["columns"], example["data"][0]))
request = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(request)))
PY
```

### 4. Carga y operación

Para una ejecución corta durante la grabación:

```bash
USERS=10 SPAWN_RATE=2 RUN_TIME=30s make load-test
```

Mostrar el nuevo `reports/load-tests/<fecha>/README.md`, p95, tasa de error, throughput y uso
de recursos. Si el tiempo es limitado, usar la evidencia versionada
`reports/load-tests/20261122T-locust-50u/`.

### 5. Drift, A/B y gobernanza

Mostrar:

- Evidently: drift en 3/25 columnas y 34 estaciones nuevas;
- A/B: `references/ab_testing/p2_ab_2026_10.json`, donde gana el champion;
- FinOps: `references/finops/p2_finops.json` y sus gráficas;
- `MODEL_CARD.md`: alcance, limitaciones y reglas de reentrenamiento.

La regeneración de Evidently descarga y procesa las fuentes completas, por lo que para una
grabación corta se recomienda mostrar el reporte ya versionado. El comando reproducible es:

```bash
make monitor-p2
```

## Cierre y recuperación

```bash
docker compose logs --tail=100
docker compose down
```

Si algo falla:

```bash
docker compose ps -a
docker compose logs mlflow registry-bootstrap api
docker compose down -v --remove-orphans
docker compose up --build -d --wait
```
