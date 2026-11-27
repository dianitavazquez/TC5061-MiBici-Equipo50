"""API REST para predicciones del modelo MiBici P2."""

from functools import lru_cache
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd

from mibici_ml.api.schemas import PredictionInput, PredictionOutput
from mibici_ml.modeling.model_recovery import load_p2_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="MiBici P2 Prediction API",
    description="API para predecir riesgos de disponibilidad en estaciones MiBici.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model():
    """Carga el modelo P2 una sola vez por proceso."""
    return load_p2_model()


@app.get("/health")
def health():
    """Verifica que el modelo esté disponible."""
    try:
        get_model()
    except Exception:
        logger.exception("Error al cargar el modelo P2")
        raise HTTPException(
            status_code=503,
            detail="Modelo P2 no disponible",
        )

    return {"status": "ok", "model": "P2"}


@app.get("/model-info")
def model_info():
    """Expone la versión y el linaje mínimo del modelo activo para operación y demo."""
    model_path = Path(os.getenv("MIBICI_MODEL_PATH", "models/serving/p2"))
    manifest_path = model_path / "model-info.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=503, detail="Metadatos del modelo no disponibles")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "registered_model": manifest["registered_model"],
        "version": manifest["version"],
        "alias": manifest.get("alias_at_download"),
        "source_run_id": manifest["run_id"],
        "model_uri": manifest["model_uri"],
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    """Genera las dos predicciones binarias del modelo P2."""
    try:
        model = get_model()
    except Exception:
        logger.exception("Error al cargar el modelo P2")
        raise HTTPException(
            status_code=503,
            detail="Modelo P2 no disponible",
        )

    try:
        # Construir DataFrame con las variables del modelo P2.
        features = pd.DataFrame(
            [payload.model_dump()],
            columns=list(PredictionInput.model_fields),
        )

        # Ajustar los tipos de datos según la firma de MLflow.
        features = features.astype(
            {
                "capacity": "int64",
                "station_id": "int64",
                "is_weekend": "int32",
            }
        )

        # Ejecutar la predicción.
        predictions = model.predict(features)

        # El modelo P2 devuelve dos predicciones por registro.
        risk_no_bikes, risk_no_docks = predictions[0]

        return PredictionOutput(
            risk_no_bikes=int(risk_no_bikes),
            risk_no_docks=int(risk_no_docks),
        )

    except Exception:
        logger.exception("Error durante la predicción")
        raise HTTPException(
            status_code=500,
            detail="Error al generar la predicción",
        )
