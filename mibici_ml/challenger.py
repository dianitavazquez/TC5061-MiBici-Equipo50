
"""Train and register an alternative P2 model as MLflow challenger."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from loguru import logger
import typer

from mibici_ml.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    P2_REGISTERED_MODEL_NAME,
)

from pathlib import Path
import pandas as pd
from mibici_ml.config import (
    CLUSTERING_TRIP_FILES,
    P2_ALPHA,
    P2_MONITORING_REFERENCE_START,
    P2_MONITORING_REFERENCE_END,
    P2_STATIONS_FILE,
    P2_TRAIN_FRACTION,
    PROJ_ROOT,
)

from mibici_ml.features import (
    build_station_hour_dataset,
    load_clean_trips,
    read_csv_compatible,
)

from mibici_ml.monitoring import _slice_period
from mibici_ml.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGETS,
    build_pipeline,
    chronological_split,
    evaluate,
    git_commit,
)

app = typer.Typer()

CHALLENGER_CONFIG = {
    "class_weight": "balanced",
    "c": 0.5,
    "max_iter": 1000,
}

def prepare_challenger_reference():
    """Rebuild the historical P2 training reference without October data."""

    raw_dir = Path(PROJ_ROOT) / "data" / "raw"

    stations = read_csv_compatible(raw_dir / P2_STATIONS_FILE)

    trips, _ = load_clean_trips(
        raw_dir,
        CLUSTERING_TRIP_FILES,
    )

    reference_full = build_station_hour_dataset(
        trips,
        stations,
        alpha=P2_ALPHA,
    )

    reference_full = _slice_period(
        reference_full,
        P2_MONITORING_REFERENCE_START,
        P2_MONITORING_REFERENCE_END,
    )

    reference, _, cutoff = chronological_split(
        reference_full,
        P2_TRAIN_FRACTION,
    )

    return reference, {"cutoff": cutoff.isoformat()}

def train_challenger() -> dict:
    """Train P2 challenger without using the current evaluation period."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        P2_REGISTERED_MODEL_NAME, "champion"
    )
    logger.info("Champion existente: versión {}", champion.version)

    reference, metadata = prepare_challenger_reference()

    if reference.empty:
        raise ValueError("El conjunto de referencia está vacío.")

    logger.info(
        "Referencia histórica preparada: {} filas",
        len(reference),
    )

    train, validation, cutoff = chronological_split(reference, 0.8)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    logger.info(
        "Entrenamiento: {} filas; validación: {} filas",
        len(train),
        len(validation),
    )

    model = build_pipeline(**CHALLENGER_CONFIG)

    with mlflow.start_run(run_name="p2-challenger-dcvd") as run:
        mlflow.set_tags({
            "team": "50",
            "question": "P2",
            "variant": "challenger",
            "stage_of_work": "alternative_training",
            "git_commit": git_commit(),
            "evaluation_period_reserved": "2026-10",
        })

        mlflow.log_params({
            "class_weight": CHALLENGER_CONFIG["class_weight"],
            "C": CHALLENGER_CONFIG["c"],
            "max_iter": CHALLENGER_CONFIG["max_iter"],
            "penalty": "l2",
            "solver": "lbfgs",
            "training_source": "monitoring_reference",
            "validation_cutoff": cutoff.isoformat(),
            "train_rows": len(train),
            "validation_rows": len(validation),
        })

        model.fit(train[features], train[TARGETS])

        predictions = model.predict(validation[features])
        probabilities = model.predict_proba(validation[features])
        metrics = evaluate(validation[TARGETS], predictions, probabilities)

        mlflow.log_metrics({
            f"validation_{key}": float(value)
            for key, value in metrics.items()
        })

        model.fit(reference[features], reference[TARGETS])

        input_example = reference[features].head(5)
        signature = infer_signature(
            input_example, model.predict(input_example)
        )

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow.log_dict({
            "config": CHALLENGER_CONFIG,
            "reference_rows": len(reference),
            "evaluation_period": "2026-10",
            "evaluation_status": "pending",
            "validation_cutoff": cutoff.isoformat(),
            "monitoring_metadata": {
                "cutoff": metadata.get("cutoff"),
            },
        }, "config/challenger_training.json")

        run_id = run.info.run_id

    registered = mlflow.register_model(
        f"models:/{model_info.model_id}",
        P2_REGISTERED_MODEL_NAME,
    )

    client.update_model_version(
        P2_REGISTERED_MODEL_NAME,
        registered.version,
        description=(
            "Challenger P2: regresión logística multisalida, "
            "class_weight=balanced, C=0.5, max_iter=1000. "
            "Entrenado con la referencia de monitoreo; "
            "octubre de 2026 reservado para A/B testing."
        ),
    )

    for key, value in {
        "source_run_id": run_id,
        "git_commit": git_commit(),
        "training_reference_rows": len(reference),
        "validation_macro_f1": metrics["macro_f1"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }.items():
        client.set_model_version_tag(
            P2_REGISTERED_MODEL_NAME,
            registered.version,
            key,
            str(value),
        )

    client.set_registered_model_alias(
        P2_REGISTERED_MODEL_NAME,
        "challenger",
        registered.version,
    )

    result = {
        "registered_model": P2_REGISTERED_MODEL_NAME,
        "version": registered.version,
        "alias": "challenger",
        "run_id": run_id,
        "validation_metrics": metrics,
    }

    logger.success("Challenger registrado: {}", json.dumps(result))
    return result


@app.command()
def main() -> None:
    """Train and register the challenger."""
    train_challenger()


if __name__ == "__main__":
    app()