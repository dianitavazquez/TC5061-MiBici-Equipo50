"""Seed a fresh MLflow Model Registry with the immutable P2 demo models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import typer

from mibici_ml.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    P2_REGISTERED_MODEL_NAME,
)
from mibici_ml.modeling.model_recovery import DEFAULT_BUNDLED_MODEL_PATH

app = typer.Typer(no_args_is_help=True)

CHALLENGER_MODEL_PATH = DEFAULT_BUNDLED_MODEL_PATH.parent / "p2_challenger"
MODEL_VARIANTS = (
    {
        "alias": "champion",
        "model_path": DEFAULT_BUNDLED_MODEL_PATH,
        "lifecycle": "production",
        "description": "Champion aprobado por macro-F1 y validado por el A/B test de octubre.",
    },
    {
        "alias": "challenger",
        "model_path": CHALLENGER_MODEL_PATH,
        "lifecycle": "candidate",
        "description": "Challenger conservado para comparación y futuras evaluaciones.",
    },
)


def read_model_manifest(path: Path) -> dict[str, Any]:
    """Read and validate the provenance committed beside a bundled MLflow model."""
    manifest_path = path / "model-info.json"
    if not (path / "MLmodel").is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Modelo o manifiesto incompleto en {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"registered_model", "version", "run_id", "files_sha256"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Manifiesto incompleto en {manifest_path}: faltan {sorted(missing)}")
    return manifest


def _find_seeded_version(client: MlflowClient, model_name: str, manifest: dict[str, Any]):
    """Find a version previously seeded from the same immutable bundled artifact."""
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
    except MlflowException:
        return None
    fingerprint = manifest["files_sha256"].get("MLmodel")
    for version in versions:
        tags = dict(version.tags or {})
        if (
            tags.get("source_registry_version") == str(manifest["version"])
            and tags.get("bundled_mlmodel_sha256") == fingerprint
        ):
            return version
    return None


def _wait_until_ready(
    client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 60
):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = client.get_model_version(model_name, version)
        status = str(current.status).upper()
        if status == "READY":
            return current
        if status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Falló el registro de {model_name} versión {version}")
        time.sleep(0.5)
    raise TimeoutError(f"{model_name} versión {version} no quedó lista en {timeout_seconds}s")


def ensure_legacy_version(
    client: MlflowClient,
    model_name: str,
    model_path: Path = DEFAULT_BUNDLED_MODEL_PATH,
) -> None:
    """Create the archived v1 slot so demo versions match their original Registry numbers."""
    try:
        if client.search_model_versions(f"name = '{model_name}'"):
            return
    except MlflowException:
        pass

    with mlflow.start_run(run_name="demo-registry-bootstrap-legacy-v1") as run:
        mlflow.set_tags(
            {
                "team": "50",
                "question": "P2",
                "stage_of_work": "registry-bootstrap",
                "lifecycle": "archived",
            }
        )
        artifact_path = "p2-legacy-v1"
        mlflow.log_artifacts(str(model_path), artifact_path=artifact_path)
        registered = mlflow.register_model(
            f"runs:/{run.info.run_id}/{artifact_path}",
            model_name,
        )
    version = _wait_until_ready(client, model_name, str(registered.version))
    client.set_model_version_tag(model_name, str(version.version), "lifecycle", "archived")
    client.set_model_version_tag(model_name, str(version.version), "source_registry_version", "1")
    client.set_model_version_tag(model_name, str(version.version), "demo_seeded", "true")
    client.update_model_version(
        model_name,
        str(version.version),
        description=(
            "Versión histórica archivada para conservar la numeración original. "
            "No tiene alias operativo."
        ),
    )


def seed_variant(
    client: MlflowClient,
    *,
    model_name: str,
    alias: str,
    model_path: Path,
    lifecycle: str,
    description: str,
) -> dict[str, Any]:
    """Register one bundled model idempotently and assign its lifecycle alias."""
    manifest = read_model_manifest(model_path)
    existing = _find_seeded_version(client, model_name, manifest)
    if existing is None:
        run_name = f"demo-registry-bootstrap-{alias}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "team": "50",
                    "question": "P2",
                    "stage_of_work": "registry-bootstrap",
                    "source_run_id": manifest["run_id"],
                    "source_registry_version": str(manifest["version"]),
                    "lifecycle": lifecycle,
                }
            )
            mlflow.log_params(
                {
                    "registered_model": model_name,
                    "alias": alias,
                    "bundled_source_version": manifest["version"],
                }
            )
            artifact_path = f"p2-{alias}"
            mlflow.log_artifacts(str(model_path), artifact_path=artifact_path)
            registered = mlflow.register_model(
                f"runs:/{run.info.run_id}/{artifact_path}",
                model_name,
            )
        version = _wait_until_ready(client, model_name, str(registered.version))
    else:
        version = existing

    lineage = {
        "source_run_id": manifest["run_id"],
        "source_registry_version": manifest["version"],
        "source_model_uri": manifest.get("model_uri", ""),
        "bundled_mlmodel_sha256": manifest["files_sha256"].get("MLmodel", ""),
        "lifecycle": lifecycle,
        "demo_seeded": "true",
    }
    for key, value in lineage.items():
        client.set_model_version_tag(model_name, str(version.version), key, str(value))
    client.update_model_version(
        model_name,
        str(version.version),
        description=f"{description} Linaje original: run {manifest['run_id']}.",
    )
    client.set_registered_model_alias(model_name, alias, str(version.version))
    return {
        "alias": alias,
        "registry_version": int(version.version),
        "source_registry_version": int(manifest["version"]),
        "source_run_id": manifest["run_id"],
        "reused": existing is not None,
    }


def bootstrap_registry(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = f"{MLFLOW_EXPERIMENT}-demo",
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Populate an empty Registry and verify that the champion alias is loadable."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    ensure_legacy_version(client, P2_REGISTERED_MODEL_NAME)
    results = [
        seed_variant(client, model_name=P2_REGISTERED_MODEL_NAME, **variant)
        for variant in MODEL_VARIANTS
    ]
    client.update_registered_model(
        P2_REGISTERED_MODEL_NAME,
        description=(
            "Clasificador P2 de riesgo MiBici. Los aliases champion y challenger se "
            "inicializan desde artefactos inmutables con linaje preservado para el demo."
        ),
    )
    client.set_registered_model_tag(P2_REGISTERED_MODEL_NAME, "team", "50")
    client.set_registered_model_tag(P2_REGISTERED_MODEL_NAME, "question", "P2")
    client.set_registered_model_tag(P2_REGISTERED_MODEL_NAME, "governance", "MODEL_CARD.md")

    if verify:
        mlflow.pyfunc.load_model(f"models:/{P2_REGISTERED_MODEL_NAME}@champion")
    return results


@app.command("seed")
def seed_command(
    tracking_uri: str = typer.Option(
        os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI),
        envvar="MLFLOW_TRACKING_URI",
    ),
    experiment_name: str = typer.Option(
        os.getenv("MLFLOW_EXPERIMENT_NAME", f"{MLFLOW_EXPERIMENT}-demo"),
        envvar="MLFLOW_EXPERIMENT_NAME",
    ),
) -> None:
    """Seed champion/challenger into the configured MLflow server."""
    results = bootstrap_registry(tracking_uri, experiment_name)
    for result in results:
        logger.success(
            "{} -> versión {} (origen v{}, reutilizada={})",
            result["alias"],
            result["registry_version"],
            result["source_registry_version"],
            result["reused"],
        )


if __name__ == "__main__":
    app()
