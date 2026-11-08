"""Download the registered models that the API serves so the image can include them.

The MLflow artifacts live in local Windows paths that a container cannot read, so each model
is downloaded here by name and alias, and the Dockerfile copies the resulting folder.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici_ml.config import (
    MLFLOW_TRACKING_URI,
    SERVING_ALIAS,
    SERVING_MODELS,
    SERVING_MODELS_DIR,
)

app = typer.Typer()


def export_model(client: MlflowClient, name: str, alias: str, destination: Path) -> dict:
    """Download the version that the alias points to and return its lineage."""
    version = client.get_model_version_by_alias(name, alias)
    if destination.exists():
        shutil.rmtree(destination)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{version.version}", dst_path=str(destination)
    )
    return {
        "nombre": name,
        "alias": alias,
        "version": int(version.version),
        "run_id": version.run_id,
        "git_commit": version.tags.get("git_commit"),
    }


@app.command()
def main(
    question: list[str] = typer.Option(list(SERVING_MODELS), help="Pregunta; se puede repetir"),
    alias: str = SERVING_ALIAS,
    output_dir: Path = SERVING_MODELS_DIR,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> None:
    """Download the models that the API serves and write a manifest with their versions."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key in question:
        manifest[key] = export_model(client, SERVING_MODELS[key], alias, output_dir / key)
        logger.success(f"{key}: {SERVING_MODELS[key]} v{manifest[key]['version']} ({alias})")
    path = output_dir / "modelos.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Modelos listos en {output_dir}")


if __name__ == "__main__":
    app()
