"""Recover and load the P2 model selected in the MLflow Model Registry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici_ml.config import MLFLOW_TRACKING_URI, P2_REGISTERED_MODEL_NAME, PROJ_ROOT

DEFAULT_ALIAS = "champion"
DEFAULT_BUNDLED_MODEL_PATH = PROJ_ROOT / "models" / "serving" / "p2"
load_pyfunc_model = mlflow.pyfunc.load_model

app = typer.Typer(no_args_is_help=True)


def registered_model_uri(
    name: str = P2_REGISTERED_MODEL_NAME,
    *,
    version: str | int | None = None,
    alias: str | None = DEFAULT_ALIAS,
) -> str:
    """Build an MLflow model URI from exactly one immutable version or alias."""
    if version is not None:
        return f"models:/{name}/{version}"
    if alias:
        return f"models:/{name}@{alias}"
    raise ValueError("Se requiere una versión o un alias del modelo registrado.")


def resolve_model_version(
    client: MlflowClient,
    name: str,
    *,
    version: str | int | None = None,
    alias: str | None = DEFAULT_ALIAS,
) -> tuple[Any, str | None]:
    """Resolve an alias to a fixed version, or validate an explicit version."""
    if version is not None:
        return client.get_model_version(name, str(version)), None
    if not alias:
        raise ValueError("Se requiere una versión o un alias del modelo registrado.")
    return client.get_model_version_by_alias(name, alias), alias


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_model_root(downloaded_path: Path) -> Path:
    if (downloaded_path / "MLmodel").is_file():
        return downloaded_path
    candidates = list(downloaded_path.rglob("MLmodel"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"La descarga debe contener exactamente un archivo MLmodel; se encontraron "
            f"{len(candidates)} en {downloaded_path}."
        )
    return candidates[0].parent


def recover_model(
    *,
    tracking_uri: str,
    name: str = P2_REGISTERED_MODEL_NAME,
    version: str | int | None = None,
    alias: str | None = DEFAULT_ALIAS,
    destination: Path = DEFAULT_BUNDLED_MODEL_PATH,
) -> dict[str, Any]:
    """Download a registered model as a fixed version and write its provenance."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    model_version, resolved_alias = resolve_model_version(
        client, name, version=version, alias=alias
    )
    fixed_uri = registered_model_uri(name, version=model_version.version, alias=None)

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="p2-model-", dir=destination.parent) as temporary:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=fixed_uri,
                dst_path=temporary,
            )
        )
        model_root = _find_model_root(downloaded)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(model_root, destination)

    files = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "registered_model": name,
        "version": int(model_version.version),
        "alias_at_download": resolved_alias,
        "model_uri": fixed_uri,
        "run_id": model_version.run_id,
        "files_sha256": files,
    }
    (destination / "model-info.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_p2_model(
    *,
    source: str | None = None,
    model_path: str | Path | None = None,
    tracking_uri: str | None = None,
    name: str | None = None,
    version: str | int | None = None,
    alias: str | None = None,
):
    """Load P2 from the bundled copy by default, or directly from the Registry."""
    source = (source or os.getenv("MIBICI_MODEL_SOURCE", "bundled")).lower()
    if source == "bundled":
        path = Path(
            model_path or os.getenv("MIBICI_MODEL_PATH", DEFAULT_BUNDLED_MODEL_PATH)
        ).resolve()
        if not (path / "MLmodel").is_file():
            raise FileNotFoundError(f"No se encontró un modelo MLflow empaquetado en {path}.")
        return load_pyfunc_model(str(path))
    if source != "registry":
        raise ValueError("MIBICI_MODEL_SOURCE debe ser 'bundled' o 'registry'.")

    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    name = name or os.getenv("MLFLOW_MODEL_NAME", P2_REGISTERED_MODEL_NAME)
    version = version or os.getenv("MLFLOW_MODEL_VERSION")
    alias = alias or os.getenv("MLFLOW_MODEL_ALIAS", DEFAULT_ALIAS)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return load_pyfunc_model(
        registered_model_uri(name, version=version, alias=None if version else alias)
    )


@app.command("download")
def download_command(
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, envvar="MLFLOW_TRACKING_URI"),
    name: str = typer.Option(P2_REGISTERED_MODEL_NAME, envvar="MLFLOW_MODEL_NAME"),
    version: str | None = typer.Option(None, envvar="MLFLOW_MODEL_VERSION"),
    alias: str = typer.Option(DEFAULT_ALIAS, envvar="MLFLOW_MODEL_ALIAS"),
    destination: Path = typer.Option(DEFAULT_BUNDLED_MODEL_PATH),
) -> None:
    """Download P2 from the Registry for inclusion in the container image."""
    manifest = recover_model(
        tracking_uri=tracking_uri,
        name=name,
        version=version,
        alias=None if version else alias,
        destination=destination,
    )
    logger.success(
        "Modelo {} versión {} recuperado en {}",
        manifest["registered_model"],
        manifest["version"],
        destination,
    )


if __name__ == "__main__":
    app()
