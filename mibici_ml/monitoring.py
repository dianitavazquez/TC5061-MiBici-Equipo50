"""Build the P2 production reference and generate Evidently drift reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import pandas as pd
import typer

from mibici_ml.config import (
    CLUSTERING_TRIP_FILES,
    P2_ALPHA,
    P2_MONITORING_CURRENT_DATA_FILE,
    P2_MONITORING_CURRENT_END,
    P2_MONITORING_CURRENT_FILE,
    P2_MONITORING_CURRENT_START,
    P2_MONITORING_PSI_THRESHOLD,
    P2_MONITORING_REFERENCE_END,
    P2_MONITORING_REFERENCE_FILE,
    P2_MONITORING_REFERENCE_START,
    P2_MONITORING_REPORT_DIR,
    P2_MONITORING_SAMPLE_SIZE,
    P2_MONITORING_SUMMARY_DIR,
    P2_STATIONS_FILE,
    P2_TRAIN_FRACTION,
    PROJ_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from mibici_ml.dataset import (
    DEFAULT_FOLDER_ID,
    download_public_file,
    list_public_drive_folder,
    read_csv_compatible,
    sha256_file,
)
from mibici_ml.features import build_station_hour_dataset, load_clean_trips
from mibici_ml.modeling.model_recovery import DEFAULT_BUNDLED_MODEL_PATH
from mibici_ml.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGETS,
    chronological_split,
    evaluate,
)

app = typer.Typer(no_args_is_help=True)

MONITORED_COLUMNS = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, *TARGETS]
MONITORING_CATEGORICAL_COLUMNS = [*CATEGORICAL_FEATURES, "is_weekend", *TARGETS]
MONITORING_NUMERICAL_COLUMNS = [
    column for column in NUMERIC_FEATURES if column not in MONITORING_CATEGORICAL_COLUMNS
]


@app.callback()
def main() -> None:
    """Monitor the P2 champion against later MiBici production data."""


def ensure_current_source(
    raw_dir: Path = RAW_DATA_DIR,
    folder_id: str = DEFAULT_FOLDER_ID,
    force: bool = False,
) -> dict:
    """Download the current production month without changing the training manifest."""
    available = {item["name"]: item for item in list_public_drive_folder(folder_id)}
    if P2_MONITORING_CURRENT_FILE not in available:
        raise RuntimeError(
            f"No se encontró {P2_MONITORING_CURRENT_FILE} en la carpeta pública de Drive."
        )

    source = available[P2_MONITORING_CURRENT_FILE]
    destination = raw_dir / P2_MONITORING_CURRENT_FILE
    if force or not destination.exists():
        logger.info(f"Descargando lote actual {P2_MONITORING_CURRENT_FILE}")
        download_public_file(source["file_id"], destination)
    else:
        logger.info(f"Usando lote actual existente {destination}")

    return {
        **source,
        "url": f"https://drive.google.com/file/d/{source['file_id']}/view",
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _slice_period(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    if start_timestamp >= end_timestamp:
        raise ValueError("El inicio del periodo debe ser anterior al fin.")
    return frame[frame["hour"].ge(start_timestamp) & frame["hour"].lt(end_timestamp)].copy()


def prepare_monitoring_frames(
    raw_dir: Path = RAW_DATA_DIR,
    reference_path: Path = P2_MONITORING_REFERENCE_FILE,
    current_path: Path = P2_MONITORING_CURRENT_DATA_FILE,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Create the exact P2 train reference and the October 2026 production batch.

    The reference reproduces the champion's chronological 80% train partition. The current
    panel uses September only as historical context for lagged variables and is sliced to
    October after feature engineering.
    """
    if not rebuild and reference_path.exists() and current_path.exists():
        reference = pd.read_parquet(reference_path)
        current = pd.read_parquet(current_path)
        cutoff = reference["hour"].max() + pd.Timedelta(hours=1)
        return reference, current, {"cutoff": cutoff.isoformat(), "cache_used": True}

    stations_path = raw_dir / P2_STATIONS_FILE
    if not stations_path.exists():
        raise FileNotFoundError(f"No existe la nomenclatura requerida {stations_path}")
    stations = read_csv_compatible(stations_path)

    logger.info("Construyendo la referencia del conjunto realmente usado para ajustar P2")
    reference_trips, reference_counts = load_clean_trips(raw_dir, CLUSTERING_TRIP_FILES)
    reference_full = build_station_hour_dataset(reference_trips, stations, alpha=P2_ALPHA)
    reference_full = _slice_period(
        reference_full, P2_MONITORING_REFERENCE_START, P2_MONITORING_REFERENCE_END
    )
    reference, _, cutoff = chronological_split(reference_full, P2_TRAIN_FRACTION)

    logger.info("Construyendo octubre de 2026 con septiembre como contexto de rezagos")
    context_files = [CLUSTERING_TRIP_FILES[-1], P2_MONITORING_CURRENT_FILE]
    current_trips, current_counts = load_clean_trips(raw_dir, context_files)
    current_full = build_station_hour_dataset(current_trips, stations, alpha=P2_ALPHA)
    current = _slice_period(current_full, P2_MONITORING_CURRENT_START, P2_MONITORING_CURRENT_END)
    if reference.empty or current.empty:
        raise ValueError("La preparación produjo un conjunto de referencia o actual vacío.")

    reference_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(reference_path, index=False)
    current.to_parquet(current_path, index=False)
    metadata = {
        "cutoff": cutoff.isoformat(),
        "cache_used": False,
        "reference_trip_quality": reference_counts,
        "current_context_trip_quality": current_counts,
    }
    return reference, current, metadata


def summarize_quality(frame: pd.DataFrame) -> dict:
    """Return full-data quality indicators meaningful for a station-hour panel."""
    duplicate_keys = int(frame.duplicated(["station_id", "hour"]).sum())
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "stations": int(frame["station_id"].nunique()),
        "start": frame["hour"].min().isoformat(),
        "end": frame["hour"].max().isoformat(),
        "missing_values": int(frame[MONITORED_COLUMNS].isna().sum().sum()),
        "duplicate_station_hours": duplicate_keys,
        "constant_columns": [
            column for column in MONITORED_COLUMNS if frame[column].nunique(dropna=False) <= 1
        ],
        "target_positive_rate": {target: float(frame[target].mean()) for target in TARGETS},
    }


def summarize_station_coverage(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Quantify station categories that the fitted one-hot encoder has never observed."""
    reference_stations = set(reference["station_id"].unique())
    current_stations = set(current["station_id"].unique())
    unseen_values = sorted(current_stations - reference_stations)
    absent_values = sorted(reference_stations - current_stations)
    unseen = [value.item() if hasattr(value, "item") else value for value in unseen_values]
    absent = [value.item() if hasattr(value, "item") else value for value in absent_values]
    return {
        "reference_stations": len(reference_stations),
        "current_stations": len(current_stations),
        "shared_stations": len(reference_stations & current_stations),
        "unseen_in_training": unseen,
        "unseen_station_rows": int(current["station_id"].isin(unseen_values).sum()),
        "unseen_station_row_share": float(current["station_id"].isin(unseen_values).mean()),
        "absent_from_current": absent,
    }


def evaluate_champion(
    current: pd.DataFrame, model_path: Path = DEFAULT_BUNDLED_MODEL_PATH
) -> dict:
    """Evaluate the immutable bundled champion when delayed October labels are available."""
    if not (model_path / "MLmodel").exists():
        raise FileNotFoundError(f"No se encontró el champion empaquetado en {model_path}")

    # Importing lazily keeps data-only helpers usable without initializing MLflow.
    import mlflow.sklearn

    model = mlflow.sklearn.load_model(str(model_path))
    features = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    predictions = model.predict(current[features])
    probabilities = model.predict_proba(current[features])
    model_info_path = model_path / "model-info.json"
    model_info = (
        json.loads(model_info_path.read_text(encoding="utf-8")) if model_info_path.exists() else {}
    )
    return {
        "registered_model": model_info.get("registered_model"),
        "version": model_info.get("version"),
        "run_id": model_info.get("run_id"),
        "metrics": evaluate(current[TARGETS], predictions, probabilities),
    }


def deterministic_sample(frame: pd.DataFrame, size: int) -> pd.DataFrame:
    """Bound report cost while keeping sampling reproducible."""
    if size <= 0:
        raise ValueError("sample_size debe ser positivo.")
    if len(frame) <= size:
        return frame.copy()
    return frame.sample(n=size, random_state=RANDOM_SEED).sort_index()


def extract_drift_metrics(snapshot: object) -> tuple[dict, list[dict]]:
    """Extract a stable compact summary from an Evidently snapshot."""
    payload = snapshot.dict()
    status_by_metric = {
        test["metric_config"]["metric_id"]: test["status"]
        for test in payload.get("tests", [])
        if test.get("metric_config", {}).get("metric_id")
    }
    dataset_drift = {}
    columns = []
    for metric in payload["metrics"]:
        metric_type = metric.get("config", {}).get("type", "")
        if metric_type.endswith("DriftedColumnsCount"):
            dataset_drift = metric["value"]
            dataset_drift["status"] = status_by_metric.get(metric["id"], "UNKNOWN")
        elif metric_type.endswith("ValueDrift"):
            columns.append(
                {
                    "column": metric["config"]["column"],
                    "method": metric["config"]["method"],
                    "threshold": metric["config"]["threshold"],
                    "drift_score": metric["value"],
                    "drift_detected": status_by_metric.get(metric["id"]) == "FAIL",
                }
            )
    return dataset_drift, sorted(columns, key=lambda item: item["drift_score"], reverse=True)


def build_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    sample_size: int = P2_MONITORING_SAMPLE_SIZE,
    threshold: float = P2_MONITORING_PSI_THRESHOLD,
):
    """Run feature/target drift and data-quality summaries with Evidently."""
    if not 0 < threshold:
        raise ValueError("El umbral PSI debe ser positivo.")
    reference_sample = deterministic_sample(reference, sample_size)
    current_sample = deterministic_sample(current, sample_size)
    columns = ["hour", *MONITORED_COLUMNS]
    # Evidently 0.7.23 validates category labels as native Python values; converting to
    # strings also prevents station identifiers and binary flags from being inferred as
    # continuous measurements.
    for column in MONITORING_CATEGORICAL_COLUMNS:
        reference_sample[column] = reference_sample[column].astype(str)
        current_sample[column] = current_sample[column].astype(str)
    definition = DataDefinition(
        timestamp="hour",
        numerical_columns=MONITORING_NUMERICAL_COLUMNS,
        categorical_columns=MONITORING_CATEGORICAL_COLUMNS,
    )
    reference_dataset = Dataset.from_pandas(reference_sample[columns], definition)
    current_dataset = Dataset.from_pandas(current_sample[columns], definition)
    report = Report(
        [
            DataDriftPreset(
                columns=MONITORED_COLUMNS,
                num_method="psi",
                cat_method="psi",
                num_threshold=threshold,
                cat_threshold=threshold,
            ),
            DataSummaryPreset(columns=MONITORED_COLUMNS),
        ],
        include_tests=True,
        metadata={
            "reference_period": f"{P2_MONITORING_REFERENCE_START} a corte de entrenamiento",
            "current_period": f"{P2_MONITORING_CURRENT_START} a {P2_MONITORING_CURRENT_END}",
        },
    )
    snapshot = report.run(current_dataset, reference_dataset)
    return snapshot, len(reference_sample), len(current_sample)


@app.command("run")
def run_monitoring(
    raw_dir: Path = RAW_DATA_DIR,
    force_download: bool = False,
    rebuild: bool = False,
    sample_size: int = P2_MONITORING_SAMPLE_SIZE,
) -> None:
    """Download October 2026, compare it with P2 train data, and save the report."""
    source = ensure_current_source(raw_dir, force=force_download)
    reference, current, preparation = prepare_monitoring_frames(raw_dir, rebuild=rebuild)
    snapshot, reference_sample_rows, current_sample_rows = build_evidently_report(
        reference, current, sample_size
    )

    P2_MONITORING_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    P2_MONITORING_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    html_path = P2_MONITORING_REPORT_DIR / "drift_2026_10.html"
    json_path = P2_MONITORING_REPORT_DIR / "drift_2026_10.json"
    summary_path = P2_MONITORING_SUMMARY_DIR / "p2_drift_2026_10.json"
    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    dataset_drift, column_drift = extract_drift_metrics(snapshot)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidently_version": "0.7.23",
        "method": "PSI",
        "threshold": P2_MONITORING_PSI_THRESHOLD,
        "reference": {
            "description": "Partición cronológica usada para ajustar el champion P2",
            "configured_source_period": {
                "start_inclusive": P2_MONITORING_REFERENCE_START,
                "end_exclusive": P2_MONITORING_REFERENCE_END,
            },
            "actual_training_cutoff_exclusive": preparation["cutoff"],
            "quality": summarize_quality(reference),
            "sample_rows_in_evidently": reference_sample_rows,
        },
        "current": {
            "description": "Comportamiento posterior observado en octubre de 2026",
            "period": {
                "start_inclusive": P2_MONITORING_CURRENT_START,
                "end_exclusive": P2_MONITORING_CURRENT_END,
            },
            "source": source,
            "quality": summarize_quality(current),
            "sample_rows_in_evidently": current_sample_rows,
        },
        "preparation": preparation,
        "station_coverage": summarize_station_coverage(reference, current),
        "champion_performance_current": evaluate_champion(current),
        "dataset_drift": dataset_drift,
        "column_drift": column_drift,
        "artifacts": {
            "html": str(html_path.relative_to(PROJ_ROOT)),
            "json": str(json_path.relative_to(PROJ_ROOT)),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Reporte Evidently: {html_path}")
    logger.success(f"Resumen versionable: {summary_path}")


if __name__ == "__main__":
    app()
