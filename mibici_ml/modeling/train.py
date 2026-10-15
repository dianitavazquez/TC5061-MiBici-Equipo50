"""Train and track the P2 logistic-regression baseline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mibici_ml.config import FIGURES_DIR, RAW_DATA_DIR
from mibici_ml.dataset import read_csv_compatible
from mibici_ml.features import build_station_hour_dataset, load_clean_trips

app = typer.Typer(no_args_is_help=True)

TARGETS = ["risk_no_bikes", "risk_no_docks"]
CATEGORICAL_FEATURES = ["station_id"]
NUMERIC_FEATURES = [
    "capacity",
    "departures_lag_1h",
    "arrivals_lag_1h",
    "net_departures_lag_1h",
    "departures_mean_3h",
    "departures_mean_24h",
    "departures_mean_168h",
    "arrivals_mean_3h",
    "arrivals_mean_24h",
    "arrivals_mean_168h",
    "net_departures_mean_3h",
    "net_departures_mean_24h",
    "net_departures_mean_168h",
    "risk_no_bikes_lag_1h",
    "risk_no_docks_lag_1h",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
]


@app.callback()
def main():
    """Train MiBici models and register their experiments."""


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.8):
    """Split by unique timestamps, preserving the future for evaluation."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction debe estar en el intervalo (0, 1).")
    timestamps = np.sort(frame["hour"].unique())
    cutoff_index = min(int(len(timestamps) * train_fraction), len(timestamps) - 1)
    cutoff = pd.Timestamp(timestamps[cutoff_index])
    train = frame[frame["hour"] < cutoff].copy()
    test = frame[frame["hour"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("La partición cronológica produjo un conjunto vacío.")
    return train, test, cutoff


def build_pipeline() -> Pipeline:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    classifier = MultiOutputClassifier(
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate(y_true: pd.DataFrame, predictions: np.ndarray, probabilities: list) -> dict:
    metrics = {}
    for index, target in enumerate(TARGETS):
        actual = y_true[target].to_numpy()
        predicted = predictions[:, index]
        probability = probabilities[index][:, 1]
        metrics[f"{target}_f1"] = f1_score(actual, predicted, zero_division=0)
        metrics[f"{target}_precision"] = precision_score(actual, predicted, zero_division=0)
        metrics[f"{target}_recall"] = recall_score(actual, predicted, zero_division=0)
        metrics[f"{target}_average_precision"] = average_precision_score(actual, probability)
        metrics[f"{target}_roc_auc"] = roc_auc_score(actual, probability)
        metrics[f"{target}_positive_rate"] = float(actual.mean())
    metrics["macro_f1"] = float(np.mean([metrics[f"{target}_f1"] for target in TARGETS]))
    return metrics


def save_confusion_matrices(
    y_true: pd.DataFrame, predictions: np.ndarray, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, target in enumerate(TARGETS):
        ConfusionMatrixDisplay.from_predictions(
            y_true[target], predictions[:, index], ax=axes[index], colorbar=False
        )
        axes[index].set_title(target)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def read_data_version(raw_dir: Path) -> str:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        return "unversioned"
    return json.loads(manifest_path.read_text(encoding="utf-8"))["data_version"]


@app.command("p2-baseline")
def train_p2_baseline(
    raw_dir: Path = RAW_DATA_DIR,
    date_start: str = "2025-10-01",
    date_end: str = "2026-10-01",
    alpha: float = 0.5,
    train_fraction: float = 0.8,
    experiment_name: str = "mibici-equipo50",
    tracking_uri: str = "sqlite:///mlflow.db",
):
    """Train both P2 proxy-risk labels and record the run in MLflow."""
    logger.info("Cargando y limpiando viajes")
    trips, quality_counts = load_clean_trips(raw_dir)
    stations = read_csv_compatible(raw_dir / "nomenclatura_2026_09.csv")
    logger.info("Construyendo panel estación-hora y variables rezagadas")
    frame = build_station_hour_dataset(trips, stations, alpha=alpha)
    start_timestamp = pd.Timestamp(date_start)
    end_timestamp = pd.Timestamp(date_end)
    if start_timestamp >= end_timestamp:
        raise ValueError("date_start debe ser anterior a date_end.")
    frame = frame[frame["hour"].ge(start_timestamp) & frame["hour"].lt(end_timestamp)].copy()
    train, test, cutoff = chronological_split(frame, train_fraction)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = build_pipeline()
    logger.info(f"Entrenando con {len(train):,} filas; evaluando con {len(test):,}")
    model.fit(train[features], train[TARGETS])
    predictions = model.predict(test[features])
    probabilities = model.predict_proba(test[features])
    metrics = evaluate(test[TARGETS], predictions, probabilities)

    artifact_path = FIGURES_DIR / "p2_confusion_matrices.png"
    save_confusion_matrices(test[TARGETS], predictions, artifact_path)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="P2-logistic-regression-demand-risk"):
        mlflow.set_tags(
            {
                "team": "50",
                "question": "P2",
                "git_commit": git_commit(),
                "target_type": "demand_pressure_proxy",
            }
        )
        mlflow.log_params(
            {
                "model": "multioutput_logistic_regression",
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced",
                "alpha": alpha,
                "horizon_hours": 1,
                "date_start_inclusive": date_start,
                "date_end_exclusive": date_end,
                "train_fraction": train_fraction,
                "split_cutoff": cutoff.isoformat(),
                "data_start": frame["hour"].min().isoformat(),
                "data_end": frame["hour"].max().isoformat(),
                "data_version": read_data_version(raw_dir),
                **quality_counts,
                "train_rows": len(train),
                "test_rows": len(test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path), artifact_path="figures")
        for metadata_file in [raw_dir / "manifest.json", raw_dir / "profile.json"]:
            if metadata_file.exists():
                mlflow.log_artifact(str(metadata_file), artifact_path="data_metadata")
        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    logger.success(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
