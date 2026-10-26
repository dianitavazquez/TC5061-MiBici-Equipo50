"""Train and track the P1, P2, and P3 MiBici models."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import joblib
from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import typer

from mibici_ml.config import (
    CLUSTERING_END_DATE,
    CLUSTERING_FEATURES_FILE,
    CLUSTERING_K,
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_N_INIT,
    CLUSTERING_SEED,
    CLUSTERING_START_DATE,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    P1_COMPARISON_FILE,
    P1_FEATURES_FILE,
    P1_L2_REGULARIZATION,
    P1_LEARNING_RATE,
    P1_MANIFEST_FILE,
    P1_MAX_ITER,
    P1_MAX_LEAF_NODES,
    P1_MODEL_FILE,
    P1_TEST_FRACTION,
    P1_TRIPS_FILE,
    P2_ALPHA,
    P2_END_DATE,
    P2_MAX_ITER,
    P2_REGISTERED_MODEL_DESCRIPTION,
    P2_REGISTERED_MODEL_NAME,
    P2_SEARCH_MODELS,
    P2_START_DATE,
    P2_STATIONS_FILE,
    P2_TRAIN_FRACTION,
    PROJ_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
)
from mibici_ml.dataset import read_csv_compatible
from mibici_ml.features import build_station_hour_dataset, load_clean_trips
from mibici_ml.plots import (
    plot_cluster_map,
    plot_cluster_profiles,
    save_confusion_matrices,
    save_precision_recall_curves,
    save_regression_scatter,
)

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
P2_MODELS = {
    "baseline": {"class_weight": None, "c": 1.0},
    **P2_SEARCH_MODELS,
}
P2_MODEL_ALIASES = {"candidato": "candidato-1"}


@app.callback()
def main():
    """Train MiBici models and register their experiments."""


def chronological_split(frame: pd.DataFrame, train_fraction: float = P2_TRAIN_FRACTION):
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


def build_pipeline(
    class_weight: str | dict | None = None, c: float = 1.0, max_iter: int = P2_MAX_ITER
) -> Pipeline:
    """Build the P2 multi-output logistic-regression pipeline."""
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
            max_iter=max_iter,
            class_weight=class_weight,
            C=c,
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


def prepare_p2_data(
    raw_dir: Path = RAW_DATA_DIR,
    date_start: str = P2_START_DATE,
    date_end: str = P2_END_DATE,
    alpha: float = P2_ALPHA,
    train_fraction: float = P2_TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, dict[str, int], pd.DataFrame]:
    """Build P2 once so every candidate uses exactly the same chronological split."""
    logger.info("Cargando y limpiando viajes")
    trips, quality_counts = load_clean_trips(raw_dir)
    stations = read_csv_compatible(raw_dir / P2_STATIONS_FILE)
    logger.info("Construyendo panel estación-hora y variables rezagadas")
    frame = build_station_hour_dataset(trips, stations, alpha=alpha)
    start_timestamp = pd.Timestamp(date_start)
    end_timestamp = pd.Timestamp(date_end)
    if start_timestamp >= end_timestamp:
        raise ValueError("date_start debe ser anterior a date_end.")
    frame = frame[frame["hour"].ge(start_timestamp) & frame["hour"].lt(end_timestamp)].copy()
    train, test, cutoff = chronological_split(frame, train_fraction)
    return train, test, cutoff, quality_counts, frame


def evaluate_monthly_stability(
    test: pd.DataFrame, predictions: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    """Summarize monthly F1 variability on the untouched chronological test partition."""
    monthly_rows = []
    months = test["hour"].dt.to_period("M").astype(str)
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        target_scores = [
            f1_score(test.loc[mask, target], predictions[mask, index], zero_division=0)
            for index, target in enumerate(TARGETS)
        ]
        monthly_rows.append(
            {
                "month": month,
                **dict(zip(TARGETS, target_scores)),
                "macro_f1": np.mean(target_scores),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    stability = {
        "monthly_macro_f1_mean": float(monthly["macro_f1"].mean()),
        "monthly_macro_f1_std": float(monthly["macro_f1"].std(ddof=0)),
        "monthly_macro_f1_min": float(monthly["macro_f1"].min()),
    }
    return stability, monthly


def p2_run_config(
    model_name: str,
    model_config: dict,
    cutoff: pd.Timestamp,
    frame: pd.DataFrame,
    train_rows: int,
    test_rows: int,
    quality_counts: dict[str, int],
    raw_dir: Path,
) -> dict:
    """Return the complete, JSON-serializable configuration logged for a P2 run."""
    return {
        "question": "P2",
        "model_name": model_name,
        "model_family": "multioutput_logistic_regression",
        "class_weight": model_config["class_weight"],
        "C": model_config["c"],
        "max_iter": model_config["max_iter"],
        "penalty": "l2",
        "solver": "lbfgs",
        "alpha": P2_ALPHA,
        "horizon_hours": 1,
        "date_start_inclusive": P2_START_DATE,
        "date_end_exclusive": P2_END_DATE,
        "train_fraction": P2_TRAIN_FRACTION,
        "split_cutoff": cutoff.isoformat(),
        "data_start": frame["hour"].min().isoformat(),
        "data_end": frame["hour"].max().isoformat(),
        "data_version": read_data_version(raw_dir),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "targets": TARGETS,
        "train_rows": train_rows,
        "test_rows": test_rows,
        **quality_counts,
    }


def train_and_log_p2_model(
    model_name: str,
    model_config: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cutoff: pd.Timestamp,
    quality_counts: dict[str, int],
    frame: pd.DataFrame,
    training_dataset,
    raw_dir: Path = RAW_DATA_DIR,
    nested: bool = False,
) -> dict:
    """Fit, evaluate, and log one fully reproducible P2 model."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    run_name = f"p2-{model_name}-logistic-regression"
    model = build_pipeline(**model_config)
    logger.info(
        "Entrenando {} con {:,} filas; evaluando con {:,}",
        run_name,
        len(train),
        len(test),
    )
    fit_started = time.perf_counter()
    model.fit(train[features], train[TARGETS])
    fit_seconds = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    predictions = model.predict(test[features])
    probabilities = model.predict_proba(test[features])
    inference_seconds = time.perf_counter() - inference_started
    metrics = evaluate(test[TARGETS], predictions, probabilities)
    stability_metrics, monthly = evaluate_monthly_stability(test, predictions)
    metrics.update(stability_metrics)
    metrics.update(
        {
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "inference_rows_per_second": len(test) / inference_seconds,
        }
    )

    plot_dir = FIGURES_DIR / "p2_search"
    confusion_path = plot_dir / f"{run_name}_confusion_matrices.png"
    pr_path = plot_dir / f"{run_name}_precision_recall.png"
    save_confusion_matrices(test[TARGETS], predictions, TARGETS, confusion_path)
    save_precision_recall_curves(test[TARGETS], probabilities, TARGETS, pr_path)
    resolved_config = p2_run_config(
        model_name,
        model_config,
        cutoff,
        frame,
        len(train),
        len(test),
        quality_counts,
        raw_dir,
    )
    stage = "baseline" if model_name == "baseline" else "candidate"

    with mlflow.start_run(run_name=run_name, nested=nested) as active_run:
        mlflow.set_tags(
            {
                "mlflow.runName": run_name,
                "team": "50",
                "question": "P2",
                "task": "classification",
                "model_family": "logistic_regression",
                "stage_of_work": stage,
                "variant": model_name,
                "git_commit": git_commit(),
                "target_type": "demand_pressure_proxy",
                "contains_personal_data": "false",
            }
        )
        mlflow.log_params(
            {
                key: (
                    json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                )
                for key, value in resolved_config.items()
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_input(training_dataset, context="training")
        mlflow.log_artifact(str(confusion_path), artifact_path="plots")
        mlflow.log_artifact(str(pr_path), artifact_path="plots")
        mlflow.log_text(monthly.to_csv(index=False), "plots/monthly_stability.csv")
        for metadata_file in [raw_dir / "manifest.json", raw_dir / "profile.json"]:
            if metadata_file.exists():
                mlflow.log_artifact(str(metadata_file), artifact_path="data")
        mlflow.log_dict(resolved_config, "config/resolved_training_config.json")
        for config_file in [
            PROJ_ROOT / "MLproject",
            PROJ_ROOT / "python_env.yaml",
            PROJ_ROOT / "requirements.txt",
        ]:
            mlflow.log_artifact(str(config_file), artifact_path="config")

        input_example = train[features].head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    logger.success("{}: {}", run_name, json.dumps(metrics))
    return {
        "run_id": active_run.info.run_id,
        "logged_model_id": model_info.model_id,
        "model": model_name,
        "stage_of_work": stage,
        **metrics,
    }


def create_p2_training_dataset(train: pd.DataFrame, raw_dir: Path = RAW_DATA_DIR):
    """Create aggregate station-hour lineage metadata without logging raw trip records."""
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGETS
    return mlflow.data.from_pandas(
        train[columns],
        source=(raw_dir / "manifest.json").resolve().as_uri(),
        name=f"p2_station_hour_{read_data_version(raw_dir)}",
    )


def train_p2(
    raw_dir: Path = RAW_DATA_DIR,
    date_start: str = P2_START_DATE,
    date_end: str = P2_END_DATE,
    alpha: float = P2_ALPHA,
    train_fraction: float = P2_TRAIN_FRACTION,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    models: tuple[str, ...] = ("baseline", "candidato-1"),
    model_params: dict | None = None,
) -> dict[str, dict]:
    """Train selected P2 configurations on the same chronological split."""
    models = tuple(P2_MODEL_ALIASES.get(name, name) for name in models)
    unknown = set(models) - set(P2_MODELS)
    if unknown:
        raise ValueError(f"Modelos P2 no válidos: {unknown}. Usa {list(P2_MODELS)}")

    train, test, cutoff, quality_counts, frame = prepare_p2_data(
        raw_dir, date_start, date_end, alpha, train_fraction
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    training_dataset = create_p2_training_dataset(train, raw_dir)
    results = {}
    for model_name in models:
        config = {**P2_MODELS[model_name], "max_iter": P2_MAX_ITER}
        config.update(
            {key: value for key, value in (model_params or {}).items() if value is not None}
        )
        results[model_name] = train_and_log_p2_model(
            model_name,
            config,
            train,
            test,
            cutoff,
            quality_counts,
            frame,
            training_dataset,
            raw_dir,
        )
    return results


def search_p2(
    raw_dir: Path = RAW_DATA_DIR,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> dict:
    """Run baseline plus three P2 candidates as children of one organized search."""
    train, test, cutoff, quality_counts, frame = prepare_p2_data(raw_dir)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    training_dataset = create_p2_training_dataset(train, raw_dir)
    search_name = "p2-logistic-class-weight-search"
    variants = ("baseline", *P2_SEARCH_MODELS)
    results = []

    with mlflow.start_run(run_name=search_name) as parent_run:
        mlflow.set_tags(
            {
                "mlflow.runName": search_name,
                "team": "50",
                "question": "P2",
                "model_family": "logistic_regression",
                "stage_of_work": "search",
                "git_commit": git_commit(),
            }
        )
        mlflow.log_dict(
            {"baseline": P2_MODELS["baseline"], "candidates": P2_SEARCH_MODELS},
            "config/search_space.json",
        )
        for model_name in variants:
            model_config = {**P2_MODELS[model_name], "max_iter": P2_MAX_ITER}
            results.append(
                train_and_log_p2_model(
                    model_name,
                    model_config,
                    train,
                    test,
                    cutoff,
                    quality_counts,
                    frame,
                    training_dataset,
                    raw_dir,
                    nested=True,
                )
            )

        comparison = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
        winner = comparison.iloc[0]
        baseline = comparison[comparison["model"] == "baseline"].iloc[0]
        improvement = winner["macro_f1"] - baseline["macro_f1"]
        decision = (
            f"# Selección P2\n\n"
            f"Se selecciona **{winner['model']}** por obtener el mayor macro-F1 "
            f"({winner['macro_f1']:.6f}), una diferencia de {improvement:+.6f} frente al "
            f"baseline. Su desviación mensual de macro-F1 fue "
            f"{winner['monthly_macro_f1_std']:.6f}; el ajuste tomó "
            f"{winner['fit_seconds']:.2f} s y la inferencia {winner['inference_seconds']:.2f} s. "
            "Macro-F1 es la métrica principal porque equilibra riesgos omitidos y falsas "
            "alarmas para el equipo de rebalanceo.\n"
        )
        mlflow.log_text(comparison.to_csv(index=False), "selection/model_comparison.csv")
        mlflow.log_text(decision, "selection/decision.md")
        mlflow.set_tags(
            {
                "selected_model": winner["model"],
                "selected_run_id": winner["run_id"],
                "selection_metric": "macro_f1",
            }
        )
        client = mlflow.tracking.MlflowClient()
        client.set_tag(winner["run_id"], "selected_for_production", "true")
        registered_version = mlflow.register_model(
            f"models:/{winner['logged_model_id']}", P2_REGISTERED_MODEL_NAME
        )
        client.set_registered_model_alias(
            P2_REGISTERED_MODEL_NAME, "champion", registered_version.version
        )
        source_run = client.get_run(winner["run_id"])
        source_experiment = client.get_experiment(source_run.info.experiment_id)
        lineage = {
            "experiment_id": source_experiment.experiment_id,
            "experiment_name": source_experiment.name,
            "source_run_id": winner["run_id"],
            "git_commit": source_run.data.tags["git_commit"],
            "data_version": source_run.data.params["data_version"],
            "parent_search_run_id": parent_run.info.run_id,
            "question": "P2",
            "stage": "production",
        }
        client.update_registered_model(
            P2_REGISTERED_MODEL_NAME, description=P2_REGISTERED_MODEL_DESCRIPTION
        )
        client.update_model_version(
            P2_REGISTERED_MODEL_NAME,
            registered_version.version,
            description=(
                f"Modelo {winner['model']} seleccionado por macro-F1="
                f"{winner['macro_f1']:.6f}. Origen: experimento "
                f"{source_experiment.name}, corrida {winner['run_id']}, commit "
                f"{lineage['git_commit']} y datos {lineage['data_version']}."
            ),
        )
        for key, value in lineage.items():
            client.set_model_version_tag(
                P2_REGISTERED_MODEL_NAME, registered_version.version, key, str(value)
            )
        mlflow.set_tags(
            {
                "registered_model": P2_REGISTERED_MODEL_NAME,
                "registered_model_version": registered_version.version,
                "registered_model_alias": "champion",
            }
        )

    return {
        "parent_run_id": parent_run.info.run_id,
        "selected_model": winner["model"],
        "selected_run_id": winner["run_id"],
        "comparison": comparison,
    }


@app.command("train-p2")
def train_p2_command(
    model: list[str] = typer.Option(
        ["baseline", "candidato-1"], help="Configuración P2; se puede repetir"
    ),
) -> None:
    """Train the P2 baseline and candidate configurations."""
    train_p2(models=tuple(model))


@app.command("search-p2")
def search_p2_command() -> None:
    """Run the nested baseline-versus-three-candidates P2 search."""
    search_p2()


FEATURE_COLUMNS = [
    "station_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "departures_lag_1",
    "departures_lag_24",
    "departures_lag_168",
    "departures_mean_24",
    "departures_mean_168",
]


def evaluate_model(model, X_test, y_test):
    """Evaluate a regression model using MAE and RMSE."""
    predictions = model.predict(X_test)
    # Demand cannot be negative.
    predictions = predictions.clip(min=0)

    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }, predictions


def train_p1(
    features_path: Path = P1_FEATURES_FILE,
    manifest_path: Path = P1_MANIFEST_FILE,
    model_path: Path = P1_MODEL_FILE,
    test_fraction: float = P1_TEST_FRACTION,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    models: tuple[str, ...] = ("baseline", "candidato"),
    max_iter: int = P1_MAX_ITER,
    learning_rate: float = P1_LEARNING_RATE,
    max_leaf_nodes: int = P1_MAX_LEAF_NODES,
    l2_regularization: float = P1_L2_REGULARIZATION,
):
    """Train and compare a baseline and a candidate regression model."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    valid_models = {"baseline", "candidato"}
    unknown = set(models) - valid_models
    if unknown:
        raise ValueError(f"Modelos P1 no válidos: {unknown}. Usa {sorted(valid_models)}")

    logger.info("Reading features from {}", features_path)
    df = pd.read_parquet(features_path)

    required = set(FEATURE_COLUMNS + ["timestamp", "target_departures"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[FEATURE_COLUMNS + ["target_departures"]].isna().any().any():
        raise ValueError("Features or target contain missing values.")

    # Use a chronological split based on timestamps, not random rows.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    timestamps = sorted(df["timestamp"].unique())

    split_index = int(len(timestamps) * (1 - test_fraction))
    if split_index <= 0 or split_index >= len(timestamps):
        raise ValueError("Not enough timestamps for the requested split.")

    cutoff = pd.Timestamp(timestamps[split_index])

    train_df = df[df["timestamp"] < cutoff]
    test_df = df[df["timestamp"] >= cutoff]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target_departures"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target_departures"]

    if train_df.empty or test_df.empty:
        raise ValueError("The chronological split produced an empty partition.")

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    dataset_version = manifest.get("dataset_sha256", "unknown")
    commit = git_commit()

    logger.info(
        "Train rows: {}; test rows: {}; cutoff: {}",
        f"{len(train_df):,}",
        f"{len(test_df):,}",
        cutoff,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Candidate model: histogram-based gradient boosting.
    candidate = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )

    results = {}

    configured_models = {
        "baseline": ("p1-baseline-median", DummyRegressor(strategy="median")),
        "candidato": ("p1-candidato-hist-gradient-boosting", candidate),
    }

    for variant in models:
        model_name, model = configured_models[variant]
        logger.info("Training model: {}", model_name)

        with mlflow.start_run(run_name=model_name):
            mlflow.set_tags(
                {
                    "team": "50",
                    "question": "P1",
                    "task": "regression",
                    "git_commit": commit,
                    "dataset_version": dataset_version,
                }
            )

            mlflow.log_params(
                {
                    "model_name": model_name,
                    "dataset_version": dataset_version,
                    "features_file": features_path.name,
                    "train_start": str(train_df["timestamp"].min()),
                    "train_end": str(train_df["timestamp"].max()),
                    "test_start": str(test_df["timestamp"].min()),
                    "test_end": str(test_df["timestamp"].max()),
                    "split_strategy": "chronological",
                    "test_fraction": test_fraction,
                    "random_state": RANDOM_SEED,
                    "n_features": len(FEATURE_COLUMNS),
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                }
            )

            model.fit(X_train, y_train)
            metrics, predictions = evaluate_model(model, X_test, y_test)

            mlflow.log_metrics(
                {
                    "test_mae": metrics["mae"],
                    "test_rmse": metrics["rmse"],
                }
            )

            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "predictions": predictions,
            }

            # Log a sample plot to compare observed and predicted demand.
            plot_path = Path("reports") / "figures" / f"{model_name}_p1.png"
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            save_regression_scatter(y_test, predictions, model_name, plot_path)

            mlflow.log_artifact(str(plot_path), artifact_path="plots")
            mlflow.log_artifact(str(manifest_path), artifact_path="data_metadata")

            if model_name == "p1-candidato-hist-gradient-boosting":

                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    serialization_format="cloudpickle",
                )

                model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_path)

                mlflow.log_params(
                    {
                        "max_iter": model.max_iter,
                        "learning_rate": model.learning_rate,
                        "max_leaf_nodes": model.max_leaf_nodes,
                        "l2_regularization": model.l2_regularization,
                    }
                )

            logger.info(
                "{}: MAE={:.4f}, RMSE={:.4f}",
                model_name,
                metrics["mae"],
                metrics["rmse"],
            )

    comparison = pd.DataFrame(
        [{"model": name, **result["metrics"]} for name, result in results.items()]
    )

    comparison_path = P1_COMPARISON_FILE
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    logger.info("\n{}", comparison.to_string(index=False))
    logger.success("Comparison saved to {}", comparison_path)
    if "candidato" in models:
        logger.success("Candidate model saved to {}", model_path)


@app.command("train-p1")
def train_p1_command(
    model: list[str] = typer.Option(
        ["baseline", "candidato"], help="Configuración P1; se puede repetir"
    ),
) -> None:
    """Train the selected P1 baseline or candidate configurations."""
    train_p1(models=tuple(model))


P3_USAGE_FEATURES = [
    "viajes_dia",
    "balance",
    "prop_manana",
    "prop_mediodia",
    "prop_tarde",
    "prop_noche",
    "prop_fin_semana",
    "prop_redondo",
    "duracion_mediana",
]
P3_MODELS = {
    "baseline": ["viajes_dia", "prop_fin_semana"],
    "candidato": P3_USAGE_FEATURES,
}


def build_p3_preprocessor() -> Pipeline:
    """Log-transform trips per day and standardize all P3 variables."""
    log_trips = ColumnTransformer(
        [("log_viajes", FunctionTransformer(np.log1p), ["viajes_dia"])],
        remainder="passthrough",
    )
    return Pipeline([("log_viajes", log_trips), ("escala", StandardScaler())])


def build_p3_model(k: int, n_init: int = CLUSTERING_N_INIT) -> Pipeline:
    """Build the preprocessing and K-Means P3 pipeline."""
    kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=CLUSTERING_SEED)
    return Pipeline([("preprocesamiento", build_p3_preprocessor()), ("kmeans", kmeans)])


def evaluate_clustering(x_eval: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": silhouette_score(x_eval, groups),
        "davies_bouldin": davies_bouldin_score(x_eval, groups),
        "calinski_harabasz": calinski_harabasz_score(x_eval, groups),
        "grupo_mas_chico_pct": np.bincount(groups).min() / len(groups) * 100,
    }


def summarize_clusters(profile: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    grouped = profile.assign(grupo=groups).groupby("grupo")
    table = grouped[[*P3_USAGE_FEATURES, "capacidad"]].mean().round(3)
    table.insert(0, "n_estaciones", grouped.size())
    return table.reset_index()


def log_cluster_figures(
    run_name: str, assignments: pd.DataFrame, x_eval: np.ndarray, groups: np.ndarray
) -> None:
    figures = {
        "mapa_grupos.png": plot_cluster_map(assignments),
        "perfil_grupos.png": plot_cluster_profiles(x_eval, groups, P3_USAGE_FEATURES),
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        mlflow.log_figure(figure, name)
        figure.savefig(FIGURES_DIR / f"{run_name}_{name}")


def train_and_log_cluster_model(
    profile: pd.DataFrame,
    x_eval: np.ndarray,
    model_name: str,
    k: int,
    data_version: str,
    n_init: int = CLUSTERING_N_INIT,
) -> None:
    variables = P3_MODELS[model_name]
    pipeline = build_p3_model(k, n_init=n_init)
    groups = pipeline.fit_predict(profile[variables])
    metrics = evaluate_clustering(x_eval, groups)
    run_name = f"p3-{model_name}-kmeans-k{k}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {"team": "50", "question": "P3", "task": "clustering", "git_commit": git_commit()}
        )
        mlflow.log_params(
            {
                "modelo": model_name,
                "algoritmo": "KMeans",
                "k": k,
                "n_init": n_init,
                "semilla": CLUSTERING_SEED,
                "variables": ",".join(variables),
                "fecha_inicio": CLUSTERING_START_DATE,
                "fecha_fin": CLUSTERING_END_DATE,
                "version_datos": data_version,
                "min_dias_activos": CLUSTERING_MIN_ACTIVE_DAYS,
                "n_estaciones": len(profile),
            }
        )
        mlflow.log_metrics(metrics)
        columns = ["estacion_id", "nombre", "zona", "latitud", "longitud"]
        assignments = profile[columns].assign(grupo=groups)
        mlflow.log_text(assignments.to_csv(index=False), "grupos_estaciones.csv")
        mlflow.log_text(
            summarize_clusters(profile, groups).to_csv(index=False), "perfil_grupos.csv"
        )
        mlflow.log_artifact(str(CLUSTERING_MANIFEST_FILE))
        log_cluster_figures(run_name, assignments, x_eval, groups)
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            input_example=profile[variables].head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    logger.info(
        "{}: {}", run_name, ", ".join(f"{name} {value:.3f}" for name, value in metrics.items())
    )


def train_p3(
    models: tuple[str, ...] = ("baseline", "candidato"),
    ks: tuple[int, ...] = (CLUSTERING_K,),
    n_init: int = CLUSTERING_N_INIT,
):
    """Train and log each requested P3 model/k combination."""
    unknown = set(models) - set(P3_MODELS)
    if unknown:
        raise ValueError(f"Modelos no válidos: {unknown}. Usa {list(P3_MODELS)}")
    if not CLUSTERING_FEATURES_FILE.exists():
        raise FileNotFoundError(f"No existe {CLUSTERING_FEATURES_FILE}. Corre features build-p3.")
    profile = pd.read_parquet(CLUSTERING_FEATURES_FILE)
    manifest = json.loads(CLUSTERING_MANIFEST_FILE.read_text(encoding="utf-8"))
    x_eval = build_p3_preprocessor().fit_transform(profile[P3_USAGE_FEATURES])
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    for model_name in models:
        for k in ks:
            train_and_log_cluster_model(
                profile, x_eval, model_name, k, manifest["version_datos"], n_init=n_init
            )
    logger.success(f"Runs P3 registrados en {MLFLOW_EXPERIMENT}")


@app.command("train-p3")
def train_p3_command(
    model: list[str] = typer.Option(["baseline", "candidato"], help="baseline o candidato"),
    k: list[int] = typer.Option([CLUSTERING_K], help="Número de grupos; se puede repetir"),
) -> None:
    """Train the P3 clustering models."""
    train_p3(tuple(model), tuple(k))


@app.command("all")
def train_all(prepare_data: bool = True) -> None:
    """Prepare data, build features, and train all three project models."""
    if prepare_data:
        from mibici_ml.dataset import prepare_all_data
        from mibici_ml.features import build_p1_command, build_p3_command

        logger.info("Descargando y preparando los datasets compartidos")
        prepare_all_data()
        logger.info("Construyendo features de P1")
        build_p1_command(P1_TRIPS_FILE, P1_FEATURES_FILE)
        logger.info("Construyendo features de P3")
        build_p3_command(
            output_path=CLUSTERING_FEATURES_FILE,
        )
    logger.info("Entrenando P1: regresión")
    train_p1(P1_FEATURES_FILE, P1_MANIFEST_FILE, P1_MODEL_FILE)
    logger.info("Entrenando P2: clasificación logística")
    train_p2()
    logger.info("Entrenando P3: clustering")
    train_p3()
    logger.success("Pipeline P1 + P2 + P3 completado")


if __name__ == "__main__":
    app()
