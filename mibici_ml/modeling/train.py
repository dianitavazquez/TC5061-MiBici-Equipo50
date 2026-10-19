"""Train and track the P1, P2, and P3 MiBici models."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import joblib
from loguru import logger
import mlflow
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
    CLUSTERING_MANIFEST_FILE,
    CLUSTERING_MIN_ACTIVE_DAYS,
    CLUSTERING_SEED,
    CLUSTERING_START_DATE,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    P1_COMPARISON_FILE,
    P1_FEATURES_FILE,
    P1_MANIFEST_FILE,
    P1_MODEL_FILE,
    P1_TRIPS_FILE,
    P2_ALPHA,
    P2_END_DATE,
    P2_START_DATE,
    P2_STATIONS_FILE,
    P2_TRAIN_FRACTION,
    RAW_DATA_DIR,
)
from mibici_ml.dataset import read_csv_compatible
from mibici_ml.features import build_station_hour_dataset, load_clean_trips
from mibici_ml.plots import (
    plot_cluster_map,
    plot_cluster_profiles,
    save_confusion_matrices,
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
    date_start: str = P2_START_DATE,
    date_end: str = P2_END_DATE,
    alpha: float = P2_ALPHA,
    train_fraction: float = P2_TRAIN_FRACTION,
    experiment_name: str = MLFLOW_EXPERIMENT,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Train both P2 proxy-risk labels and record the run in MLflow."""
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

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    model = build_pipeline()
    logger.info(f"Entrenando con {len(train):,} filas; evaluando con {len(test):,}")
    model.fit(train[features], train[TARGETS])
    predictions = model.predict(test[features])
    probabilities = model.predict_proba(test[features])
    metrics = evaluate(test[TARGETS], predictions, probabilities)

    artifact_path = FIGURES_DIR / "p2_confusion_matrices.png"
    save_confusion_matrices(test[TARGETS], predictions, TARGETS, artifact_path)
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


@app.command("train-p1")
def train_p1(
    features_path: Path = P1_FEATURES_FILE,
    manifest_path: Path = P1_MANIFEST_FILE,
    model_path: Path = P1_MODEL_FILE,
    test_fraction: float = 0.2,
    tracking_uri: str = MLFLOW_TRACKING_URI,
):
    """Train and compare a baseline and a candidate regression model."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

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
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
    )

    results = {}

    models = {
        "baseline_median": DummyRegressor(strategy="median"),
        "hist_gradient_boosting": candidate,
    }

    for model_name, model in models.items():
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
                    "random_state": 42,
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

            if model_name == "hist_gradient_boosting":

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
    comparison.to_csv(comparison_path, index=False)

    logger.info("\n{}", comparison.to_string(index=False))
    logger.success("Comparison saved to {}", comparison_path)
    logger.success("Candidate model saved to {}", model_path)


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
P3_N_INIT = 10


def build_p3_preprocessor() -> Pipeline:
    """Log-transform trips per day and standardize all P3 variables."""
    log_trips = ColumnTransformer(
        [("log_viajes", FunctionTransformer(np.log1p), ["viajes_dia"])],
        remainder="passthrough",
    )
    return Pipeline([("log_viajes", log_trips), ("escala", StandardScaler())])


def build_p3_model(k: int) -> Pipeline:
    """Build the preprocessing and K-Means P3 pipeline."""
    kmeans = KMeans(n_clusters=k, n_init=P3_N_INIT, random_state=CLUSTERING_SEED)
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
) -> None:
    variables = P3_MODELS[model_name]
    pipeline = build_p3_model(k)
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
                "n_init": P3_N_INIT,
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


def train_p3(models: tuple[str, ...] = ("baseline", "candidato"), ks: tuple[int, ...] = (4,)):
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
            train_and_log_cluster_model(profile, x_eval, model_name, k, manifest["version_datos"])
    logger.success(f"Runs P3 registrados en {MLFLOW_EXPERIMENT}")


@app.command("train-p3")
def train_p3_command(
    model: list[str] = typer.Option(["baseline", "candidato"], help="baseline o candidato"),
    k: list[int] = typer.Option([4], help="Número de grupos; se puede repetir"),
) -> None:
    """Train the P3 clustering models."""
    train_p3(tuple(model), tuple(k))


@app.command("all")
def train_all(prepare_data: bool = True) -> None:
    """Prepare data, build features, and train all three project models."""
    if prepare_data:
        from mibici_ml.dataset import ingest, prepare_p3
        from mibici_ml.features import build_p1_command, build_p3_command

        logger.info("Preparando dataset y features de P1")
        ingest(RAW_DATA_DIR, P1_TRIPS_FILE, P1_MANIFEST_FILE)
        build_p1_command(P1_TRIPS_FILE, P1_FEATURES_FILE)
        logger.info("Preparando dataset y features de P3")
        prepare_p3()
        build_p3_command(
            output_path=CLUSTERING_FEATURES_FILE,
        )
    logger.info("Entrenando P1: regresión")
    train_p1(P1_FEATURES_FILE, P1_MANIFEST_FILE, P1_MODEL_FILE)
    logger.info("Entrenando P2: clasificación logística")
    train_p2_baseline()
    logger.info("Entrenando P3: clustering")
    train_p3()
    logger.success("Pipeline P1 + P2 + P3 completado")


if __name__ == "__main__":
    app()
